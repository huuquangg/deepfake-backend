"""
Batch Frequency-Domain Feature Extraction API
Extracts FFT, DCT, and SRM features from up to 30 frames
Returns CSV-style data matching the OpenFace API format
"""
import os
import sys
import logging
import glob
import tempfile
import shutil
import math
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import cv2
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Batch Frequency-Domain Feature Extraction API",
    description="Extract frequency domain features (FFT, DCT, SRM) from up to 30 frames",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
TEMP_DIR = os.getenv("TEMP_DIR", "/tmp/frequency_batch")
BATCH_SIZE = 30
TARGET_SIZE = (256, 256)
CLAMP_T = 3.0
ENTROPY_BINS = 41
EPS = 1e-8

# FFT Configuration
LOW_CUT = 0.10
MID_CUT = 0.30
DC_RADIUS_PX = 2
HF_RANGE = (0.15, 1.00)
TOPK_PEAKS = 3
JPEG_PERIOD = 8
SMOOTH_K = 5
APS_BINS = 12
RPS_BINS = 64

# Ensure directories exist
os.makedirs(TEMP_DIR, exist_ok=True)


# ==================== SRM KERNELS ====================
def get_srm_kernels() -> List[np.ndarray]:
    """Get 20 SRM high-pass filter kernels"""
    k = []
    # Laplacian & High-pass
    k.append(np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32))
    k.append(np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32))
    k.append(np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32))
    
    # Second Derivative
    k.append(np.array([[0, 0, 0], [1, -2, 1], [0, 0, 0]], dtype=np.float32))
    k.append(np.array([[0, 1, 0], [0, -2, 0], [0, 1, 0]], dtype=np.float32))
    k.append(np.array([[1, 0, 0], [0, -2, 0], [0, 0, 1]], dtype=np.float32))
    k.append(np.array([[0, 0, 1], [0, -2, 0], [1, 0, 0]], dtype=np.float32))
    
    # Sobel / Scharr
    k.append(np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32))
    k.append(np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32))
    k.append(np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], dtype=np.float32))
    k.append(np.array([[-3, -10, -3], [0, 0, 0], [3, 10, 3]], dtype=np.float32))
    
    # Predictive Residual
    k.append(np.array([[0, 0, 0], [1, -1, 0], [0, 0, 0]], dtype=np.float32))
    k.append(np.array([[0, 0, 0], [0, -1, 1], [0, 0, 0]], dtype=np.float32))
    k.append(np.array([[0, 1, 0], [0, -1, 0], [0, 0, 0]], dtype=np.float32))
    k.append(np.array([[0, 0, 0], [0, -1, 0], [0, 1, 0]], dtype=np.float32))
    k.append(np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]], dtype=np.float32))
    k.append(np.array([[0, 0, 1], [0, -1, 0], [0, 0, 0]], dtype=np.float32))
    k.append(np.array([[0, 0, 0], [0, -1, 0], [1, 0, 0]], dtype=np.float32))
    k.append(np.array([[0, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float32))
    
    # LoG
    k.append(np.array([[0, 0, -1, 0, 0], [0, -1, -2, -1, 0], [-1, -2, 16, -2, -1],
                       [0, -1, -2, -1, 0], [0, 0, -1, 0, 0]], dtype=np.float32))
    
    return k


# ==================== SRM FEATURE EXTRACTION ====================
def clamp_residual(residual: np.ndarray, t: float = CLAMP_T) -> np.ndarray:
    return np.clip(residual, -t, t)


def moments_stats(x: np.ndarray) -> Tuple[float, float, float, float]:
    """Calculate mean, variance, skewness, kurtosis"""
    x = x.astype(np.float64)
    mu = x.mean()
    var = x.var()
    std = math.sqrt(var) if var > 0 else 0.0
    if std == 0:
        return float(mu), float(var), 0.0, -3.0
    
    x_centered = x - mu
    m3 = np.mean(x_centered**3)
    m4 = np.mean(x_centered**4)
    skew = m3 / (std**3)
    kurt = m4 / (std**4) - 3.0
    return float(mu), float(var), float(skew), float(kurt)


def shannon_entropy(x: np.ndarray, bins: int = ENTROPY_BINS, clamp: float = CLAMP_T) -> float:
    """Calculate Shannon entropy"""
    hist, _ = np.histogram(x, bins=bins, range=(-clamp, clamp), density=False)
    total = hist.sum()
    if total == 0:
        return 0.0
    p = hist.astype(np.float64) / total
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def energy(x: np.ndarray) -> float:
    """Calculate energy (mean of squares)"""
    return float(np.mean(x.astype(np.float64)**2))


def extract_srm_features(img_gray: np.ndarray, kernels: List[np.ndarray]) -> Dict[str, float]:
    """Extract SRM features using all kernels"""
    features = {}
    for i, ker in enumerate(kernels, 1):
        residual = cv2.filter2D(img_gray, ddepth=cv2.CV_32F, kernel=ker, 
                               anchor=(-1, -1), delta=0, borderType=cv2.BORDER_REFLECT)
        residual = clamp_residual(residual, CLAMP_T)
        
        mu, var, skew, kurt = moments_stats(residual)
        ent = shannon_entropy(residual, bins=ENTROPY_BINS, clamp=CLAMP_T)
        eng = energy(residual)
        
        features[f"SRM_mean_{i}"] = mu
        features[f"SRM_var_{i}"] = var
        features[f"SRM_skew_{i}"] = skew
        features[f"SRM_kurt_{i}"] = kurt
        features[f"SRM_entropy_{i}"] = ent
        features[f"SRM_energy_{i}"] = eng
    
    return features


# ==================== DCT FEATURE EXTRACTION ====================
def extract_dct_features(img_gray: np.ndarray) -> Dict[str, float]:
    """Extract DCT features"""
    # Apply DCT
    dct = cv2.dct(img_gray.astype(np.float32))
    
    h, w = dct.shape
    cy, cx = h // 2, w // 2
    
    # Define frequency bands
    r_max = min(cy, cx)
    r_low = int(r_max * 0.33)
    r_mid = int(r_max * 0.67)
    
    # Create masks for low, mid, high frequencies
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((y - cy)**2 + (x - cx)**2)
    
    mask_low = dist <= r_low
    mask_mid = (dist > r_low) & (dist <= r_mid)
    mask_high = dist > r_mid
    
    # Extract coefficients for each band
    dct_low = dct[mask_low]
    dct_mid = dct[mask_mid]
    dct_high = dct[mask_high]
    
    features = {}
    
    # Band statistics
    for band_name, band_data in [("low", dct_low), ("mid", dct_mid), ("high", dct_high)]:
        mu, var, skew, kurt = moments_stats(band_data)
        features[f"DCT_mean_{band_name}"] = mu
        features[f"DCT_var_{band_name}"] = var
        features[f"DCT_skew_{band_name}"] = skew
        features[f"DCT_kurt_{band_name}"] = kurt
        features[f"DCT_entropy_{band_name}"] = shannon_entropy(band_data, bins=50, clamp=100)
    
    # Total energy
    features["DCT_energy_total"] = float(np.sum(dct**2))
    
    # Zigzag pattern (first 20 coefficients)
    zigzag_indices = [
        (0, 0), (0, 1), (1, 0), (2, 0), (1, 1), (0, 2), (0, 3), (1, 2),
        (2, 1), (3, 0), (4, 0), (3, 1), (2, 2), (1, 3), (0, 4), (0, 5),
        (1, 4), (2, 3), (3, 2), (4, 1)
    ]
    for idx, (y, x) in enumerate(zigzag_indices):
        if y < h and x < w:
            features[f"DCT_zigzag_{idx}"] = float(dct[y, x])
        else:
            features[f"DCT_zigzag_{idx}"] = 0.0
    
    # Histogram features (8 bins per band)
    for band_name, band_data in [("low", dct_low), ("mid", dct_mid), ("high", dct_high)]:
        hist, _ = np.histogram(band_data, bins=8, density=True)
        for bin_idx, val in enumerate(hist):
            features[f"DCT_hist_{band_name}_bin_{bin_idx}"] = float(val)
    
    return features


# ==================== FFT FEATURE EXTRACTION ====================
def make_hann(n):
    """Create Hanning window"""
    h1 = np.hanning(n)
    return np.outer(h1, h1).astype(np.float32)


def fft2_power_and_log(x):
    """Compute FFT power spectrum and log"""
    F = np.fft.fft2(x)
    F = np.fft.fftshift(F)
    P = (F.real**2 + F.imag**2)
    P_log = np.log(P + EPS)
    return P, P_log


def coord_grids(n):
    """Generate coordinate grids for FFT analysis"""
    cy = (n-1)/2.0
    cx = (n-1)/2.0
    y, x = np.indices((n, n))
    dy = y - cy
    dx = x - cx
    r = np.sqrt(dx*dx + dy*dy)
    theta = np.mod(np.arctan2(dy, dx), np.pi)
    r_norm = r / (np.sqrt(2)*((n-1)/2.0))
    return r, r_norm, theta


def remove_dc(A, r, dc_radius_px):
    """Remove DC component"""
    out = A.copy()
    out[r <= dc_radius_px] = 0.0
    return out


def band_energies(P_like, r_norm):
    """Calculate energy in different frequency bands"""
    low_mask = (r_norm > 0.0) & (r_norm <= LOW_CUT)
    mid_mask = (r_norm > LOW_CUT) & (r_norm <= MID_CUT)
    high_mask = (r_norm > MID_CUT) & (r_norm <= 1.0 + 1e-6)
    E_low = float(P_like[low_mask].sum())
    E_mid = float(P_like[mid_mask].sum())
    E_high = float(P_like[high_mask].sum())
    return E_low, E_mid, E_high


def radial_stats(P_like, r_norm):
    """Calculate radial statistics"""
    w = P_like
    wsum = float(w.sum()) + EPS
    rc = float((r_norm * w).sum() / wsum)
    var = float((w * (r_norm - rc)**2).sum() / wsum)
    return rc, var


def radial_profile(P_like, r_norm, edges, normalize=True):
    """Calculate radial power profile"""
    r_flat = r_norm.ravel()
    p_flat = P_like.ravel()
    idx = np.clip(np.digitize(r_flat, edges) - 1, 0, len(edges)-2)
    rps = np.bincount(idx, weights=p_flat, minlength=len(edges)-1).astype(np.float64)
    if normalize:
        s = rps.sum()
        if s > 0:
            rps = rps / s
    return rps


def angular_profile(P_like, theta, aps_bins=APS_BINS, normalize=True):
    """Calculate angular power profile"""
    edges = np.linspace(0.0, np.pi, aps_bins+1)
    t_flat = theta.ravel()
    p_flat = P_like.ravel()
    idx = np.clip(np.digitize(t_flat, edges) - 1, 0, len(edges)-2)
    aps = np.bincount(idx, weights=p_flat, minlength=len(edges)-1).astype(np.float64)
    if normalize:
        s = aps.sum()
        if s > 0:
            aps = aps / s
    return aps


def rolloff(P_like, r_norm, perc, rps_bins=RPS_BINS):
    """Calculate spectral rolloff"""
    edges = np.linspace(0.0, 1.0, rps_bins+1)
    rps = radial_profile(P_like, r_norm, edges, normalize=False)
    cumsum = np.cumsum(rps)
    total = cumsum[-1] + EPS
    target = perc * total
    idx = np.searchsorted(cumsum, target)
    idx = min(max(idx, 0), rps_bins-1)
    r_center = 0.5*(edges[idx] + edges[idx+1])
    return float(r_center)


def spectral_flatness(P_like):
    """Calculate spectral flatness"""
    x = np.maximum(P_like.ravel(), EPS)
    amean = float(np.mean(x))
    gmean = float(np.exp(np.mean(np.log(x))))
    return float(gmean / (amean + EPS))


def spectral_entropy(P_like):
    """Calculate spectral entropy"""
    x = np.maximum(P_like.ravel(), EPS)
    q = x / (x.sum() + EPS)
    H = -float((q * np.log(q)).sum())
    H /= math.log(len(q) + EPS)
    return H


def smooth_1d(x, k=SMOOTH_K):
    """Smooth 1D signal"""
    if k <= 1:
        return x
    ker = np.ones(int(k), dtype=np.float64) / float(k)
    return np.convolve(x, ker, mode="same")


def hf_slope_beta(rps_power, hf_range=HF_RANGE):
    """Calculate high-frequency slope"""
    edges = np.linspace(0.0, 1.0, len(rps_power)+1)
    centers = 0.5*(edges[:-1] + edges[1:])
    lo, hi = hf_range
    m = (centers >= lo) & (centers <= hi) & (rps_power > 0)
    if m.sum() < 3:
        return 0.0
    x = np.log(centers[m] + EPS)
    y = np.log(rps_power[m] + EPS)
    A = np.vstack([np.ones_like(x), x]).T
    coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    return float(coef[1])


def topk_peaks(rps_power, k=TOPK_PEAKS, min_index=3, smooth_k=SMOOTH_K):
    """Find top-k peaks in radial power spectrum"""
    rps_sm = smooth_1d(rps_power, k=smooth_k)
    idx_sorted = np.argsort(rps_sm)[::-1]
    picks = []
    for idx in idx_sorted:
        if idx < min_index:
            continue
        if any(abs(idx - p[0]) <= 2 for p in picks):
            continue
        picks.append((idx, rps_sm[idx]))
        if len(picks) >= k:
            break
    
    edges = np.linspace(0.0, 1.0, len(rps_power)+1)
    out = []
    for idx, val in picks:
        r_center = 0.5*(edges[idx] + edges[idx+1])
        out.append((float(r_center), float(val)))
    while len(out) < k:
        out.append((0.0, 0.0))
    return out


def jpeg_8x8_markers(P_log, period=JPEG_PERIOD):
    """Detect JPEG 8x8 block artifacts"""
    n = P_log.shape[0]
    k = int(round(n / period))
    if k < 1:
        return (0.0, 0.0, 0.0)
    cx = cy = (n-1)//2
    
    def patch_mean(u, v, rad=2):
        u = int(u)
        v = int(v)
        u0, u1 = max(0, u-rad), min(n, u+rad+1)
        v0, v1 = max(0, v-rad), min(n, v+rad+1)
        return float(np.mean(P_log[v0:v1, u0:u1]))
    
    pts_x = [(cx+k, cy), (cx-k, cy)]
    pts_y = [(cx, cy+k), (cx, cy-k)]
    pts_d = [(cx+k, cy+k), (cx-k, cy-k), (cx+k, cy-k), (cx-k, cy+k)]
    ex = np.mean([patch_mean(u, v) for (u, v) in pts_x])
    ey = np.mean([patch_mean(u, v) for (u, v) in pts_y])
    ed = np.mean([patch_mean(u, v) for (u, v) in pts_d])
    return (ex, ey, ed)


def extract_fft_features(img_gray: np.ndarray) -> Dict[str, float]:
    """Extract FFT features"""
    n = img_gray.shape[0]
    
    # Apply Hanning window
    hann = make_hann(n)
    img_windowed = img_gray * hann
    
    # Normalize
    img_windowed = img_windowed - img_windowed.mean()
    std = img_windowed.std()
    if std > 1e-6:
        img_windowed = img_windowed / std
    
    # Compute FFT
    P, P_log = fft2_power_and_log(img_windowed)
    r, r_norm, theta = coord_grids(n)
    
    # Remove DC component
    P = remove_dc(P, r, DC_RADIUS_PX)
    P_log = remove_dc(P_log, r, DC_RADIUS_PX)
    P_pos = np.maximum(P, EPS)
    
    features = {}
    
    # Band energies
    E_low, E_mid, E_high = band_energies(P_pos, r_norm)
    features["fft_psd_total"] = float(P_pos.sum())
    features["fft_E_low"] = E_low
    features["fft_E_mid"] = E_mid
    features["fft_E_high"] = E_high
    features["fft_E_high_over_low"] = float(E_high / (E_low + EPS))
    features["fft_E_mid_over_low"] = float(E_mid / (E_low + EPS))
    
    # Radial statistics
    rc, rbw = radial_stats(P_pos, r_norm)
    features["fft_radial_centroid"] = rc
    features["fft_radial_bandwidth"] = rbw
    
    # Rolloff points
    features["fft_rolloff_85"] = rolloff(P_pos, r_norm, 0.85)
    features["fft_rolloff_95"] = rolloff(P_pos, r_norm, 0.95)
    
    # Spectral shape
    features["fft_spectral_flatness"] = spectral_flatness(P_pos)
    features["fft_spectral_entropy"] = spectral_entropy(P_pos)
    
    # Radial and angular profiles
    edges_r = np.linspace(0.0, 1.0, RPS_BINS+1)
    rps = radial_profile(P_pos, r_norm, edges_r, normalize=True)
    aps = angular_profile(P_pos, theta, APS_BINS, normalize=True)
    
    for i, val in enumerate(aps):
        features[f"fft_aps_{i}"] = float(val)
    
    for i, val in enumerate(rps):
        features[f"fft_rps_{i}"] = float(val)
    
    # High-frequency slope
    rps_power = radial_profile(P_pos, r_norm, edges_r, normalize=False)
    features["fft_hf_slope_beta"] = hf_slope_beta(rps_power, HF_RANGE)
    
    # Peak detection
    peaks = topk_peaks(rps_power, k=TOPK_PEAKS, min_index=3, smooth_k=SMOOTH_K)
    for i, (r_val, peak_val) in enumerate(peaks, 1):
        features[f"fft_peak{i}_r"] = r_val
        features[f"fft_peak{i}_val"] = peak_val
    
    # JPEG markers
    ex, ey, ed = jpeg_8x8_markers(P_log, JPEG_PERIOD)
    features["fft_jpeg_8x8_x"] = ex
    features["fft_jpeg_8x8_y"] = ey
    features["fft_jpeg_8x8_diag"] = ed
    
    return features


# ==================== COMBINED EXTRACTION ====================
def extract_all_features(img_path: str) -> Dict[str, Any]:
    """Extract all frequency domain features from an image"""
    try:
        # Read image
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Cannot read image: {img_path}")
        
        h0, w0 = img.shape[:2]
        
        # Convert to grayscale and resize
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, TARGET_SIZE, interpolation=cv2.INTER_AREA)
        gray = gray.astype(np.float32)
        
        # Extract features
        kernels = get_srm_kernels()
        srm_features = extract_srm_features(gray, kernels)
        dct_features = extract_dct_features(gray)
        fft_features = extract_fft_features(gray)
        
        # Combine all features
        all_features = {
            "filename": os.path.basename(img_path),
            **srm_features,
            **dct_features,
            **fft_features,
            "width": w0,
            "height": h0,
            "color_mode": "gray",
            "resize_to": TARGET_SIZE[0],
            "do_hann": 1
        }
        
        return all_features
        
    except Exception as e:
        logger.error(f"Error extracting features from {img_path}: {e}")
        raise


def get_feature_columns() -> List[str]:
    """Get the ordered list of all feature column names (283 total)"""
    cols = ["filename"]
    
    # SRM features: 20 kernels × 6 features = 120
    for i in range(1, 21):
        cols.extend([
            f"SRM_mean_{i}", f"SRM_var_{i}", f"SRM_skew_{i}",
            f"SRM_kurt_{i}", f"SRM_entropy_{i}", f"SRM_energy_{i}"
        ])
    
    # DCT features: 4 band stats × 3 bands + 1 energy + 20 zigzag + 8 bins × 3 bands = 57
    for band in ["low", "mid", "high"]:
        cols.extend([f"DCT_mean_{band}", f"DCT_var_{band}", 
                     f"DCT_skew_{band}", f"DCT_kurt_{band}"])
    for band in ["low", "mid", "high"]:
        cols.append(f"DCT_entropy_{band}")
    cols.append("DCT_energy_total")
    for i in range(20):
        cols.append(f"DCT_zigzag_{i}")
    for band in ["low", "mid", "high"]:
        for bin_idx in range(8):
            cols.append(f"DCT_hist_{band}_bin_{bin_idx}")
    
    # FFT features: 13 global + 12 aps + 64 rps + 6 peaks + 3 jpeg + 5 metadata = 103
    cols.extend([
        "fft_psd_total", "fft_E_low", "fft_E_mid", "fft_E_high",
        "fft_E_high_over_low", "fft_E_mid_over_low",
        "fft_radial_centroid", "fft_radial_bandwidth",
        "fft_rolloff_85", "fft_rolloff_95",
        "fft_spectral_flatness", "fft_spectral_entropy", "fft_hf_slope_beta"
    ])
    for i in range(APS_BINS):
        cols.append(f"fft_aps_{i}")
    for i in range(RPS_BINS):
        cols.append(f"fft_rps_{i}")
    for i in range(1, 4):
        cols.extend([f"fft_peak{i}_r", f"fft_peak{i}_val"])
    cols.extend(["fft_jpeg_8x8_x", "fft_jpeg_8x8_y", "fft_jpeg_8x8_diag"])
    
    # Metadata: 5
    cols.extend(["width", "height", "color_mode", "resize_to", "do_hann"])
    
    return cols


# ==================== API ENDPOINTS ====================
@app.get("/")
def read_root():
    return {
        "message": "Batch Frequency-Domain Feature Extraction API 🚀",
        "version": "1.0.0",
        "batch_size": BATCH_SIZE,
        "features": {
            "SRM": "120 features (20 kernels × 6 stats)",
            "DCT": "57 features",
            "FFT": "103 features",
            "total": "283 features"
        },
        "endpoints": {
            "health": "/health",
            "extract_batch": "/extract/batch"
        }
    }


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "frequency-extraction",
        "temp_dir": TEMP_DIR,
        "batch_size": BATCH_SIZE,
        "target_size": TARGET_SIZE,
        "features": {
            "SRM_kernels": 20,
            "DCT_features": 57,
            "FFT_features": 103,
            "total_features": 283
        }
    }


@app.post("/extract/batch")
async def extract_batch(
    files: List[UploadFile] = File(...),
    session_id: Optional[str] = Form(None),
    cleanup: bool = Form(True)
):
    """
    Extract frequency domain features from up to 30 frames
    
    Pipeline:
    1. Upload frames (up to 30)
    2. Extract SRM, DCT, FFT features for each frame
    3. Return CSV-style data with all features
    4. Clean up temporary files (if cleanup=True)
    
    Args:
        files: List of frame image files (max 30)
        session_id: Optional session ID
        cleanup: Whether to delete temp files after processing (default: True)
    
    Returns:
        JSON with CSV data containing 283 frequency domain features per frame
    """
    
    # Validate batch size
    if len(files) == 0:
        raise HTTPException(status_code=400, detail="No files provided")
    
    if len(files) > BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size exceeds limit. Max {BATCH_SIZE} frames, got {len(files)}"
        )
    
    # Validate file types
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    for file in files:
        if not file.filename:
            raise HTTPException(status_code=400, detail="File missing filename")
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: {file.filename}. Allowed: {allowed_extensions}"
            )
    
    # Generate session ID
    request_id = session_id or f"freq_{os.urandom(8).hex()}"
    
    # Create temporary directory
    temp_frames_dir = os.path.join(TEMP_DIR, request_id)
    os.makedirs(temp_frames_dir, exist_ok=True)
    
    try:
        # Step 1: Save uploaded frames
        logger.info(f"[{request_id}] Saving {len(files)} frames...")
        saved_files = []
        for idx, file in enumerate(files):
            filename = f"frame_{idx:04d}{Path(file.filename).suffix}"
            file_path = os.path.join(temp_frames_dir, filename)
            
            content = await file.read()
            with open(file_path, "wb") as f:
                f.write(content)
            saved_files.append(file_path)
        
        logger.info(f"[{request_id}] Saved {len(saved_files)} frames")
        
        # Step 2: Extract features from all frames
        logger.info(f"[{request_id}] Extracting frequency domain features...")
        all_features = []
        
        for file_path in saved_files:
            features = extract_all_features(file_path)
            all_features.append(features)
        
        logger.info(f"[{request_id}] Extracted features from {len(all_features)} frames")
        
        # Step 3: Convert to CSV-style format (matching OpenFace API structure)
        headers = get_feature_columns()
        
        # Build csv_data array - one entry per frame (matching OpenFace format)
        csv_data_list = []
        for idx, features in enumerate(all_features):
            csv_data_list.append({
                "filename": f"frame_{idx:04d}.jpg.csv",
                "frame_index": idx,  # 0-based frame index for alignment
                "headers": headers,
                "num_rows": 1,  # One row per frame
                "num_columns": len(headers),
                "data": [features]  # Single feature dict in array
            })
        
        # Build response matching OpenFace API structure exactly
        response = {
            "status": "success",
            "session_id": request_id,
            "frames_uploaded": len(files),
            "csv_files_generated": len(csv_data_list),
            "csv_data": csv_data_list,  # Array of csv_data objects
            "summary": {
                "total_csv_files": len(csv_data_list),
                "total_data_rows": len(all_features),
                "features_per_row": len(headers),
                "cleanup_performed": cleanup,
                "output_directory": temp_frames_dir if not cleanup else None
            },
            "cleanup_status": f"Cleaned up {len(saved_files)} temporary files" if cleanup else "Files preserved"
        }
        
        return JSONResponse(content=response)
        
    except Exception as e:
        logger.error(f"[{request_id}] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        # Cleanup temporary files
        if cleanup and os.path.exists(temp_frames_dir):
            try:
                shutil.rmtree(temp_frames_dir)
                logger.info(f"[{request_id}] Cleaned up temporary files")
            except Exception as e:
                logger.warning(f"[{request_id}] Failed to cleanup: {e}")


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8092))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info(f"Starting Batch Frequency-Domain Feature Extraction API on {host}:{port}")
    logger.info(f"Batch size: {BATCH_SIZE} frames")
    logger.info(f"Features: 283 total (SRM: 120, DCT: 57, FFT: 103)")
    uvicorn.run(app, host=host, port=port)
