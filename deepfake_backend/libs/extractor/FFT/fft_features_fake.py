#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FFT feature extractor -> CSV (compact ~104 cols)
- Grayscale, resize_to=256
- Hanning window, remove DC
- POWER (|F|^2) cho mọi đặc trưng năng lượng/phân bố
- JPEG markers đo trên LOG-POWER
"""

import os, csv, math, glob
import numpy as np
import cv2

# =========== CONFIG ===========
INPUT_DIR   = r"/Applications/Tien/deepfake/Dataset/celeb_df_crop/fake"
OUTPUT_CSV  = r"/Applications/Tien/deepfake/extract-celeb/FFT/fft_features_fake.csv"
RESIZE_TO   = 256
COLOR_MODE  = "gray"     # hiện dùng gray
DO_HANN     = 1
APS_BINS    = 12
RPS_BINS    = 64
EPS         = 1e-8
LOW_CUT     = 0.10
MID_CUT     = 0.30
DC_RADIUS_PX= 2
HF_RANGE    = (0.15, 1.00)
TOPK_PEAKS  = 3
JPEG_PERIOD = 8
SMOOTH_K    = 5
# ==============================

def list_images(root):
    exts = (".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp")
    files = [p for p in glob.iglob(os.path.join(root, "**", "*"), recursive=True)
             if p.lower().endswith(exts)]
    return sorted(files)   # đảm bảo thứ tự ổn định

def make_hann(n):
    h1 = np.hanning(n)
    return np.outer(h1, h1).astype(np.float32)

def to_gray(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

def resize_square(img, size):
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)

def fft2_power_and_log(x):
    F = np.fft.fft2(x)
    F = np.fft.fftshift(F)
    P = (F.real**2 + F.imag**2)
    P_log = np.log(P + EPS)
    return P, P_log

def coord_grids(n):
    cy = (n-1)/2.0; cx = (n-1)/2.0
    y, x = np.indices((n, n))
    dy = y - cy; dx = x - cx
    r = np.sqrt(dx*dx + dy*dy)
    theta = np.mod(np.arctan2(dy, dx), np.pi)
    r_norm = r / (np.sqrt(2)*((n-1)/2.0))
    return r, r_norm, theta

def remove_dc(A, r, dc_radius_px):
    out = A.copy()
    out[r <= dc_radius_px] = 0.0
    return out

def band_energies(P_like, r_norm):
    low_mask  = (r_norm > 0.0) & (r_norm <= LOW_CUT)
    mid_mask  = (r_norm > LOW_CUT) & (r_norm <= MID_CUT)
    high_mask = (r_norm > MID_CUT) & (r_norm <= 1.0 + 1e-6)
    E_low  = float(P_like[low_mask].sum())
    E_mid  = float(P_like[mid_mask].sum())
    E_high = float(P_like[high_mask].sum())
    return E_low, E_mid, E_high

def radial_stats(P_like, r_norm):
    w = P_like
    wsum = float(w.sum()) + EPS
    rc = float((r_norm * w).sum() / wsum)
    var = float((w * (r_norm - rc)**2).sum() / wsum)
    return rc, var

def radial_profile(P_like, r_norm, edges, normalize=True):
    r_flat = r_norm.ravel()
    p_flat = P_like.ravel()
    idx = np.clip(np.digitize(r_flat, edges) - 1, 0, len(edges)-2)
    rps = np.bincount(idx, weights=p_flat, minlength=len(edges)-1).astype(np.float64)
    if normalize:
        s = rps.sum()
        if s > 0: rps /= s
    return rps

def angular_profile(P_like, theta, aps_bins=12, normalize=True):
    edges = np.linspace(0.0, np.pi, aps_bins+1)
    t_flat = theta.ravel()
    p_flat = P_like.ravel()
    idx = np.clip(np.digitize(t_flat, edges) - 1, 0, len(edges)-2)
    aps = np.bincount(idx, weights=p_flat, minlength=len(edges)-1).astype(np.float64)
    if normalize:
        s = aps.sum()
        if s > 0: aps /= s
    return aps

def rolloff(P_like, r_norm, perc, rps_bins=RPS_BINS):
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
    x = np.maximum(P_like.ravel(), EPS)
    amean = float(np.mean(x))
    gmean = float(np.exp(np.mean(np.log(x))))
    return float(gmean / (amean + EPS))

def spectral_entropy(P_like):
    x = np.maximum(P_like.ravel(), EPS)
    q = x / (x.sum() + EPS)
    H = -float((q * np.log(q)).sum())
    H /= math.log(len(q) + EPS)
    return H

def smooth_1d(x, k=5):
    if k <= 1: return x
    ker = np.ones(int(k), dtype=np.float64) / float(k)
    return np.convolve(x, ker, mode="same")

def hf_slope_beta_from_rps_power(rps_power, hf_range=HF_RANGE):
    edges = np.linspace(0.0, 1.0, len(rps_power)+1)
    centers = 0.5*(edges[:-1] + edges[1:])
    lo, hi = hf_range
    m = (centers >= lo) & (centers <= hi) & (rps_power > 0)
    if m.sum() < 3: return 0.0
    x = np.log(centers[m] + EPS)
    y = np.log(rps_power[m] + EPS)
    A = np.vstack([np.ones_like(x), x]).T
    coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    return float(coef[1])

def topk_peaks_on_rps(rps_power, k=3, min_index=3, smooth_k=SMOOTH_K):
    rps_sm = smooth_1d(rps_power, k=smooth_k)
    idx_sorted = np.argsort(rps_sm)[::-1]
    picks = []
    for idx in idx_sorted:
        if idx < min_index: continue
        if any(abs(idx - p[0]) <= 2 for p in picks): continue
        picks.append((idx, rps_sm[idx]))
        if len(picks) >= k: break
    edges = np.linspace(0.0, 1.0, len(rps_power)+1)
    out = []
    for idx, val in picks:
        r_center = 0.5*(edges[idx] + edges[idx+1])
        out.append((float(r_center), float(val)))
    while len(out) < k:
        out.append((0.0, 0.0))
    return out

def jpeg_8x8_markers_energy(P_log, period=JPEG_PERIOD):
    n = P_log.shape[0]
    k = int(round(n / period))
    if k < 1: return (0.0, 0.0, 0.0)
    cx = cy = (n-1)//2
    def patch_mean(u, v, rad=2):
        u = int(u); v = int(v)
        u0,u1 = max(0,u-rad), min(n, u+rad+1)
        v0,v1 = max(0,v-rad), min(n, v+rad+1)
        return float(np.mean(P_log[v0:v1, u0:u1]))
    pts_x = [(cx+k, cy), (cx-k, cy)]
    pts_y = [(cx, cy+k), (cx, cy-k)]
    pts_d = [(cx+k, cy+k), (cx-k, cy-k), (cx+k, cy-k), (cx-k, cy+k)]
    ex = np.mean([patch_mean(u,v) for (u,v) in pts_x])
    ey = np.mean([patch_mean(u,v) for (u,v) in pts_y])
    ed = np.mean([patch_mean(u,v) for (u,v) in pts_d])
    return (ex, ey, ed)

def build_header():
    hdr = ["filename",  # ← sửa từ 'image_id' thành 'filename'
           "fft_psd_total",
           "fft_E_low","fft_E_mid","fft_E_high","fft_E_high_over_low","fft_E_mid_over_low",
           "fft_radial_centroid","fft_radial_bandwidth",
           "fft_rolloff_85","fft_rolloff_95",
           "fft_spectral_flatness","fft_spectral_entropy","fft_hf_slope_beta"]
    hdr += [f"fft_aps_{i}" for i in range(APS_BINS)]
    hdr += [f"fft_rps_{i}" for i in range(RPS_BINS)]
    hdr += ["fft_peak1_r","fft_peak1_val","fft_peak2_r","fft_peak2_val","fft_peak3_r","fft_peak3_val",
            "fft_jpeg_8x8_x","fft_jpeg_8x8_y","fft_jpeg_8x8_diag",
            "width","height","color_mode","resize_to","do_hann"]
    return hdr

def process_image(path, precomp):
    try:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None: raise RuntimeError("imread failed")
        h0, w0 = img.shape[:2]
        img = to_gray(img)
        img = resize_square(img, RESIZE_TO).astype(np.float32)
        img = img - img.mean()
        std = img.std()
        if std > 1e-6: img = img / std
        if DO_HANN: img = img * precomp["hann"]

        P, P_log = fft2_power_and_log(img)
        r, r_norm, theta = precomp["r"], precomp["r_norm"], precomp["theta"]
        P = remove_dc(P, r, DC_RADIUS_PX)
        P_log = remove_dc(P_log, r, DC_RADIUS_PX)
        P_pos = np.maximum(P, EPS)

        E_low, E_mid, E_high = band_energies(P_pos, r_norm)
        psd_total = float(P_pos.sum())
        E_high_over_low = float(E_high / (E_low + EPS))
        E_mid_over_low  = float(E_mid  / (E_low + EPS))
        rc, rbw = radial_stats(P_pos, r_norm)
        r85 = rolloff(P_pos, r_norm, 0.85)
        r95 = rolloff(P_pos, r_norm, 0.95)
        flat = spectral_flatness(P_pos)
        sent = spectral_entropy(P_pos)

        edges_r = np.linspace(0.0, 1.0, RPS_BINS+1)
        rps = radial_profile(P_pos, r_norm, edges_r, normalize=True)
        aps = angular_profile(P_pos, theta, APS_BINS, normalize=True)
        rps_power = radial_profile(P_pos, r_norm, edges_r, normalize=False)
        beta  = hf_slope_beta_from_rps_power(rps_power, HF_RANGE)
        peaks = topk_peaks_on_rps(rps_power, k=TOPK_PEAKS, min_index=3, smooth_k=SMOOTH_K)
        jx, jy, jd = jpeg_8x8_markers_energy(P_log, JPEG_PERIOD)

        row = [os.path.basename(path),
               psd_total,
               E_low, E_mid, E_high, E_high_over_low, E_mid_over_low,
               rc, rbw,
               r85, r95,
               flat, sent, beta]
        row += list(map(float, aps.tolist()))
        row += list(map(float, rps.tolist()))
        for rpk, vpk in peaks:
            row += [rpk, vpk]
        row += [jx, jy, jd,
                w0, h0, COLOR_MODE, RESIZE_TO, DO_HANN]
        return row

    except Exception as e:
        print(f"[WARN] Failed on {path}: {e}")
        n_feats = len(build_header()) - 1
        return [os.path.basename(path)] + [float("nan")]*n_feats

def precompute(n):
    H = make_hann(n) if DO_HANN else np.ones((n,n), dtype=np.float32)
    r, r_norm, theta = coord_grids(n)
    return {"hann": H, "r": r, "r_norm": r_norm, "theta": theta}

def main():
    paths = list_images(INPUT_DIR)
    if not paths:
        print(f"No images found under: {INPUT_DIR}")
        return
    os.makedirs(os.path.dirname(OUTPUT_CSV) or ".", exist_ok=True)

    precomp = precompute(RESIZE_TO)
    header = build_header()

    try:
        from tqdm import tqdm
        iterator = tqdm(paths, desc="Extracting FFT features")
    except Exception:
        iterator = paths

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for p in iterator:
            row = process_image(p, precomp)
            writer.writerow(row)

    print(f"✅ Done. Wrote CSV: {OUTPUT_CSV}")
    print(f"📸 Total images processed: {len(paths)}")

if __name__ == "__main__":
    main()
