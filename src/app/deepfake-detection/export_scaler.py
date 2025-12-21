#!/usr/bin/env python3
"""
Export StandardScaler from Training Notebook

This script extracts and saves the StandardScaler used during training
so the consumer can apply the same normalization to CSV features.

Run this after training to save the scaler.
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# Paths
FUSION_CSV = "/home/huuquangdang/huu.quang.dang/thesis/deepfake-1801/src/libs/dataset/samples/fusion_ff_fake.csv"
OUTPUT_PATH = "/home/huuquangdang/huu.quang.dang/thesis/deepfake-1801/src/app/deepfake-detection/csv_scaler.pkl"

def extract_csv_features_from_sample():
    """
    Extract CSV features matching consumer's build_inputs:
    - 283 frequency features (SRM, DCT, FFT + 3 metadata: color_mode, do_hann, resize_to)
    - 669 OpenFace features (excluding: frame, face_id, timestamp, confidence, success)
    - Total: 952 features
    """
    print(f"Loading sample CSV from: {FUSION_CSV}")
    df = pd.read_csv(FUSION_CSV)
    
    print(f"CSV shape: {df.shape}")
    print(f"Columns: {list(df.columns[:20])}... (showing first 20)")
    
    # Frequency features: 283 total
    # SRM: 20 kernels × 6 stats = 120
    srm_cols = []
    for i in range(1, 21):
        srm_cols.extend([f"SRM_mean_{i}", f"SRM_var_{i}", f"SRM_skew_{i}", 
                        f"SRM_kurt_{i}", f"SRM_entropy_{i}", f"SRM_energy_{i}"])
    
    # DCT: 57 features
    dct_cols = []
    for band in ["low", "mid", "high"]:
        dct_cols.extend([f"DCT_mean_{band}", f"DCT_var_{band}", 
                        f"DCT_skew_{band}", f"DCT_kurt_{band}"])
    for band in ["low", "mid", "high"]:
        dct_cols.append(f"DCT_entropy_{band}")
    dct_cols.append("DCT_energy_total")
    for i in range(20):
        dct_cols.append(f"DCT_zigzag_{i}")
    for band in ["low", "mid", "high"]:
        for bin_idx in range(8):
            dct_cols.append(f"DCT_hist_{band}_bin_{bin_idx}")
    
    # FFT: 103 features
    fft_cols = [
        "fft_psd_total", "fft_E_low", "fft_E_mid", "fft_E_high",
        "fft_E_high_over_low", "fft_E_mid_over_low",
        "fft_radial_centroid", "fft_radial_bandwidth",
        "fft_rolloff_85", "fft_rolloff_95",
        "fft_spectral_flatness", "fft_spectral_entropy", "fft_hf_slope_beta"
    ]
    for i in range(12):
        fft_cols.append(f"fft_aps_{i}")
    for i in range(64):
        fft_cols.append(f"fft_rps_{i}")
    for i in range(1, 4):
        fft_cols.extend([f"fft_peak{i}_r", f"fft_peak{i}_val"])
    fft_cols.extend(["fft_jpeg_8x8_x", "fft_jpeg_8x8_y", "fft_jpeg_8x8_diag"])
    
    # Metadata: 5 features (width, height, color_mode, resize_to, do_hann)
    meta_cols = ["width", "height", "color_mode", "resize_to", "do_hann"]
    
    freq_cols = srm_cols + dct_cols + fft_cols + meta_cols
    
    print(f"Frequency features: {len(freq_cols)} (should be 283)")
    
    # OpenFace features: 669 (excluding frame, face_id, timestamp, confidence, success)
    # Using feature_6 to feature_674 (skip first 5)
    of_cols = [f"feature_{i}" for i in range(6, 675)]
    
    print(f"OpenFace features: {len(of_cols)} (should be 669)")
    
    all_feature_cols = freq_cols + of_cols
    print(f"Total features: {len(all_feature_cols)} (should be 952)")
    
    # Check which columns are missing
    missing_cols = [c for c in all_feature_cols if c not in df.columns]
    if missing_cols:
        print(f"WARNING: {len(missing_cols)} columns missing from CSV")
        print(f"First 10 missing: {missing_cols[:10]}")
        # Fill missing with zeros
        for col in missing_cols:
            df[col] = 0.0
    
    # Convert color_mode to numeric (0 for gray, 1 for rgb, etc.)
    if 'color_mode' in df.columns:
        df['color_mode'] = df['color_mode'].map({'gray': 0, 'rgb': 1, 'bgr': 1}).fillna(0).astype(float)
    
    # Extract features in order
    csv_features = df[all_feature_cols].values.astype(np.float32)
    
    print(f"Extracted CSV features shape: {csv_features.shape}")
    print(f"Feature range: min={csv_features.min():.2f}, max={csv_features.max():.2f}, mean={csv_features.mean():.2f}")
    
    assert csv_features.shape[1] == 952, f"Expected 952 features, got {csv_features.shape[1]}"
    
    return csv_features

def fit_and_save_scaler():
    """Fit StandardScaler on sample data and save"""
    print("\n" + "="*50)
    print("Fitting StandardScaler...")
    print("="*50 + "\n")
    
    # Extract CSV features
    csv_features = extract_csv_features_from_sample()
    
    # Fit scaler
    scaler = StandardScaler()
    scaler.fit(csv_features)
    
    print(f"Scaler fitted on {csv_features.shape[0]} samples")
    print(f"Feature mean (first 10): {scaler.mean_[:10]}")
    print(f"Feature std (first 10): {scaler.scale_[:10]}")
    
    # Save scaler
    output_path = Path(OUTPUT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(scaler, OUTPUT_PATH)
    print(f"\nStandardScaler saved to: {OUTPUT_PATH}")
    print(f"File size: {output_path.stat().st_size} bytes")
    
    # Test load
    print("\nTesting scaler load...")
    loaded_scaler = joblib.load(OUTPUT_PATH)
    test_transform = loaded_scaler.transform(csv_features[:1])
    print(f"Test transform shape: {test_transform.shape}")
    print(f"Test transform (first 10): {test_transform[0, :10]}")
    
    print("\n✓ StandardScaler successfully saved and tested!")

if __name__ == "__main__":
    fit_and_save_scaler()
