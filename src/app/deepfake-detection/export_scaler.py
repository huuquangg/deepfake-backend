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
    Extract CSV features (columns 283:957) from sample fusion CSV
    and fit a StandardScaler to match training
    """
    print(f"Loading sample CSV from: {FUSION_CSV}")
    df = pd.read_csv(FUSION_CSV)
    
    print(f"CSV shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()[:10]}... (showing first 10)")
    
    # Extract frequency + openface features (columns after metadata)
    # Based on your CSV structure:
    # - SRM features: SRM_mean_1 to SRM_energy_20 (120 features)
    # - DCT features: DCT_mean_low to DCT_hist_high_bin_7 (57 features)
    # - FFT features: fft_psd_total to fft_jpeg_8x8_diag (103 features)
    # - Metadata: width, height, color_mode, resize_to, do_hann (5 features)
    # - OpenFace features: feature_1 to feature_674 (674 features)
    
    # Total: 120 + 57 + 103 + 3 = 283 frequency features (excluding some metadata)
    #        674 openface features
    #        = 957 total
    
    # Find feature column ranges
    freq_start_col = "SRM_mean_1"
    freq_end_col = "fft_jpeg_8x8_diag"
    of_start_col = "feature_1"
    of_end_col = "feature_674"
    
    freq_start_idx = df.columns.get_loc(freq_start_col)
    freq_end_idx = df.columns.get_loc(freq_end_col) + 1
    of_start_idx = df.columns.get_loc(of_start_col)
    of_end_idx = df.columns.get_loc(of_end_col) + 1
    
    print(f"Frequency features: columns {freq_start_idx}:{freq_end_idx} ({freq_end_idx - freq_start_idx} features)")
    print(f"OpenFace features: columns {of_start_idx}:{of_end_idx} ({of_end_idx - of_start_idx} features)")
    
    # Extract features
    freq_features = df.iloc[:, freq_start_idx:freq_end_idx].values
    of_features = df.iloc[:, of_start_idx:of_end_idx].values
    
    # Concatenate
    all_csv_features = np.concatenate([freq_features, of_features], axis=1)
    
    print(f"Combined CSV features shape: {all_csv_features.shape}")
    
    if all_csv_features.shape[1] != 957:
        print(f"WARNING: Expected 957 features but got {all_csv_features.shape[1]}")
        print("Adjusting to 957...")
        if all_csv_features.shape[1] > 957:
            all_csv_features = all_csv_features[:, :957]
        else:
            # Pad with zeros if needed
            padding = np.zeros((all_csv_features.shape[0], 957 - all_csv_features.shape[1]))
            all_csv_features = np.concatenate([all_csv_features, padding], axis=1)
    
    return all_csv_features

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
