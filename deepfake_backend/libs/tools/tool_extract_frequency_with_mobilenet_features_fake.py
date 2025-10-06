"""
Extract Frequency + MobileNet features for FAKE dataset
Save separately for multi-input training
"""
# /Applications/Tien/deepfake-backend/deepfake-backend/deepfake_backend/libs/tools/tool_extract_features_fake.py

import os
import requests
import numpy as np
import time
from pathlib import Path

# =========================
# CONFIG - FAKE DATASET
# =========================
FOLDER_PATH = "/Applications/Tien/deepfake/Dataset/celeb_df_crop/fake"
OUTPUT_DIR = "/Applications/Tien/deepfake/Dataset/features/fake"
CHECKPOINT_FILE = "checkpoint_fake.txt"

FREQ_DIR = os.path.join(OUTPUT_DIR, "frequency")
MOBILE_DIR = os.path.join(OUTPUT_DIR, "mobilenet")
os.makedirs(FREQ_DIR, exist_ok=True)
os.makedirs(MOBILE_DIR, exist_ok=True)

# API endpoints
FREQUENCY_API = "http://127.0.0.1:8000/api/detection/frequency/analyze"
MOBILENET_API = "http://127.0.0.1:8000/api/detection/extract-features?model_name=mobilenet"


def main():
    # Load checkpoint
    processed = set()
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            processed = set(line.strip() for line in f)
        print(f"Loaded checkpoint: {len(processed)} files processed")
    
    # Get all files
    files = sorted([f for f in os.listdir(FOLDER_PATH) 
                   if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    to_process = [f for f in files if f not in processed]
    
    print("="*80)
    print("EXTRACT FAKE DATASET - Save separately")
    print("="*80)
    print(f"Total files: {len(files)}")
    print(f"Already processed: {len(processed)}")
    print(f"Remaining: {len(to_process)}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Checkpoint: {CHECKPOINT_FILE}")
    print("="*80)
    
    if not to_process:
        print("All files already processed!")
        return
    
    success = 0
    errors = 0
    
    for idx, filename in enumerate(to_process, 1):
        filepath = os.path.join(FOLDER_PATH, filename)
        basename = Path(filename).stem
        
        print(f"\n[{idx}/{len(to_process)}] {filename}")
        print(f"Progress: {(len(processed) + idx) / len(files) * 100:.2f}%")
        
        try:
            # Extract frequency
            print("  → Frequency...")
            with open(filepath, "rb") as f:
                resp = requests.post(FREQUENCY_API, 
                                   files={"file": (filename, f, "image/jpeg")}, 
                                   timeout=60)
            
            if resp.status_code == 200:
                freq_result = resp.json()
                freq_npy = freq_result['feature_npy']
                
                if os.path.exists(freq_npy):
                    freq = np.load(freq_npy)  # (224, 224, 32)
                    np.save(os.path.join(FREQ_DIR, f"{basename}.npy"), freq)
                    print(f"  ✓ Frequency: {freq.shape}")
                else:
                    print(f"  ✗ Frequency file not found")
                    errors += 1
                    continue
            else:
                print(f"  ✗ Frequency API failed ({resp.status_code})")
                errors += 1
                continue
            
            # Extract MobileNet
            print("  → MobileNet...")
            with open(filepath, "rb") as f:
                resp = requests.post(MOBILENET_API, 
                                   files={"file": (filename, f, "image/jpeg")}, 
                                   timeout=60)
            
            if resp.status_code == 200:
                mobile_result = resp.json()
                mobile = np.array(mobile_result['embedding'], dtype=np.float32)  # (1280,)
                np.save(os.path.join(MOBILE_DIR, f"{basename}.npy"), mobile)
                print(f"  ✓ MobileNet: {mobile.shape}")
            else:
                print(f"  ✗ MobileNet API failed ({resp.status_code})")
                errors += 1
                continue
            
            # Update checkpoint
            with open(CHECKPOINT_FILE, "a") as cp:
                cp.write(f"{filename}\n")
            
            success += 1
            time.sleep(0.1)
            
        except requests.exceptions.Timeout:
            print(f"  ✗ TIMEOUT")
            errors += 1
        except Exception as e:
            print(f"  ✗ Error: {type(e).__name__}: {e}")
            errors += 1
    
    # Summary
    print("\n" + "="*80)
    print("FAKE DATASET - EXTRACTION COMPLETE")
    print("="*80)
    print(f"Processed this run: {len(to_process)}")
    print(f"Success: {success}")
    print(f"Errors: {errors}")
    print(f"Total completed: {len(processed) + success} / {len(files)}")
    print(f"Output saved to: {OUTPUT_DIR}")
    print("="*80)


if __name__ == "__main__":
    main()