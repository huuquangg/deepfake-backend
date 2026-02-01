#!/usr/bin/env python3
"""
Diagnostic tool to check consumer configuration and model behavior
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from consumer import Config, of_fixed_columns

def check_dimensions():
    """Check feature dimensions"""
    print("="*70)
    print("FEATURE DIMENSION CHECK")
    print("="*70)
    
    cfg = Config()
    of_cols = of_fixed_columns()
    
    freq_features = 283
    of_features = len(of_cols)
    total = freq_features + of_features
    
    print(f"Frequency features: {freq_features}")
    print(f"OpenFace features (no metadata): {of_features}")
    print(f"Total CSV features: {total}")
    print(f"Config csv_features: {cfg.csv_features}")
    print(f"Model expects: 952")
    print()
    
    if total == cfg.csv_features == 952:
        print("✓ Feature dimensions correct")
    else:
        print(f"✗ Dimension mismatch: {total} != {cfg.csv_features}")
        return False
    
    return True

def check_model():
    """Check model loading and output interpretation"""
    print("\n" + "="*70)
    print("MODEL CHECK")
    print("="*70)
    
    try:
        from consumer import DeepfakeDetector, Config
        import numpy as np
        
        cfg = Config()
        detector = DeepfakeDetector(model_path=cfg.model_path)
        
        # Create dummy inputs
        img_input = np.random.rand(1, 15, 224, 224, 3).astype(np.float32)
        csv_input = np.random.rand(1, 15, 952).astype(np.float32)
        
        print(f"Model loaded: {cfg.model_path}")
        print(f"Input shapes: img={img_input.shape}, csv={csv_input.shape}")
        
        # Test prediction
        result = detector.predict(img_input, csv_input)
        
        print(f"\nPrediction result:")
        print(f"  Label: {result['pred_label']}")
        print(f"  Prob(real): {result['prob_real']:.6f}")
        print(f"  Prob(fake): {result['prob_fake']:.6f}")
        print(f"  Confidence: {result['confidence']:.6f}")
        print(f"  Time: {result['inference_ms']}ms")
        
        print("\n✓ Model working correctly")
        print("  Note: With random input, predictions are meaningless")
        print("  Real issues would be: scaler missing, empty features in messages")
        
        return True
        
    except Exception as e:
        print(f"✗ Model check failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_config():
    """Check configuration"""
    print("\n" + "="*70)
    print("CONFIGURATION CHECK")
    print("="*70)
    
    cfg = Config()
    
    print(f"RabbitMQ URL: {cfg.rabbitmq_url}")
    print(f"RabbitMQ Queue: {cfg.rabbitmq_queue}")
    print(f"Model path: {cfg.model_path}")
    print(f"Scaler path: {cfg.scaler_path}")
    print(f"Sequence length: {cfg.sequence_length}")
    print(f"Pad sequence: {cfg.pad_sequence}")
    
    if cfg.pad_sequence:
        print("\n⚠ WARNING: Padding enabled for sequences < 15 frames")
        print("  This may cause degraded predictions on short batches")
        print("  Consider buffering frames per session until 15 available")
    
    # Check files exist
    print("\nFile checks:")
    if os.path.exists(cfg.model_path):
        print(f"  ✓ Model file exists")
    else:
        print(f"  ✗ Model file NOT FOUND: {cfg.model_path}")
        return False
    
    if os.path.exists(cfg.scaler_path):
        print(f"  ✓ Scaler file exists")
    else:
        print(f"  ⚠ Scaler file NOT FOUND: {cfg.scaler_path}")
        print(f"    Predictions will use unscaled features (may reduce accuracy)")
    
    return True

def main():
    print("\n" + "="*70)
    print("DEEPFAKE CONSUMER DIAGNOSTICS")
    print("="*70 + "\n")
    
    checks = [
        ("Dimensions", check_dimensions),
        ("Configuration", check_config),
        ("Model", check_model),
    ]
    
    results = []
    for name, check_fn in checks:
        try:
            result = check_fn()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} check failed with exception: {e}")
            results.append((name, False))
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    all_pass = all(r for _, r in results)
    
    if all_pass:
        print("\n✓ All checks passed!")
        print("\nIf you're still seeing 'all real' predictions:")
        print("  1. Check RabbitMQ messages have non-empty 'features' array")
        print("  2. Enable DEBUG logging: export LOG_LEVEL=DEBUG")
        print("  3. Check input_stats in logs (min/max/mean values)")
        print("  4. Verify video-streaming is publishing both frequency + openface features")
    else:
        print("\n✗ Some checks failed - fix above issues first")
    
    return 0 if all_pass else 1

if __name__ == "__main__":
    sys.exit(main())
