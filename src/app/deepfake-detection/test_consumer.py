#!/usr/bin/env python3
"""
Test script to validate consumer.py with sample RabbitMQ JSON

This script simulates processing a single message without RabbitMQ,
useful for debugging the input building and prediction logic.
"""

import json
import sys
import os
from pathlib import Path

# Add parent directory to path to import consumer
sys.path.insert(0, str(Path(__file__).parent))

from consumer import Config, FrameFetcher, InputBuilder, DeepfakeDetector

def test_with_sample_json(json_path: str):
    """Test consumer with sample RabbitMQ JSON"""
    
    print("=" * 80)
    print("DEEPFAKE CONSUMER TEST")
    print("=" * 80)
    
    # Load sample JSON
    print(f"\n1. Loading sample JSON from: {json_path}")
    with open(json_path, 'r') as f:
        message = json.load(f)
    
    print(f"   ✓ Session ID: {message.get('session_id')}")
    print(f"   ✓ Batch ID: {message.get('batch_id')}")
    print(f"   ✓ Frame count: {message.get('frame_count')}")
    print(f"   ✓ Frame refs: {len(message.get('frame_refs', []))}")
    print(f"   ✓ Features: {len(message.get('features', []))}")
    
    # Initialize components
    print("\n2. Initializing consumer components...")
    cfg = Config()
    print(f"   ✓ Config loaded")
    print(f"     - Sequence length: {cfg.sequence_length}")
    print(f"     - Image size: {cfg.image_size}")
    print(f"     - CSV features: {cfg.csv_features}")
    print(f"     - Pad sequence: {cfg.pad_sequence}")
    print(f"     - Window start mode: {cfg.window_start_mode}")
    
    frame_fetcher = FrameFetcher(
        cache_size=cfg.frame_cache_size,
        timeout_s=cfg.frame_fetch_timeout_s,
        image_size=cfg.image_size,
    )
    print(f"   ✓ FrameFetcher initialized")
    
    input_builder = InputBuilder(config=cfg, frame_fetcher=frame_fetcher)
    print(f"   ✓ InputBuilder initialized")
    
    # Build inputs
    print("\n3. Building model inputs...")
    try:
        img_input, csv_input, uris_used, window_start = input_builder.build_inputs(message)
        print(f"   ✓ Inputs built successfully")
        print(f"     - Image input shape: {img_input.shape}")
        print(f"     - CSV input shape: {csv_input.shape}")
        print(f"     - Window start: {window_start}")
        print(f"     - URIs used: {len(uris_used)}")
        
        # Show some statistics
        print(f"\n   Image statistics:")
        print(f"     - Min: {img_input.min():.4f}")
        print(f"     - Max: {img_input.max():.4f}")
        print(f"     - Mean: {img_input.mean():.4f}")
        
        print(f"\n   CSV statistics:")
        print(f"     - Min: {csv_input.min():.4f}")
        print(f"     - Max: {csv_input.max():.4f}")
        print(f"     - Mean: {csv_input.mean():.4f}")
        print(f"     - Has NaN: {bool((csv_input != csv_input).any())}")
        print(f"     - Has Inf: {bool((csv_input == float('inf')).any())}")
        
    except Exception as e:
        print(f"   ✗ Failed to build inputs: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Load model and predict
    print("\n4. Loading model and running prediction...")
    try:
        detector = DeepfakeDetector(model_path=cfg.model_path)
        print(f"   ✓ Model loaded")
        
        pred = detector.predict(img_input, csv_input)
        print(f"   ✓ Prediction completed")
        print(f"\n   RESULTS:")
        print(f"     - Label: {pred['pred_label'].upper()}")
        print(f"     - Confidence: {pred['confidence']:.4f}")
        print(f"     - Fake probability: {pred['pred_prob']:.6f}")
        print(f"     - Real probability: {pred['real_prob']:.6f}")
        print(f"     - Inference time: {pred['inference_ms']} ms")
        
    except Exception as e:
        print(f"   ✗ Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 80)
    print("TEST PASSED ✓")
    print("=" * 80)
    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_consumer.py <path_to_sample_json>")
        print("\nExample:")
        print("  python test_consumer.py rbmq.json")
        sys.exit(1)
    
    json_path = sys.argv[1]
    if not os.path.exists(json_path):
        print(f"Error: File not found: {json_path}")
        sys.exit(1)
    
    success = test_with_sample_json(json_path)
    sys.exit(0 if success else 1)
