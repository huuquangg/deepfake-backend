# Deepfake Detection Consumer

Realtime deepfake detection consumer that processes RabbitMQ messages, fetches frames, and runs TCN inference.

## Architecture

```
RabbitMQ (deepfake.features)
    ↓
Consumer receives: {session_id, batch_id, frame_refs:[...], features:[...]}
    ↓
Frame Fetcher → Downloads frames from video-streaming HTTP
    ↓
Input Builder → Constructs (1,15,224,224,3) + (1,15,957) tensors
    ↓
TCN Model → Inference (no modification needed)
    ↓
WebSocket → Broadcast predictions (TODO)
```

## Setup

### 1. Export StandardScaler from Training

The scaler must match your training preprocessing:

```bash
cd /home/huuquangdang/huu.quang.dang/thesis/deepfake-1801/src/app/deepfake-detection

# Export scaler from sample or training data
python3 export_scaler.py
```

This creates `csv_scaler.pkl` containing the fitted StandardScaler.

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Environment Configuration

```bash
export RABBITMQ_URL="amqp://admin:P@ssw0rd123@localhost:5672/"
export RABBITMQ_QUEUE="feature.extraction.results"
export MODEL_PATH="/home/huuquangdang/huu.quang.dang/thesis/deepfake-1801/src/app/deepfake-detection/TCN_TemporalConvNet_Residual Blocks_final.h5"
export SCALER_PATH="/home/huuquangdang/huu.quang.dang/thesis/deepfake-1801/src/app/deepfake-detection/csv_scaler.pkl"
```

## Usage

### Start Consumer

```bash
python3 consumer.py
```

The consumer will:
1. Connect to RabbitMQ
2. Load TCN model and scaler
3. Process messages as they arrive
4. Log predictions

### Test with Sample Message

Publish a test message to RabbitMQ:

```bash
cd /home/huuquangdang/huu.quang.dang/thesis/deepfake-1801/src/app/deepfake-detection
python3 test_consumer.py
```

## Message Format (Input)

The consumer expects RabbitMQ messages with this structure:

```json
{
  "session_id": "abc123",
  "batch_id": "abc123_20231220_120000_000",
  "timestamp": "2023-12-20T12:00:00Z",
  "frame_count": 30,
  "frame_refs": [
    {
      "frame_index": 0,
      "filename": "frame_0000.jpg",
      "rel_path": "abc123/abc123_20231220_120000_000/frame_0000.jpg",
      "uri": "http://localhost:8091/frames/abc123/abc123_20231220_120000_000/frame_0000.jpg"
    },
    ...
  ],
  "features": [
    {
      "frame_index": 0,
      "filename": "frame_0000.jpg.csv",
      "_source": "frequency",
      "SRM_mean_1": 0.123,
      ...
    },
    {
      "frame_index": 0,
      "filename": "frame_0000.jpg.csv",
      "_source": "openface",
      "feature_1": 1.0,
      ...
    },
    ...
  ]
}
```

## Prediction Output

Predictions are logged and will be broadcast via WebSocket (TODO):

```json
{
  "session_id": "abc123",
  "batch_id": "abc123_20231220_120000_000",
  "window_start_frame": 0,
  "frame_count": 30,
  "prediction": {
    "pred_prob": 0.87,
    "pred_label": "fake",
    "confidence": 0.87,
    "real_prob": 0.13,
    "inference_ms": 234
  },
  "timestamp": 1703073600.123
}
```

## Input Building Details

### CSV Input (1, 15, 957)

For each frame in the 15-frame window:
1. Extract frequency features (283): SRM, DCT, FFT
2. Extract openface features (674): feature_1..feature_674
3. Concatenate → 957 features
4. Apply StandardScaler (fitted during training)
5. Stack 15 frames → (15, 957)
6. Add batch dimension → (1, 15, 957)

### Image Input (1, 15, 224, 224, 3)

For each frame in the 15-frame window:
1. Fetch frame from `frame_refs[i].uri`
2. Decode JPEG → numpy array
3. Resize to (224, 224)
4. Convert to float32 / 255.0
5. Keep BGR color space (matches training)
6. Stack 15 frames → (15, 224, 224, 3)
7. Add batch dimension → (1, 15, 224, 224, 3)

## Frame Alignment

Both `frame_refs` and `features` are aligned by `frame_index` (0-based):
- Consumer groups by `frame_index`
- Requires both frequency + openface rows for each frame
- Missing frames cause window to be skipped

## Performance

- **Throughput**: ~2-5 FPS on CPU, ~10-20 FPS on GPU
- **Latency**: 
  - Frame fetch: ~50-100ms for 15 frames
  - Preprocessing: ~20-50ms
  - Inference: ~100-500ms (CPU), ~20-50ms (GPU)
  - Total: ~200-650ms per prediction

## Troubleshooting

### Missing Frames (404)

Check that:
- `video-streaming` service is running on port 8091
- `ENABLE_FRAME_STORAGE=true` in video-streaming config
- `Storage/frames/` directory exists and is writable

### Feature Count Mismatch

Ensure your frequency + openface APIs return exactly:
- 283 frequency features
- 674 openface features
- Total: 957 features

### Scaler Not Found

Run `python3 export_scaler.py` to create `csv_scaler.pkl`.

### Model Input Shape Error

Verify your `.h5` expects:
- Input 1: (batch, 15, 224, 224, 3)
- Input 2: (batch, 15, 957)

Run:
```python
import tensorflow as tf
model = tf.keras.models.load_model("TCN_TemporalConvNet_Residual Blocks_final.h5")
print(model.summary())
```

## Next Steps

1. **WebSocket Broadcasting**: Add WebSocket server to broadcast predictions
2. **Sliding Windows**: Process overlapping 15-frame windows for smoother predictions
3. **Batch Processing**: Handle multiple messages in parallel
4. **Monitoring**: Add Prometheus metrics for throughput/latency
5. **Error Handling**: Dead-letter queue for failed predictions
