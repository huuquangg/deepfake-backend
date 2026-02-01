# Batch Deepfake Detection API

30-frame batch processing API that orchestrates the complete pipeline:
1. **Frame Upload** - Upload up to 30 frames
2. **OpenFace Extraction** - Extract facial features from frames
3. **CSV Processing** - Read extracted features from CSV
4. **Deepfake Prediction** - Run inference on features
5. **Cleanup** - Delete CSV files after processing

## Features

- ✅ Batch processing up to 30 frames per request
- ✅ Automatic OpenFace feature extraction
- ✅ CSV parsing and feature extraction (674 features per frame)
- ✅ Mock deepfake prediction (replace with actual model)
- ✅ Automatic CSV cleanup after processing
- ✅ Configurable model selection
- ✅ Session management for tracking

## Quick Start

### Docker Deployment

```bash
# Build and run
docker-compose -f docker-compose.batch.yml up -d

# Check logs
docker logs batch-deepfake-api -f

# Health check
curl http://localhost:8001/health | jq
```

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt
pip install pandas numpy scikit-learn

# Run server
python batch_api.py

# Server will start at http://0.0.0.0:8001
```

## API Endpoints

### 1. Health Check
```bash
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "openface_binary": "/usr/local/bin/FeatureExtraction",
  "openface_available": true,
  "temp_dir": "/tmp/openface_batch",
  "output_dir": "/data/batch_output",
  "batch_size": 30
}
```

### 2. Batch Prediction
```bash
POST /batch/predict
```

**Parameters:**
- `files` (file[], required): Frame image files (max 30)
- `session_id` (string, optional): Session identifier
- `model_name` (string, optional): Model name (default: "mobilenetv2_celebv2")
- `cleanup` (boolean, optional): Delete CSV after processing (default: true)

**Example Request:**
```bash
curl -X POST http://localhost:8001/batch/predict \
  -F "files=@frame_001.jpg" \
  -F "files=@frame_002.jpg" \
  -F "files=@frame_003.jpg" \
  -F "session_id=video_123" \
  -F "model_name=mobilenetv2_celebv2" \
  -F "cleanup=true"
```

**Response:**
```json
{
  "status": "success",
  "session_id": "video_123",
  "frames_processed": 3,
  "predictions": [
    {
      "frame_id": 0,
      "csv_source": "frame_0000.csv",
      "feature_count": 674,
      "prediction": {
        "score": 0.78,
        "label": "fake",
        "confidence": 0.56
      },
      "model": "mobilenetv2_celebv2"
    },
    {
      "frame_id": 1,
      "csv_source": "frame_0001.csv",
      "feature_count": 674,
      "prediction": {
        "score": 0.45,
        "label": "real",
        "confidence": 0.10
      },
      "model": "mobilenetv2_celebv2"
    }
  ],
  "summary": {
    "total_frames": 3,
    "fake_count": 2,
    "real_count": 1,
    "average_score": 0.623,
    "model_used": "mobilenetv2_celebv2"
  },
  "cleanup": "Deleted 3 CSV files",
  "output_dir": null
}
```

## Testing

### Shell Script
```bash
# Make executable
chmod +x test_batch_api.sh

# Run tests
./test_batch_api.sh
```

### Python Client
```bash
# Install requests
pip install requests

# Run test client
python test_batch_client.py
```

### Manual Testing with curl

**Single frame:**
```bash
curl -X POST http://localhost:8001/batch/predict \
  -F "files=@sample.jpg" \
  -F "session_id=test-001" \
  -F "cleanup=true" | jq
```

**Multiple frames (30):**
```bash
curl -X POST http://localhost:8001/batch/predict \
  -F "session_id=batch-test" \
  $(for i in $(seq -f "%04g" 0 29); do echo "-F files=@frames/frame_$i.jpg"; done) | jq
```

**Keep CSV files:**
```bash
curl -X POST http://localhost:8001/batch/predict \
  -F "files=@sample.jpg" \
  -F "cleanup=false" | jq

# CSV files will be in /data/batch_output/<session_id>/
```

## Python Client Usage

```python
from test_batch_client import BatchDeepfakeClient

# Initialize client
client = BatchDeepfakeClient(base_url="http://localhost:8001")

# Check health
health = client.health_check()
print(health)

# Predict single frame
result = client.predict_batch(
    frame_paths=["frame1.jpg", "frame2.jpg"],
    session_id="my-video",
    model_name="mobilenetv2_celebv2",
    cleanup=True
)

# Predict from directory (auto-detects up to 30 frames)
result = client.predict_from_directory(
    directory="./frames",
    max_frames=30,
    pattern="*.jpg"
)

# Access results
print(f"Frames processed: {result['frames_processed']}")
print(f"Average score: {result['summary']['average_score']}")

for prediction in result['predictions']:
    print(f"Frame {prediction['frame_id']}: {prediction['prediction']['label']}")
```

## Architecture

```
┌──────────────┐
│ Upload Frames│ (up to 30 .jpg/.png)
└──────┬───────┘
       │
       ▼
┌──────────────────┐
│ OpenFace Extract │ (FeatureExtraction binary)
└──────┬───────────┘
       │
       ▼
┌──────────────┐
│ CSV Generated│ (~700 features per frame)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Parse CSV    │ (Extract 674 feature vectors)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ ML Inference │ (Deepfake model prediction)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Return JSON  │ (Predictions + Summary)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Cleanup CSV  │ (Delete if cleanup=true)
└──────────────┘
```

## Configuration

Environment variables:

```bash
# OpenFace binary path
OPENFACE_BINARY=/usr/local/bin/FeatureExtraction

# Temporary storage for uploaded frames
TEMP_DIR=/tmp/openface_batch

# Output directory for CSV files
OUTPUT_DIR=/data/batch_output

# Server configuration
PORT=8001
HOST=0.0.0.0
```

## CSV Feature Format

OpenFace generates CSV with ~700 columns:
- **Metadata** (5 cols): frame, face_id, timestamp, confidence, success
- **Features** (674 cols): gaze, landmarks, pose, action units, etc.

The API skips the first 5 metadata columns and extracts the remaining 674 features for prediction.

## Integration with Deepfake Model

**Current Implementation:**
```python
# Mock prediction (line ~200 in batch_api.py)
mock_score = np.random.uniform(0.3, 0.9)
```

**Replace with actual model:**
```python
# Load your model at startup
import tensorflow as tf
model = tf.keras.models.load_model('path/to/model.h5')

# In batch_predict function:
features_array = np.array(all_features)
predictions = model.predict(features_array)

for idx, pred_score in enumerate(predictions):
    predictions.append({
        "frame_id": idx,
        "prediction": {
            "score": float(pred_score[0]),
            "label": "fake" if pred_score[0] > 0.5 else "real",
            "confidence": float(abs(pred_score[0] - 0.5) * 2)
        }
    })
```

## Performance

- **Batch Size**: 30 frames max per request
- **Processing Time**: ~3-5 seconds per frame (OpenFace extraction)
- **Total Time**: ~90-150 seconds for 30 frames
- **Timeout**: 180 seconds (3 minutes)
- **Throughput**: ~20-30 fps equivalent (batch processing)

## Error Handling

The API handles:
- ✅ Invalid file types (only .jpg, .jpeg, .png, .bmp)
- ✅ Batch size exceeded (max 30 frames)
- ✅ OpenFace extraction failures
- ✅ CSV parsing errors
- ✅ Timeout for long-running batches
- ✅ Cleanup failures (logged as warnings)

## Cleanup Behavior

**cleanup=true (default):**
- CSV files are deleted after prediction
- Temporary frame files always deleted
- Response contains `output_dir: null`

**cleanup=false:**
- CSV files preserved in `/data/batch_output/<session_id>/`
- Useful for debugging or reprocessing
- Response contains full path to output directory

## Troubleshooting

**OpenFace not found:**
```bash
# Check binary exists
docker exec batch-deepfake-api ls -l /usr/local/bin/FeatureExtraction

# Verify environment variable
docker exec batch-deepfake-api env | grep OPENFACE_BINARY
```

**CSV files not generated:**
```bash
# Check OpenFace logs in container
docker logs batch-deepfake-api

# Test OpenFace manually
docker exec -it batch-deepfake-api /usr/local/bin/FeatureExtraction -h
```

**Timeout errors:**
```bash
# Increase timeout in batch_api.py (line ~160)
timeout=300  # 5 minutes

# Or reduce batch size
BATCH_SIZE = 15
```

## Next Steps

1. **Replace Mock Prediction**: Integrate actual deepfake detection model
2. **Add Model Loading**: Load TensorFlow/PyTorch models at startup
3. **Optimize Performance**: Parallel OpenFace processing per frame
4. **Add Caching**: Cache extracted features to avoid re-extraction
5. **Metrics**: Add Prometheus metrics for monitoring
6. **Authentication**: Add API key authentication
7. **Rate Limiting**: Prevent abuse with rate limits

## License

Same as parent project (OpenFace license applies to feature extraction).
