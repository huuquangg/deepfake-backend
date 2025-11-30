# OpenFace Feature Extraction API

Minimal FastAPI wrapper for OpenFace feature extraction from images and videos.

## Features

- ✅ Extract features from single images
- ✅ Extract features from videos
- ✅ Batch processing for frame directories
- ✅ RESTful API with FastAPI
- ✅ Docker support
- ✅ Automatic CSV output

## Extracted Features

OpenFace extracts:
- **2D Facial Landmarks** (68 points)
- **3D Facial Landmarks**
- **Head Pose** (pitch, yaw, roll, position)
- **Action Units** (AU intensities and presence)
- **Gaze Direction**
- **PDM Parameters**

## Installation

### Docker (Recommended)

```bash
# Build and run
docker-compose -f docker-compose.api.yml up -d

# View logs
docker-compose -f docker-compose.api.yml logs -f

# Stop
docker-compose -f docker-compose.api.yml down
```

### Local (requires OpenFace installed)

```bash
# Install dependencies
pip install -r requirements.txt

# Run API
python3 api.py
```

## API Endpoints

### Health Check
```bash
GET http://localhost:8000/health
```

### Extract from Image
```bash
curl -X POST http://localhost:8000/extract/image \
  -F "file=@image.jpg" \
  -F "session_id=test-session"
```

**Response:**
```json
{
  "status": "success",
  "message": "Feature extraction completed",
  "session_id": "test-session",
  "input_file": "image.jpg",
  "output_file": "/data/output/test-session/image.csv",
  "output_format": "csv"
}
```

### Extract from Video
```bash
curl -X POST http://localhost:8000/extract/video \
  -F "file=@video.mp4" \
  -F "session_id=test-video"
```

**Response:**
```json
{
  "status": "success",
  "message": "Video feature extraction completed",
  "session_id": "test-video",
  "input_file": "video.mp4",
  "output_file": "/data/output/test-video/video.csv",
  "output_format": "csv"
}
```

### Extract from Frame Directory
```bash
curl -X POST http://localhost:8000/extract/frames \
  -F "session_id=my-session" \
  -F "frames_dir=/path/to/frames"
```

**Response:**
```json
{
  "status": "success",
  "message": "Batch feature extraction completed",
  "session_id": "my-session",
  "frames_processed": 100,
  "output_files": [
    "/data/output/my-session/frame_0001.csv",
    "/data/output/my-session/frame_0002.csv"
  ],
  "output_dir": "/data/output/my-session"
}
```

### Download Output File
```bash
GET http://localhost:8000/download/{session_id}/{filename}

# Example
curl -O http://localhost:8000/download/test-session/image.csv
```

## Integration with Video Streaming API

To integrate with the video streaming service that uploads frames:

```bash
# 1. Frames are uploaded to [Storage]/session-id/
# 2. Call OpenFace extraction API
curl -X POST http://localhost:8000/extract/frames \
  -F "session_id=session-123" \
  -F "frames_dir=[Storage]/session-123"

# 3. Retrieve CSV outputs from /data/output/session-123/
```

## Output Format

OpenFace generates CSV files with columns:

```csv
frame, timestamp, confidence, success,
gaze_0_x, gaze_0_y, gaze_0_z, gaze_1_x, gaze_1_y, gaze_1_z,
pose_Tx, pose_Ty, pose_Tz, pose_Rx, pose_Ry, pose_Rz,
x_0, x_1, ..., x_67, y_0, y_1, ..., y_67,
X_0, X_1, ..., X_67, Y_0, Y_1, ..., Y_67, Z_0, Z_1, ..., Z_67,
AU01_r, AU02_r, AU04_r, ..., AU45_c
```

Total: ~700 columns per frame

## Environment Variables

- `OPENFACE_BINARY`: Path to FeatureExtraction binary (default: `/usr/local/bin/FeatureExtraction`)
- `TEMP_DIR`: Temporary directory for uploads (default: `/tmp/openface`)
- `OUTPUT_DIR`: Output directory for CSV files (default: `/data/output`)
- `PORT`: API port (default: `8000`)
- `HOST`: API host (default: `0.0.0.0`)

## Performance

- **Single Image**: ~1-2 seconds
- **Video (30fps, 10s)**: ~30-60 seconds
- **Batch (100 frames)**: ~60-120 seconds

## Supported File Formats

- **Images**: JPG, JPEG, PNG, BMP
- **Videos**: MP4, AVI, MOV, MKV

## Error Handling

The API returns appropriate HTTP status codes:
- `200`: Success
- `400`: Invalid input
- `404`: File not found
- `500`: Processing error
- `504`: Timeout

## Notes

- Session IDs are used to organize outputs
- Temporary files are cleaned up after processing
- Videos have a 5-minute timeout
- Batch processing has a 10-minute timeout
- Output CSV files are stored in `/data/output/{session_id}/`
