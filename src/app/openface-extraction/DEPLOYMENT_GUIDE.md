# OpenFace Batch API Deployment Guide

## Overview
This guide explains how to deploy the OpenFace Batch Feature Extraction API from a fresh clone of the repository.

## Prerequisites
- Docker and Docker Compose installed
- Git
- Minimum 4GB RAM
- Port 8001 available

## Repository Structure
The repository includes:
- `batch_api.py` - FastAPI application for batch OpenFace feature extraction
- `Dockerfile.batch` - Docker container definition
- `docker-compose.batch.yml` - Docker Compose orchestration
- `BATCH_API_README.md` - API usage documentation

**Note**: OpenFace libraries and binaries are NOT included in the repository due to size constraints. They will be downloaded automatically during the Docker build process.

## Deployment Steps

### 1. Clone the Repository
```bash
git clone https://github.com/huuquangg/deepfake-backend.git
cd deepfake-backend/src/app/openface-extraction
```

### 2. Create Required Directories
```bash
mkdir -p data/batch_output data/temp
```

### 3. Build the Docker Image
This step will download the OpenFace base image (~2GB) and install Python dependencies.

```bash
docker compose -f docker-compose.batch.yml build
```

**Expected build time**: 5-10 minutes (first time only)

### 4. Start the Container
```bash
docker compose -f docker-compose.batch.yml up -d
```

### 5. Verify Deployment
```bash
# Check container status
docker ps | grep batch

# Test health endpoint
curl http://localhost:8001/health | jq
```

**Expected response**:
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

### 6. Access API Documentation
Open your browser: http://localhost:8001/docs

## API Usage

### Extract Features from Frames
```bash
curl -X POST http://localhost:8001/extract/batch \
  -F "files=@frame1.jpg" \
  -F "files=@frame2.jpg" \
  -F "files=@frame3.jpg" \
  -F "session_id=my_session" \
  -F "cleanup=true"
```

**Parameters**:
- `files`: Upload up to 30 image files (JPEG/PNG)
- `session_id` (optional): Identifier for organizing output
- `cleanup` (default: true): Auto-delete CSV files after reading

**Response Format**:
```json
{
  "session_id": "my_session",
  "processed": 3,
  "csv_data": [
    {
      "filename": "frame1.csv",
      "headers": ["frame", "face_id", "timestamp", "confidence", "success", ...],
      "num_rows": 1,
      "num_columns": 714,
      "data": [
        {
          "frame": "1",
          "face_id": "0",
          "timestamp": "0.0000",
          "confidence": "0.98",
          ...
        }
      ]
    }
  ]
}
```

## Management Commands

### View Logs
```bash
docker logs batch-deepfake-api -f
```

### Stop Container
```bash
docker compose -f docker-compose.batch.yml down
```

### Restart Container
```bash
docker compose -f docker-compose.batch.yml restart
```

### Rebuild After Code Changes
```bash
docker compose -f docker-compose.batch.yml down
docker compose -f docker-compose.batch.yml build --no-cache
docker compose -f docker-compose.batch.yml up -d
```

## Troubleshooting

### Container Won't Start
```bash
# Check logs for errors
docker logs batch-deepfake-api

# Verify port 8001 is not in use
sudo lsof -i :8001

# Check Docker resources
docker system df
```

### OpenFace Binary Not Found
```bash
# Enter container and verify OpenFace installation
docker exec -it batch-deepfake-api bash
which FeatureExtraction
```

### Out of Memory
Reduce batch size or increase Docker memory limit in Docker Desktop settings.

### Permission Issues with Data Directories
```bash
chmod -R 777 data/
```

## Architecture

### Container Details
- **Base Image**: openface/openface:latest (Ubuntu 18.04, Python 3.6)
- **Python Version**: 3.6 (required by OpenFace base image)
- **Dependencies**: FastAPI 0.68.2, uvicorn 0.15.0, pandas 1.1.5, numpy 1.19.5
- **Port**: 8001
- **Network**: deepfake-network

### OpenFace Features Extracted
The API extracts ALL available OpenFace features:
- **2D Facial Landmarks** (68 points, x/y coordinates)
- **3D Facial Landmarks** (68 points, x/y/z coordinates)
- **PDM Parameters** (34 shape parameters)
- **Head Pose** (rotation, translation)
- **Gaze Direction** (angle and direction vectors)
- **Action Units** (17 AUs, intensity and presence)

Total: ~714 columns per frame

### Data Flow
1. Client uploads up to 30 image frames via POST request
2. API saves frames to temporary directory
3. OpenFace FeatureExtraction processes all frames
4. CSV files are generated in output directory
5. API reads CSV files and converts to JSON
6. CSV files are deleted (if cleanup=true)
7. JSON response returned to client

## Production Considerations

### Security
- Add API authentication (JWT/OAuth2)
- Implement rate limiting
- Use HTTPS with reverse proxy (nginx/traefik)
- Restrict file upload types and sizes

### Monitoring
- Integrate with Prometheus/Grafana
- Add structured logging
- Monitor disk usage in data/batch_output
- Track processing times per batch

### Scaling
- Deploy multiple replicas behind load balancer
- Use Redis for session management
- Implement queue system (Celery/RabbitMQ) for async processing
- Consider Kubernetes for orchestration

### Backup
```bash
# Backup extracted features
tar -czf backup_$(date +%Y%m%d).tar.gz data/batch_output/
```

## Environment Variables

You can customize the API by setting environment variables in `docker-compose.batch.yml`:

```yaml
environment:
  - MAX_BATCH_SIZE=30
  - OPENFACE_BINARY=/usr/local/bin/FeatureExtraction
  - TEMP_DIR=/tmp/openface_batch
  - OUTPUT_DIR=/data/batch_output
```

## Performance

### Benchmarks (tested)
- Single frame: ~0.5-1 second
- 10 frames: ~5-8 seconds
- 30 frames: ~15-25 seconds

*Results may vary based on hardware*

### Optimization Tips
- Use smaller image dimensions (640x480 recommended)
- Process frames in parallel batches
- Pre-validate image format before upload
- Use SSD for data/ directories

## Support

For issues or questions:
1. Check container logs: `docker logs batch-deepfake-api`
2. Review API documentation: http://localhost:8001/docs
3. Verify OpenFace installation inside container
4. Check GitHub issues: https://github.com/huuquangg/deepfake-backend/issues

## License

This deployment guide is part of the deepfake detection project. OpenFace is licensed under Apache 2.0.

---

**Last Updated**: November 2025  
**Tested On**: Ubuntu 20.04+, Docker 24.0+, Docker Compose 2.20+
