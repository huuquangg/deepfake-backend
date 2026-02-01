# Quick Start - Batch API

## Files Added to Git
✅ `batch_api.py` - FastAPI application for batch OpenFace feature extraction
✅ `Dockerfile.batch` - Docker container definition (Python 3.6 compatible)
✅ `docker-compose.batch.yml` - Docker Compose orchestration
✅ `BATCH_API_README.md` - API usage documentation
✅ `DEPLOYMENT_GUIDE.md` - Complete deployment guide
✅ `.gitignore` - Updated to exclude large OpenFace files

## Deployment in 3 Steps

### 1. Build
```bash
docker compose -f docker-compose.batch.yml build
```

### 2. Start
```bash
docker compose -f docker-compose.batch.yml up -d
```

### 3. Test
```bash
curl http://localhost:8001/health | jq
```

## Usage Example
```bash
curl -X POST http://localhost:8001/extract/batch \
  -F "files=@frame1.jpg" \
  -F "files=@frame2.jpg" \
  -F "session_id=test" \
  -F "cleanup=true"
```

## Management
```bash
# View logs
docker logs batch-deepfake-api -f

# Stop
docker compose -f docker-compose.batch.yml down

# Restart
docker compose -f docker-compose.batch.yml restart
```

## Documentation
- **Full Deployment Guide**: `DEPLOYMENT_GUIDE.md`
- **API Documentation**: `BATCH_API_README.md`
- **Interactive Docs**: http://localhost:8001/docs

## Note
OpenFace libraries (~2GB) are **NOT** in git. They will be downloaded automatically during Docker build from the official OpenFace base image.
