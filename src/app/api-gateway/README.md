# API Gateway (Go)

Reverse proxy that fronts all services in this repo.

## Run

```bash
cd /home/huuquangdang/huu.quang.dang/thesis/deepfake-1801/src/app/api-gateway

go run ./cmd/server
```

Default listen address: `:8096`

## Routes

All routes follow the pattern `/api/{service}/` for consistency:

### Video Streaming Service
- `/api/video-streaming/` → `VIDEO_STREAMING_URL` (default `http://localhost:8091`)
  - `/api/video-streaming/health` - Health check (GET)
  - `/api/video-streaming/ingest/frame` - Ingest single frame (POST)
  - `/api/video-streaming/stats/aggregator` - Aggregator stats (GET)
  - `/api/video-streaming/webrtc/stream/offer` - WebRTC offer (POST)
  - `/api/video-streaming/webrtc/stream/candidate` - WebRTC ICE candidate (POST)
  - `/api/video-streaming/webrtc/stream/{id}/close` - Close WebRTC session (POST)
  - `/api/video-streaming/webrtc/stats` - WebRTC statistics (GET)
  - `/api/video-streaming/sessions/{id}/frames` - Upload frame (POST)
  - `/api/video-streaming/sessions/{id}/frames/batch` - Upload frame batch (POST)
- `/frames/{session_id}/{batch_id}/{filename}` → Static frame serving

### Core Banking Service  
- `/api/core-banking/` → `CORE_BANKING_URL` (default `http://localhost:8090`)
  - `/api/core-banking/auth/register` - User registration (POST)
  - `/api/core-banking/auth/login` - User login (POST)
  - `/api/core-banking/auth/me` - Get current user (GET, JWT required)
  - `/api/core-banking/account/create` - Create account (POST, JWT required)
  - `/api/core-banking/account/info` - Get account info (GET, JWT required)
  - `/api/core-banking/account/balance` - Get balance (GET, JWT required)
  - `/api/core-banking/transaction/transfer` - Transfer funds (POST, JWT required)
  - `/api/core-banking/transaction/history` - Transaction history (GET, JWT required)

### OpenFace Feature Extraction (Single API)
- `/api/openface/` → `OPENFACE_URL` (default `http://localhost:8000`)
  - Prefix `/api/openface` is stripped before forwarding
  - `/api/openface/health` → forwards to `/health`
  - `/api/openface/extract/image` → forwards to `/extract/image` (POST)
  - `/api/openface/extract/video` → forwards to `/extract/video` (POST)
  - `/api/openface/extract/frames` → forwards to `/extract/frames` (POST)
  - `/api/openface/download/{session_id}/{filename}` → forwards to `/download/{session_id}/{filename}` (GET)

### OpenFace Batch API
- `/api/openface-batch/` → `OPENFACE_BATCH_URL` (default `http://localhost:8001`)
  - Prefix `/api/openface-batch` is stripped before forwarding
  - `/api/openface-batch/health` → forwards to `/health`
  - `/api/openface-batch/extract/batch` → forwards to `/extract/batch` (POST)

### Frequency Feature Extraction
- `/api/frequency/` → `FREQUENCY_URL` (default `http://localhost:8092`)
  - Prefix `/api/frequency` is stripped before forwarding
  - `/api/frequency/health` → forwards to `/health`
  - `/api/frequency/extract/batch` → forwards to `/extract/batch` (POST)

### Socket.IO (Real-time Predictions)
- `/socket.io/` → `VIDEO_STREAMING_SOCKET_URL` (default `http://localhost:8093`)
  - WebSocket connection for real-time deepfake detection results

## Environment variables

- `GATEWAY_ADDR` (default `:8096`)
- `VIDEO_STREAMING_URL` (default `http://localhost:8091`)
- `VIDEO_STREAMING_SOCKET_URL` (default `http://localhost:8093`)
- `CORE_BANKING_URL` (default `http://localhost:8090`)
- `OPENFACE_URL` (default `http://localhost:8000`)
- `OPENFACE_BATCH_URL` (default `http://localhost:8001`)
- `FREQUENCY_URL` (default `http://localhost:8002`)

## Health

```bash
curl http://localhost:8096/health
```
