# API Gateway (Go)

Reverse proxy that fronts all services in this repo.

## Run

```bash
cd /home/huuquangdang/huu.quang.dang/thesis/deepfake-1801/src/app/api-gateway

go run ./cmd/server
```

Default listen address: `:8096`

## Routes

- `/api/video-streaming/` -> `VIDEO_STREAMING_URL` (default `http://localhost:8091`)
- `/frames/` -> `VIDEO_STREAMING_URL` (default `http://localhost:8091`)
- `/api/core-banking/` -> `CORE_BANKING_URL` (default `http://localhost:8090`)
- `/api/openface/` -> `OPENFACE_URL` (default `http://localhost:8000`, prefix stripped)
- `/api/openface-batch/` -> `OPENFACE_BATCH_URL` (default `http://localhost:8001`, prefix stripped)
- `/api/frequency/` -> `FREQUENCY_URL` (default `http://localhost:8002`, prefix stripped)
- `/socket.io/` -> `VIDEO_STREAMING_SOCKET_URL` (default `http://localhost:8093`)

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
