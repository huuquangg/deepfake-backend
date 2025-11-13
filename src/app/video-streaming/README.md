# Video Streaming Frame Extraction Service

A production-ready Go service for streaming video and extracting frames at 30fps using FFmpeg with PostgreSQL persistence and automatic cleanup.

## 🚀 Features

- ✅ Real-time video frame extraction at 30fps
- ✅ FFmpeg integration with MJPEG codec
- ✅ UUID-based frame naming
- ✅ Automatic cleanup with 30-second sliding window
- ✅ PostgreSQL database for session persistence
- ✅ RESTful API with proper DTOs
- ✅ Repository pattern for data access
- ✅ Multiple concurrent session support
- ✅ Structured logging and error handling
- ✅ Docker support
- ✅ Clean architecture following Go best practices

## 📁 Project Structure

```
video-streaming/
├── cmd/server/              # Application entry point
├── internal/
│   ├── api/                # HTTP handlers, routes, middleware
│   ├── config/             # Configuration management
│   ├── db/                 # Database connection
│   ├── dto/                # Data Transfer Objects
│   ├── models/             # Domain models
│   ├── repository/         # Data access layer
│   └── service/            # Business logic
├── pkg/ffmpeg/             # Reusable FFmpeg utilities
├── scripts/                # Build, test, and migration scripts
├── docker-compose.yml      # Docker Compose configuration
├── Dockerfile              # Docker image definition
├── Makefile                # Build automation
└── README.md               # This file
```

## 🛠️ Prerequisites

- **Go 1.24+**
- **PostgreSQL 14+**
- **FFmpeg** (with MJPEG support)
- **Docker** (optional)

### Install Dependencies

**FFmpeg:**
```bash
# Ubuntu/Debian
sudo apt update && sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Verify
make check-ffmpeg
```

**PostgreSQL:**
```bash
# Ubuntu/Debian
sudo apt install postgresql postgresql-contrib

# macOS
brew install postgresql
```

## 🚀 Quick Start

### Option 1: Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/huuquangdang/video-streaming.git
cd video-streaming

# 2. Install dependencies
make install-deps

# 3. Setup environment
cp .env.example .env
# Edit .env with your configuration

# 4. Setup database
createdb streaming_db
make migrate

# 5. Build and run
make build
./bin/video-streaming
```

### Option 2: Docker Setup

```bash
# 1. Clone the repository
git clone https://github.com/huuquangdang/video-streaming.git
cd video-streaming

# 2. Run with Docker Compose
make docker-run

# 3. Check logs
make docker-logs

# 4. Stop
make docker-down
```

## 📖 API Documentation

Base URL: `http://localhost:8080`

### Health Check

```bash
GET /health
```

Response:
```json
{
  "status": "healthy",
  "timestamp": "2025-11-08T09:22:20Z",
  "version": "1.0.0"
}
```

### Create Session

```bash
POST /api/v1/sessions
Content-Type: application/json

{
  "user_id": "user123"
}
```

Response:
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "ready",
  "created_at": "2025-11-08T09:22:20Z"
}
```

### Upload Video

```bash
POST /api/v1/sessions/{session_id}
Content-Type: video/mp4

[Binary video data]
```

Response:
```json
{
  "message": "Video processing started",
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "processing"
}
```

### Get Session

```bash
GET /api/v1/sessions/{session_id}
```

Response:
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "user_id": "user123",
  "video_path": "/tmp/.../input_video.mp4",
  "status": "processing",
  "created_at": "2025-11-08T09:22:20Z",
  "updated_at": "2025-11-08T09:23:00Z"
}
```

### Delete Session

```bash
DELETE /api/v1/sessions/{session_id}
```

Response:
```json
{
  "message": "Session deleted successfully"
}
```

## 🧪 Testing

```bash
# Run all tests
make test

# Run specific package tests
go test -v ./internal/service/...

# Run with coverage
go test -cover ./...
```

## ⚙️ Configuration

Configuration is done via environment variables. See `.env.example`:

| Variable | Default | Description |
|----------|---------|-------------|
| `SERVER_ADDRESS` | `:8080` | Server address |
| `TMP_DIR` | `tmp` | Temporary directory for frames |
| `FRAME_RATE` | `30` | Frames per second extraction |
| `JPEG_QUALITY` | `10` | JPEG quality (2-31, higher = lower) |
| `CLEANUP_WINDOW` | `30s` | Frame retention time |
| `CLEANUP_INTERVAL` | `5s` | Cleanup check interval |
| `POSTGRES_HOST` | `localhost` | PostgreSQL host |
| `POSTGRES_PORT` | `5432` | PostgreSQL port |
| `POSTGRES_USER` | `postgres` | PostgreSQL user |
| `POSTGRES_PASSWORD` | `postgres` | PostgreSQL password |
| `POSTGRES_DB` | `streaming_db` | PostgreSQL database name |

## 🐳 Docker Commands

```bash
# Build image
make docker-build

# Run with docker-compose
make docker-run

# View logs
make docker-logs

# Stop containers
make docker-down
```

## 📝 Development

```bash
# Format code
make fmt

# Run linter
make lint

# Build
make build

# Run in development mode
make dev

# Clean build artifacts
make clean
```

## 🗃️ Database Schema

```sql
-- Sessions table
CREATE TABLE sessions (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    video_path TEXT,
    status TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL
);

-- Frames table (optional)
CREATE TABLE frames (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    filename TEXT NOT NULL,
    file_path TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL
);
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

MIT License - see LICENSE file for details

## 👤 Author

**quang-dang_opswat**

## 🙏 Acknowledgments

- FFmpeg for video processing
- PostgreSQL for data persistence
- Go community for excellent libraries


# Navigate to your project directory
cd /home/huuquangdang/huu.quang.dang/thesis/deepfake-1801/src/app/video-streaming

# Make all scripts executable
chmod +x scripts/*.sh

# Step 1: Setup database (create DB and run migrations)
./scripts/setup-db.sh

# Or manually:
createdb -U postgres streaming_db
psql -U postgres -d streaming_db -f scripts/migrate.sql

# Step 2: Build the application
./scripts/build.sh

# Step 3: Run the application
./bin/video-streaming

# OR run directly without building
go run cmd/server/main.go