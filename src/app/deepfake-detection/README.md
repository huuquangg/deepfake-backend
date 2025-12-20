# Deepfake Detection Consumer Service

Actively consumes feature extraction results from RabbitMQ and runs deepfake detection inference.

## Overview

This service:
- **Continuously monitors** RabbitMQ queue `feature.extraction.results`
- **Automatically processes** messages as they arrive
- **Fetches frames** from video-streaming HTTP endpoints
- **Runs TCN model** inference (real/fake classification)
- **Logs predictions** with confidence scores
- **Acknowledges** processed messages (or requeues on error)
- **Handles graceful shutdown** on SIGINT/SIGTERM

## Architecture

```
RabbitMQ Queue → Consumer → Frame Fetcher → Input Builder → TCN Model → Log Results → ACK
```

## Quick Start

### Option 1: Run Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Start consumer
./start_consumer.sh

# Or run directly
python3 consumer.py
```

### Option 2: Run with Docker

```bash
# Build and start
docker-compose up -d

# View logs
docker-compose logs -f deepfake-consumer

# Stop
docker-compose down
```

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `RABBITMQ_URL` | `amqp://admin:P@ssw0rd123@localhost:5672/` | RabbitMQ connection URL |
| `RABBITMQ_QUEUE` | `feature.extraction.results` | Queue to consume from |
| `MODEL_PATH` | `./TCN_TemporalConvNet_Residual Blocks_final.h5` | Path to trained model |
| `SCALER_PATH` | `./csv_scaler.pkl` | Path to feature scaler |
| `SEQUENCE_LENGTH` | `15` | Number of frames per inference window |
| `LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `PAD_SEQUENCE` | `1` | Pad sequences shorter than 15 frames |
| `REQUEUE_ON_ERROR` | `0` | Requeue failed messages |
| `DELETE_FRAMES` | `0` | Delete frames after processing |
| `WINDOW_START_MODE` | `min` | Window start mode: `min` or `zero` |
| `FRAME_FETCH_TIMEOUT` | `5` | Frame fetch timeout (seconds) |
| `FRAME_CACHE_SIZE` | `100` | Number of frames to cache |

## Input Message Format

The consumer expects RabbitMQ messages with this structure:

```json
{
  "session_id": "session-123",
  "batch_id": "batch-456",
  "frame_count": 1,
  "frame_refs": [
    {
      "frame_index": 0,
      "uri": "http://localhost:8081/frames/session-123/batch-456/frame_000000.jpg",
      "rel_path": "frames/session-123/batch-456/frame_000000.jpg"
    }
  ],
  "features": [
    {
      "frame_index": 0,
      "_source": "frequency",
      "SRM_mean_1": 0.123,
      ...
    },
    {
      "frame_index": -1,
      "frame": 1,
      "_source": "openface",
      "pose_Tx": 0.456,
      ...
    }
  ]
}
```

## Key Features

### Automatic Frame Index Normalization
- OpenFace may return `frame_index: -1` with `frame: 1`
- Consumer automatically normalizes: `frame_index = frame - 1`

### Padding for Short Sequences
- Model requires 15 frames
- If fewer frames available, repeats last frame
- If no frames available, pads with zeros

### Feature Extraction
- **Frequency**: 283 features (SRM, DCT, FFT)
- **OpenFace**: 674 features (pose, gaze, landmarks, AUs)
- **Total CSV**: 957 features per frame

### Robust Error Handling
- Logs all errors with full context
- Configurable requeue behavior
- Graceful degradation on missing data

### Monitoring & Observability
- Logs every message receipt (RECV)
- Logs predictions with confidence
- Logs ACK/NACK with delivery_tag
- Tracks processing statistics

## Output Logs

Example log output:

```
2024-12-20 10:15:30 - deepfake-detection-consumer - INFO - consumer_started queue=feature.extraction.results (Ctrl+C to stop)
2024-12-20 10:15:35 - deepfake-detection-consumer - INFO - RECV delivery_tag=1 session_id=session-123 batch_id=batch-456 frame_count=1
2024-12-20 10:15:36 - deepfake-detection-consumer - INFO - prediction delivery_tag=1 session_id=session-123 batch_id=batch-456 window_start=0 label=fake prob=0.872345 conf=0.872345 ms=234
2024-12-20 10:15:36 - deepfake-detection-consumer - INFO - ACK delivery_tag=1 session_id=session-123 batch_id=batch-456
```

## Testing

Test with sample JSON (without RabbitMQ):

```bash
python3 test_consumer.py rbmq.json
```

## Graceful Shutdown

Press `Ctrl+C` or send `SIGTERM`:
- Stops consuming new messages
- Waits for current message to complete
- Closes RabbitMQ connection
- Prints final statistics

## Troubleshooting

### Consumer not receiving messages
```bash
# Check RabbitMQ connection
curl -u admin:P@ssw0rd123 http://localhost:15672/api/queues

# Check queue has messages
docker exec rabbitmq rabbitmqctl list_queues

# Check consumer logs
tail -f logs/consumer.log
```

### Model loading fails
```bash
# Verify model file exists
ls -lh "TCN_TemporalConvNet_Residual Blocks_final.h5"

# Check TensorFlow version
python3 -c "import tensorflow as tf; print(tf.__version__)"
```

### Frame fetch failures
```bash
# Verify video-streaming service is running
curl http://localhost:8081/health

# Check frame URI accessibility
curl -I http://localhost:8081/frames/session-123/batch-456/frame_000000.jpg
```

## Integration

This consumer integrates with:
1. **video-streaming**: Provides frame HTTP endpoints
2. **RabbitMQ**: Message queue for feature data
3. **openface-extraction**: Generates facial features
4. **frequency-extraction**: Generates frequency-domain features

## Performance

- **Throughput**: ~4-5 inferences/second (GPU)
- **Latency**: 200-300ms per inference
- **Memory**: ~2GB RAM + ~1GB GPU RAM
- **Cache**: Keeps 100 recent frames in memory

## Production Considerations

1. **Scaling**: Run multiple consumer instances
2. **Monitoring**: Add Prometheus metrics
3. **Alerting**: Monitor error rate and queue depth
4. **Logging**: Ship logs to centralized logging system
5. **GPU**: Requires NVIDIA GPU with CUDA support
