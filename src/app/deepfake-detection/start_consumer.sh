#!/bin/bash
# Start the deepfake detection consumer service

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Set environment variables
export RABBITMQ_URL="${RABBITMQ_URL:-amqp://admin:P@ssw0rd123@localhost:5672/}"
export RABBITMQ_QUEUE="${RABBITMQ_QUEUE:-feature.extraction.results}"
export LOG_LEVEL="${LOG_LEVEL:-INFO}"
export PAD_SEQUENCE="${PAD_SEQUENCE:-1}"
export REQUEUE_ON_ERROR="${REQUEUE_ON_ERROR:-0}"
export DELETE_FRAMES="${DELETE_FRAMES:-0}"

echo "=========================================="
echo "Deepfake Detection Consumer"
echo "=========================================="
echo "RabbitMQ: $RABBITMQ_URL"
echo "Queue: $RABBITMQ_QUEUE"
echo "Log Level: $LOG_LEVEL"
echo "Padding: $PAD_SEQUENCE"
echo "=========================================="
echo ""

# Run consumer (will block and continuously consume messages)
python3 consumer.py
