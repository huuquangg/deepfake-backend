#!/bin/bash
# Stop the deepfake detection consumer service

echo "Stopping deepfake detection consumer..."

# Find and kill the consumer process
pkill -f "python3 consumer.py" || pkill -f "python consumer.py"

if [ $? -eq 0 ]; then
    echo "✓ Consumer stopped"
else
    echo "✗ No consumer process found"
fi
