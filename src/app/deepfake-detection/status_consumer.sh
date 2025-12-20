#!/bin/bash
# Check deepfake detection consumer status

echo "=== Deepfake Detection Consumer Status ==="
echo ""

# Check if consumer process is running
if pgrep -f "python.*consumer.py" > /dev/null; then
    PID=$(pgrep -f "python.*consumer.py")
    echo "✓ Consumer is RUNNING (PID: $PID)"
    echo ""
    
    # Show process details
    ps -p $PID -o pid,etime,pcpu,pmem,cmd
else
    echo "✗ Consumer is NOT running"
fi

echo ""
echo "=== RabbitMQ Queue Status ==="

# Try to check RabbitMQ queue
if command -v rabbitmqctl &> /dev/null; then
    rabbitmqctl list_queues name messages consumers 2>/dev/null | grep feature.extraction.results || echo "Queue info not available (need sudo or different host)"
elif command -v curl &> /dev/null; then
    curl -s -u admin:P@ssw0rd123 http://localhost:15672/api/queues/%2F/feature.extraction.results 2>/dev/null | python3 -m json.tool 2>/dev/null | grep -E "(name|messages|consumers)" || echo "Queue API not accessible"
else
    echo "Unable to check queue status (install rabbitmqctl or curl)"
fi

echo ""
