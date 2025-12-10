#!/bin/bash

# Test script for frame upload API
# Tests both single and batch upload endpoints

BASE_URL="http://localhost:8080"
SESSION_ID="test-session-$(date +%s)"

echo "🧪 Testing Frame Upload API"
echo "Session ID: $SESSION_ID"
echo ""

# Create a test image if it doesn't exist
create_test_image() {
    local filename=$1
    if ! command -v convert &> /dev/null; then
        echo "⚠️  ImageMagick not installed. Creating dummy file..."
        # Create a minimal JPEG header for testing
        echo -e "\xFF\xD8\xFF\xE0\x00\x10JFIF" > "$filename"
    else
        convert -size 640x480 xc:blue "$filename"
    fi
}

# Test 1: Single frame upload
echo "📤 Test 1: Single Frame Upload"
create_test_image "test_frame.jpeg"

response=$(curl -s -w "\n%{http_code}" -X POST \
    -F "frame=@test_frame.jpeg" \
    "$BASE_URL/api/v1/sessions/$SESSION_ID/frames")

http_code=$(echo "$response" | tail -n1)
body=$(echo "$response" | head -n-1)

echo "HTTP Status: $http_code"
echo "Response: $body"
echo ""

# Test 2: Batch frame upload (simulating high FPS)
echo "📤 Test 2: Batch Frame Upload (5 frames)"

# Create multiple test frames
for i in {1..5}; do
    create_test_image "test_frame_$i.jpeg"
done

# Prepare curl command with multiple files
curl_cmd="curl -s -w '\n%{http_code}' -X POST"
for i in {1..5}; do
    curl_cmd="$curl_cmd -F 'frames=@test_frame_$i.jpeg'"
done
curl_cmd="$curl_cmd '$BASE_URL/api/v1/sessions/$SESSION_ID/frames/batch'"

response=$(eval $curl_cmd)
http_code=$(echo "$response" | tail -n1)
body=$(echo "$response" | head -n-1)

echo "HTTP Status: $http_code"
echo "Response: $body"
echo ""

# Test 3: High-throughput simulation (20-30 fps)
echo "📤 Test 3: High-throughput Simulation (20 requests in parallel)"
echo "Simulating 20 FPS upload..."

pids=()
for i in {1..20}; do
    create_test_image "test_frame_fps_$i.jpeg"
    (
        curl -s -X POST \
            -F "frame=@test_frame_fps_$i.jpeg" \
            "$BASE_URL/api/v1/sessions/$SESSION_ID/frames" \
            > /dev/null 2>&1
    ) &
    pids+=($!)
done

# Wait for all uploads to complete
for pid in "${pids[@]}"; do
    wait $pid
done

echo "✅ Completed 20 parallel uploads"
echo ""

# Cleanup
echo "🧹 Cleaning up test files..."
rm -f test_frame*.jpeg

# Check stored frames
echo "📁 Checking stored frames in AppDomain/Storage/"
if [ -d "AppDomain/Storage" ]; then
    frame_count=$(ls -1 AppDomain/Storage/${SESSION_ID}_*.jpeg 2>/dev/null | wc -l)
    echo "Found $frame_count frames for session $SESSION_ID"
    
    # Show sample filenames
    echo "Sample filenames:"
    ls -1 AppDomain/Storage/${SESSION_ID}_*.jpeg 2>/dev/null | head -5
else
    echo "⚠️  Storage directory not found"
fi

echo ""
echo "✅ Test completed!"
