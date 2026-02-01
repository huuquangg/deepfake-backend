#!/bin/bash

set -e

APP_NAME="video-streaming"
BUILD_DIR="bin"
VERSION=$(git describe --tags --always --dirty 2>/dev/null || echo "dev")
BUILD_TIME=$(date -u '+%Y-%m-%d_%H:%M:%S')
GO_VERSION=$(go version | awk '{print $3}')

echo "========================================"
echo "Building $APP_NAME"
echo "========================================"
echo "Version:    $VERSION"
echo "Build Time: $BUILD_TIME"
echo "Go Version: $GO_VERSION"
echo "========================================"

# Create build directory
mkdir -p $BUILD_DIR

# Build with version info
go build \
  -ldflags "-X main.Version=$VERSION -X main.BuildTime=$BUILD_TIME -X main.GoVersion=$GO_VERSION" \
  -o $BUILD_DIR/$APP_NAME \
  ./cmd/server

echo ""
echo "Build complete: $BUILD_DIR/$APP_NAME"

# Make executable
chmod +x $BUILD_DIR/$APP_NAME

# Show file info
ls -lh $BUILD_DIR/$APP_NAME

echo ""
echo "To run: ./$BUILD_DIR/$APP_NAME"