#!/bin/bash

set -e

echo "========================================"
echo "Running Tests"
echo "========================================"

# Run tests with race detection and coverage
go test -v -race -coverprofile=coverage.out ./...

# Generate HTML coverage report
go tool cover -html=coverage.out -o coverage.html

# Show coverage summary
echo ""
echo "========================================"
echo "Coverage Summary"
echo "========================================"
go tool cover -func=coverage.out

echo ""
echo "Coverage report generated: coverage.html"
echo "Open it in browser: open coverage.html"