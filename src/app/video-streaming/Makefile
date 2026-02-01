.PHONY: help build run test clean install-deps check-ffmpeg migrate docker-build docker-run

APP_NAME=video-streaming
BUILD_DIR=bin
CMD_DIR=cmd/server

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install-deps: ## Install Go dependencies
	go mod download
	go mod tidy

build: ## Build the application
	@./scripts/build.sh

run: ## Run the application
	go run ./$(CMD_DIR)

test: ## Run tests with coverage
	@./scripts/test.sh

clean: ## Clean build artifacts
	rm -rf $(BUILD_DIR)
	rm -rf tmp
	rm -f coverage.out coverage.html

check-ffmpeg: ## Check if FFmpeg is installed
	@which ffmpeg > /dev/null || (echo "FFmpeg is not installed" && exit 1)
	@echo "FFmpeg is installed: $(shell ffmpeg -version | head -n 1)"

migrate: ## Run database migrations
	@echo "Running database migrations..."
	@psql -h localhost -U postgres -d streaming_db -f scripts/migrate.sql

docker-build: ## Build Docker image
	docker build -t $(APP_NAME):latest .

docker-run: ## Run with Docker Compose
	docker-compose up -d

docker-down: ## Stop Docker Compose
	docker-compose down

docker-logs: ## Show Docker logs
	docker-compose logs -f app

fmt: ## Format code
	go fmt ./...

lint: ## Run linter
	golangci-lint run ./...

dev: check-ffmpeg ## Run in development mode
	@echo "Starting development server..."
	go run ./$(CMD_DIR)