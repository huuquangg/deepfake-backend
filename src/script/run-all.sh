#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Running All Services ===${NC}\n"

# Get the script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
APP_DIR="$PROJECT_ROOT/src/app"

# Array to track background processes
declare -a PIDS=()
declare -a SERVICE_NAMES=()

# Cleanup function
cleanup() {
    echo -e "\n${YELLOW}Shutting down all services...${NC}"
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null
        fi
    done
    
    # Stop Docker Compose services
    cd "$APP_DIR/openface-extraction" 2>/dev/null && docker compose down 2>/dev/null
    cd "$APP_DIR/video-streaming" 2>/dev/null && docker compose down 2>/dev/null
    
    echo -e "${GREEN}All services stopped${NC}"
    exit 0
}

# Set up trap for cleanup
trap cleanup SIGINT SIGTERM EXIT

# Function to check if service uses Docker
uses_docker() {
    local service_dir=$1
    if [ -f "$service_dir/docker-compose.yml" ]; then
        return 0
    fi
    return 1
}

# Function to check if service is a Go service
is_go_service() {
    local service_dir=$1
    if [ -f "$service_dir/go.mod" ] && [ -f "$service_dir/bin/"* ] 2>/dev/null; then
        return 0
    fi
    return 1
}

# Function to run Docker service
run_docker_service() {
    local service_name=$1
    local service_dir=$2
    
    echo -e "${BLUE}Starting Docker service: $service_name...${NC}"
    
    cd "$service_dir"
    
    if [ -f "docker-compose.yml" ]; then
        # First, stop any existing containers and remove orphans
        if command -v docker &> /dev/null && docker compose version &> /dev/null; then
            echo -e "  Cleaning up existing containers..."
            docker compose down --remove-orphans -v 2>/dev/null
            
            # Force remove any conflicting containers
            local containers=$(docker ps -a --filter "name=$service_name" -q 2>/dev/null)
            if [ -n "$containers" ]; then
                echo -e "  Removing conflicting containers..."
                docker rm -f $containers 2>/dev/null
            fi
            
            # Also check for common container names
            docker rm -f openface 2>/dev/null
            docker rm -f batch-deepfake-api 2>/dev/null
            docker rm -f openface-extraction-api 2>/dev/null
            
            # Start the service
            docker compose up -d --remove-orphans
            if [ $? -eq 0 ]; then
                echo -e "${GREEN}✓ $service_name started (Docker Compose)${NC}\n"
            else
                echo -e "${RED}✗ Failed to start $service_name${NC}\n"
            fi
        elif command -v docker-compose &> /dev/null; then
            echo -e "  Cleaning up existing containers..."
            docker-compose down --remove-orphans -v 2>/dev/null
            
            # Force remove any conflicting containers
            local containers=$(docker ps -a --filter "name=$service_name" -q 2>/dev/null)
            if [ -n "$containers" ]; then
                echo -e "  Removing conflicting containers..."
                docker rm -f $containers 2>/dev/null
            fi
            
            # Also check for common container names
            docker rm -f openface 2>/dev/null
            docker rm -f batch-deepfake-api 2>/dev/null
            docker rm -f openface-extraction-api 2>/dev/null
            
            # Start the service
            docker-compose up -d --remove-orphans
            if [ $? -eq 0 ]; then
                echo -e "${GREEN}✓ $service_name started (docker-compose)${NC}\n"
            else
                echo -e "${RED}✗ Failed to start $service_name${NC}\n"
            fi
        else
            echo -e "${RED}✗ Docker Compose not found${NC}\n"
        fi
    fi
}

# Function to run Go service
run_go_service() {
    local service_name=$1
    local service_dir=$2
    
    echo -e "${YELLOW}Starting Go service: $service_name...${NC}"
    
    cd "$service_dir"
    
    # Find the binary
    local binary_path=$(find bin -type f -executable 2>/dev/null | head -n 1)
    
    if [ -n "$binary_path" ]; then
        # Run in background and capture PID
        ./"$binary_path" > "logs/$service_name.log" 2>&1 &
        local pid=$!
        PIDS+=("$pid")
        SERVICE_NAMES+=("$service_name")
        
        # Wait a moment to check if it started successfully
        sleep 1
        if kill -0 "$pid" 2>/dev/null; then
            echo -e "${GREEN}✓ $service_name started (PID: $pid)${NC}"
            echo -e "  Log: $service_dir/logs/$service_name.log\n"
        else
            echo -e "${RED}✗ $service_name failed to start${NC}\n"
        fi
    else
        echo -e "${RED}✗ No binary found in $service_dir/bin${NC}\n"
    fi
}

# Create logs directories
mkdir -p "$APP_DIR/core-banking/logs"
mkdir -p "$APP_DIR/video-streaming/logs"

# Run services
cd "$APP_DIR"

for service in */; do
    service_name="${service%/}"
    service_path="$APP_DIR/$service_name"
    
    # Check if it uses Docker
    if uses_docker "$service_path"; then
        run_docker_service "$service_name" "$service_path"
    # Check if it's a Go service
    elif is_go_service "$service_path"; then
        run_go_service "$service_name" "$service_path"
    fi
done

echo -e "${GREEN}=== All Services Started ===${NC}"
echo -e "${GREEN}Summary:${NC}"
echo -e "  - Running services: ${#SERVICE_NAMES[@]} Go binaries"
echo -e "  - Docker services: Check with 'docker compose ps' in service directories"
echo -e "\n${YELLOW}Press Ctrl+C to stop all services${NC}\n"

# Monitor services
while true; do
    sleep 5
    
    # Check if any Go service died
    for i in "${!PIDS[@]}"; do
        pid="${PIDS[$i]}"
        service_name="${SERVICE_NAMES[$i]}"
        
        if ! kill -0 "$pid" 2>/dev/null; then
            echo -e "${RED}✗ Service $service_name (PID: $pid) has stopped${NC}"
            unset 'PIDS[$i]'
            unset 'SERVICE_NAMES[$i]'
        fi
    done
    
    # If all services stopped, exit
    if [ ${#PIDS[@]} -eq 0 ]; then
        echo -e "${YELLOW}All Go services have stopped${NC}"
        break
    fi
done
