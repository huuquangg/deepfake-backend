#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Cleaning All Services ===${NC}\n"

# Get the script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
APP_DIR="$PROJECT_ROOT/src/app"

# Counters for summary
STOPPED_SERVICES=0
CLEANED_BINARIES=0
CLEANED_LOGS=0
CLEANED_DOCKER=0

# Function to stop running processes by port
stop_service_by_port() {
    local port=$1
    local service_name=$2
    
    local pid=$(lsof -t -i:$port 2>/dev/null)
    if [ -n "$pid" ]; then
        echo -e "${YELLOW}Stopping $service_name on port $port (PID: $pid)...${NC}"
        kill -15 "$pid" 2>/dev/null || kill -9 "$pid" 2>/dev/null
        sleep 1
        if ! kill -0 "$pid" 2>/dev/null; then
            echo -e "${GREEN}✓ $service_name stopped${NC}"
            ((STOPPED_SERVICES++))
        else
            echo -e "${RED}✗ Failed to stop $service_name${NC}"
        fi
    fi
}

# Function to clean Docker service
clean_docker_service() {
    local service_name=$1
    local service_dir=$2
    
    echo -e "${BLUE}Cleaning Docker service: $service_name...${NC}"
    
    cd "$service_dir"
    
    if [ -f "docker-compose.yml" ]; then
        # Stop containers
        if command -v docker &> /dev/null && docker compose version &> /dev/null; then
            docker compose down -v 2>/dev/null
        elif command -v docker-compose &> /dev/null; then
            docker-compose down -v 2>/dev/null
        fi
        
        # Remove images
        local images=$(docker images --filter=reference="*$service_name*" -q 2>/dev/null)
        if [ -n "$images" ]; then
            echo -e "${YELLOW}Removing Docker images for $service_name...${NC}"
            docker rmi -f $images 2>/dev/null
        fi
        
        echo -e "${GREEN}✓ $service_name Docker resources cleaned${NC}\n"
        ((CLEANED_DOCKER++))
    fi
}

# Function to clean Go service
clean_go_service() {
    local service_name=$1
    local service_dir=$2
    
    echo -e "${YELLOW}Cleaning Go service: $service_name...${NC}"
    
    cd "$service_dir"
    
    # Remove binaries
    if [ -d "bin" ]; then
        echo -e "  Removing binaries..."
        rm -rf bin
        ((CLEANED_BINARIES++))
    fi
    
    # Remove logs
    if [ -d "logs" ] && [ "$(ls -A logs 2>/dev/null)" ]; then
        echo -e "  Removing logs..."
        rm -rf logs/*.log 2>/dev/null
        ((CLEANED_LOGS++))
    fi
    
    # Remove temporary files
    if [ -d "tmp" ] && [ "$(ls -A tmp 2>/dev/null)" ]; then
        echo -e "  Removing tmp files..."
        rm -rf tmp/*
    fi
    
    echo -e "${GREEN}✓ $service_name cleaned${NC}\n"
}

# Function to check if service uses Docker
uses_docker() {
    local service_dir=$1
    if [ -f "$service_dir/docker-compose.yml" ] || [ -f "$service_dir/Dockerfile" ]; then
        return 0
    fi
    return 1
}

# Function to check if service is a Go service
is_go_service() {
    local service_dir=$1
    if [ -f "$service_dir/go.mod" ]; then
        return 0
    fi
    return 1
}

echo -e "${YELLOW}Step 1: Stopping running services...${NC}\n"

# Stop services by known ports
stop_service_by_port 8090 "core-banking"
stop_service_by_port 8091 "video-streaming"

echo ""

echo -e "${YELLOW}Step 2: Cleaning service artifacts...${NC}\n"

# Clean services
cd "$APP_DIR"

for service in */; do
    service_name="${service%/}"
    service_path="$APP_DIR/$service_name"
    
    # Check if it uses Docker
    if uses_docker "$service_path"; then
        clean_docker_service "$service_name" "$service_path"
    fi
    
    # Check if it's a Go service
    if is_go_service "$service_path"; then
        clean_go_service "$service_name" "$service_path"
    fi
done

echo -e "${YELLOW}Step 3: Cleaning Docker system...${NC}\n"

# Clean dangling images and build cache
if command -v docker &> /dev/null; then
    echo -e "Removing dangling images..."
    docker image prune -f 2>/dev/null
    
    echo -e "Cleaning build cache..."
    docker builder prune -f 2>/dev/null
    
    echo -e "${GREEN}✓ Docker system cleaned${NC}\n"
fi

echo -e "${GREEN}=== Cleanup Complete ===${NC}"
echo -e "${GREEN}Summary:${NC}"
echo -e "  - Stopped services: $STOPPED_SERVICES"
echo -e "  - Cleaned binaries: $CLEANED_BINARIES"
echo -e "  - Cleaned logs: $CLEANED_LOGS"
echo -e "  - Cleaned Docker services: $CLEANED_DOCKER"
echo -e "\n${GREEN}All services and build artifacts have been cleaned${NC}"
