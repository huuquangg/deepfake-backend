#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse command line arguments
FORCE_DOCKER=false
if [[ "$1" == "--force-docker" ]] || [[ "$1" == "-f" ]]; then
    FORCE_DOCKER=true
fi

echo -e "${GREEN}=== Building All Services ===${NC}\n"

if [ "$FORCE_DOCKER" = true ]; then
    echo -e "${YELLOW}Force rebuilding Docker images...${NC}\n"
fi

# Get the script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
APP_DIR="$PROJECT_ROOT/src/app"

# Counters for summary
GO_BUILT=0
DOCKER_BUILT=0
DOCKER_SKIPPED=0
PYTHON_READY=0
CONSUMER_READY=0
SKIPPED=0

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

# Function to check if service is a Python service
is_python_service() {
    local service_dir=$1
    if [ -f "$service_dir/requirements.txt" ] && [ -f "$service_dir/batch_api.py" ]; then
        return 0
    fi
    return 1
}

# Function to check if service is a consumer service
is_consumer_service() {
    local service_dir=$1
    if [ -f "$service_dir/requirements.txt" ] && [ -f "$service_dir/consumer.py" ]; then
        return 0
    fi
    return 1
}

# Function to check if Docker image exists
docker_image_exists() {
    local service_name=$1
    local image_name=""
    
    # Check common image naming patterns
    if docker images --format "{{.Repository}}:{{.Tag}}" | grep -q "^${service_name}:latest$"; then
        return 0
    elif docker images --format "{{.Repository}}:{{.Tag}}" | grep -q "^.*/${service_name}:latest$"; then
        return 0
    elif docker images --format "{{.Repository}}:{{.Tag}}" | grep -q "^${service_name}-app$"; then
        return 0
    fi
    
    return 1
}

# Function to build Docker service
build_docker_service() {
    local service_name=$1
    local service_dir=$2
    
    cd "$service_dir"
    
    # Check if image already exists and force flag is not set
    if [ "$FORCE_DOCKER" = false ] && docker_image_exists "$service_name"; then
        echo -e "${YELLOW}Skipping $service_name - Docker image already exists${NC}"
        echo -e "  ${BLUE}Tip: Use --force-docker to rebuild${NC}\n"
        ((DOCKER_SKIPPED++))
        return 0
    fi
    
    echo -e "${BLUE}Building Docker image for $service_name...${NC}"
    
    # Check if docker-compose.yml exists
    if [ -f "docker-compose.yml" ]; then
        # Try docker compose (new) first, fall back to docker-compose (old)
        if command -v docker &> /dev/null && docker compose version &> /dev/null; then
            if docker compose build 2>&1; then
                echo -e "${GREEN}✓ $service_name Docker image built successfully${NC}\n"
                ((DOCKER_BUILT++))
            else
                echo -e "${RED}✗ Failed to build Docker image for $service_name${NC}\n"
                return 0
            fi
        elif command -v docker-compose &> /dev/null; then
            if docker-compose build 2>&1; then
                echo -e "${GREEN}✓ $service_name Docker image built successfully${NC}\n"
                ((DOCKER_BUILT++))
            else
                echo -e "${RED}✗ Failed to build Docker image for $service_name${NC}\n"
                return 0
            fi
        else
            echo -e "${RED}✗ Neither 'docker compose' nor 'docker-compose' command found${NC}\n"
            return 0
        fi
    elif [ -f "Dockerfile" ]; then
        if docker build -t "$service_name:latest" . 2>&1; then
            echo -e "${GREEN}✓ $service_name Docker image built successfully${NC}\n"
            ((DOCKER_BUILT++))
        else
            echo -e "${RED}✗ Failed to build Docker image for $service_name${NC}\n"
            return 0
        fi
    else
        echo -e "${RED}✗ No Dockerfile or docker-compose.yml found for $service_name${NC}\n"
        return 0
    fi
}

# Function to build Go service
build_go_service() {
    local service_name=$1
    local service_dir=$2
    
    echo -e "${YELLOW}Building Go binary for $service_name...${NC}"
    
    cd "$service_dir"
    
    # Check if cmd/server exists
    if [ -d "cmd/server" ]; then
        mkdir -p bin
        if go build -o "bin/$service_name" ./cmd/server 2>&1; then
            echo -e "${GREEN}✓ $service_name binary built successfully${NC}\n"
            ((GO_BUILT++))
        else
            echo -e "${RED}✗ Failed to build $service_name binary${NC}\n"
            return 0  # Continue with other services
        fi
    # Check for main.go at service root (e.g., api-gateway)
    elif [ -f "main.go" ]; then
        mkdir -p bin
        if go build -o "bin/$service_name" . 2>&1; then
            echo -e "${GREEN}✓ $service_name binary built successfully${NC}\n"
            ((GO_BUILT++))
        else
            echo -e "${RED}✗ Failed to build $service_name binary${NC}\n"
            return 0
        fi
    else
        echo -e "${RED}✗ cmd/server directory not found for $service_name${NC}\n"
        return 0
    fi
}

# Function to check Python service dependencies
check_python_service() {
    local service_name=$1
    local service_dir=$2
    
    echo -e "${BLUE}Checking Python service: $service_name...${NC}"
    
    cd "$service_dir"
    
    # Check if requirements.txt exists
    if [ -f "requirements.txt" ]; then
        echo -e "  Found requirements.txt"
        echo -e "  ${YELLOW}Note: Install dependencies with: pip install -r requirements.txt${NC}"
        echo -e "${GREEN}✓ $service_name ready (Python service - no build needed)${NC}\n"
        ((PYTHON_READY++))
    else
        echo -e "${RED}✗ requirements.txt not found for $service_name${NC}\n"
        return 0
    fi
}

# Build services
cd "$APP_DIR"

for service in */; do
    service_name="${service%/}"
    service_path="$APP_DIR/$service_name"
    
    echo -e "${YELLOW}Checking $service_name...${NC}"
    
    # Check if it uses Docker
    if uses_docker "$service_path"; then
        build_docker_service "$service_name" "$service_path"
    # Check if it's a Go service
    elif is_go_service "$service_path"; then
        build_go_service "$service_name" "$service_path"
    # Check if it's a consumer service
    elif is_consumer_service "$service_path"; then
        check_python_service "$service_name" "$service_path"
        ((CONSUMER_READY++))
    # Check if it's a Python service
    elif is_python_service "$service_path"; then
        check_python_service "$service_name" "$service_path"
    else
        echo -e "${YELLOW}$service_name - no build configuration found - skipping${NC}\n"
        ((SKIPPED++))
    fi
done

echo -e "${GREEN}=== Build Complete ===${NC}"
echo -e "${GREEN}Summary:${NC}"
echo -e "  - Docker images built: $DOCKER_BUILT"
echo -e "  - Docker images skipped: $DOCKER_SKIPPED"
echo -e "  - Go binaries built: $GO_BUILT"
echo -e "  - Python services ready: $PYTHON_READY"
echo -e "  - Consumer services ready: $CONSUMER_READY"
echo -e "  - Skipped: $SKIPPED"

if [ $DOCKER_SKIPPED -gt 0 ]; then
    echo -e "\n${BLUE}Note: Use './src/script/build-all.sh --force-docker' to rebuild Docker images${NC}"
fi
