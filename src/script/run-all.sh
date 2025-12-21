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
PORTS_OPENED=false

# Function to open firewall ports
open_ports() {
    echo -e "${BLUE}Opening firewall ports...${NC}"
    sudo ufw allow 8090/tcp >/dev/null 2>&1  # core-banking
    sudo ufw allow 8091/tcp >/dev/null 2>&1  # video-streaming
    sudo ufw allow 8001/tcp >/dev/null 2>&1  # openface-extraction
    sudo ufw allow 8092/tcp >/dev/null 2>&1  # frequency-extraction
    sudo ufw allow 5672/tcp >/dev/null 2>&1  # rabbitmq
    sudo ufw allow 8093/tcp >/dev/null 2>&1  # socketio-server (deepfake-detection)
    PORTS_OPENED=true
    echo -e "${GREEN}✓ Firewall ports opened (8090, 8091, 8001, 8092, 5672, 8093)${NC}\n"
}

# Function to close firewall ports
close_ports() {
    if [ "$PORTS_OPENED" = true ]; then
        echo -e "${BLUE}Closing firewall ports...${NC}"
        sudo ufw delete allow 8090/tcp >/dev/null 2>&1
        sudo ufw delete allow 8091/tcp >/dev/null 2>&1
        sudo ufw delete allow 8001/tcp >/dev/null 2>&1
        sudo ufw delete allow 8092/tcp >/dev/null 2>&1
        sudo ufw delete allow 5672/tcp >/dev/null 2>&1
        sudo ufw delete allow 8093/tcp >/dev/null 2>&1
        echo -e "${GREEN}✓ Firewall ports closed${NC}"
    fi
}

# Cleanup function
cleanup() {
    echo -e "\n${YELLOW}Shutting down all services...${NC}"
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null
        fi
    done
    
    # Stop Docker Compose services (all compose files)
    cd "$APP_DIR/openface-extraction" 2>/dev/null && {
        docker compose -f docker-compose.yml down 2>/dev/null
        docker compose -f docker-compose.api.yml down 2>/dev/null
        docker compose -f docker-compose.batch.yml down 2>/dev/null
    }
    cd "$APP_DIR/video-streaming" 2>/dev/null && docker compose down 2>/dev/null
    
    # Close firewall ports
    close_ports
    
    echo -e "${GREEN}All services stopped${NC}"
    exit 0
}

# Set up trap for cleanup
trap cleanup SIGINT SIGTERM EXIT

# Open firewall ports before starting services
open_ports

# Function to check if service uses Docker
uses_docker() {
    local service_dir=$1
    if [ -f "$service_dir/docker-compose.yml" ] || \
       [ -f "$service_dir/docker-compose.api.yml" ] || \
       [ -f "$service_dir/docker-compose.batch.yml" ]; then
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

# Function to run Docker service
run_docker_service() {
    local service_name=$1
    local service_dir=$2
    
    echo -e "${BLUE}Starting Docker service: $service_name...${NC}"
    
    cd "$service_dir"
    
    # Find all docker-compose files
    local compose_files=(docker-compose.yml docker-compose.api.yml docker-compose.batch.yml)
    local started_any=false
    
    for compose_file in "${compose_files[@]}"; do
        if [ -f "$compose_file" ]; then
            echo -e "  Found $compose_file..."
            
            # First, stop any existing containers and remove orphans
            if command -v docker &> /dev/null && docker compose version &> /dev/null; then
                echo -e "  Cleaning up existing containers for $compose_file..."
                docker compose -f "$compose_file" down --remove-orphans -v 2>/dev/null
                
                # Start the service
                docker compose -f "$compose_file" up -d --remove-orphans
                if [ $? -eq 0 ]; then
                    echo -e "${GREEN}✓ $service_name started with $compose_file${NC}"
                    started_any=true
                else
                    echo -e "${RED}✗ Failed to start $service_name with $compose_file${NC}"
                fi
            elif command -v docker-compose &> /dev/null; then
                echo -e "  Cleaning up existing containers for $compose_file..."
                docker-compose -f "$compose_file" down --remove-orphans -v 2>/dev/null
                
                # Start the service
                docker-compose -f "$compose_file" up -d --remove-orphans
                if [ $? -eq 0 ]; then
                    echo -e "${GREEN}✓ $service_name started with $compose_file${NC}"
                    started_any=true
                else
                    echo -e "${RED}✗ Failed to start $service_name with $compose_file${NC}"
                fi
            else
                echo -e "${RED}✗ Docker Compose not found${NC}\n"
                return 1
            fi
        fi
    done
    
    if [ "$started_any" = true ]; then
        echo -e "${GREEN}✓ $service_name Docker services started${NC}\n"
    else
        echo -e "${YELLOW}⚠ No docker-compose files found for $service_name${NC}\n"
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

# Function to run Python service
run_python_service() {
    local service_name=$1
    local service_dir=$2
    
    echo -e "${YELLOW}Starting Python service: $service_name...${NC}"
    
    cd "$service_dir"
    
    # Check if batch_api.py exists
    if [ -f "batch_api.py" ]; then
        mkdir -p logs
        # Run in background and capture PID
        python3 batch_api.py > "logs/$service_name.log" 2>&1 &
        local pid=$!
        PIDS+=("$pid")
        SERVICE_NAMES+=("$service_name")
        
        # Wait a moment to check if it started successfully
        sleep 2
        if kill -0 "$pid" 2>/dev/null; then
            echo -e "${GREEN}✓ $service_name started (PID: $pid)${NC}"
            echo -e "  Log: $service_dir/logs/$service_name.log\n"
        else
            echo -e "${RED}✗ $service_name failed to start${NC}\n"
            echo -e "  Check log: $service_dir/logs/$service_name.log\n"
        fi
    else
        echo -e "${RED}✗ batch_api.py not found in $service_dir${NC}\n"
    fi
}

# Function to run consumer service
run_consumer_service() {
    local service_name=$1
    local service_dir=$2
    
    echo -e "${YELLOW}Starting Consumer service: $service_name...${NC}"
    
    cd "$service_dir"
    
    # Check if consumer.py exists
    if [ -f "consumer.py" ]; then
        mkdir -p logs
        
        # Check if start script exists
        if [ -f "start_consumer.sh" ]; then
            ./start_consumer.sh > "logs/$service_name.log" 2>&1 &
        else
            python3 consumer.py > "logs/$service_name.log" 2>&1 &
        fi
        
        local pid=$!
        PIDS+=("$pid")
        SERVICE_NAMES+=("$service_name")
        
        # Wait a moment to check if it started successfully (consumers need more time to load models)
        sleep 5
        if kill -0 "$pid" 2>/dev/null; then
            echo -e "${GREEN}✓ $service_name consumer started (PID: $pid)${NC}"
            echo -e "  Log: $service_dir/logs/$service_name.log\n"
        else
            echo -e "${RED}✗ $service_name consumer failed to start${NC}\n"
            echo -e "  Check log: $service_dir/logs/$service_name.log\n"
        fi
    else
        echo -e "${RED}✗ consumer.py not found in $service_dir${NC}\n"
    fi
}

# Create logs directories
mkdir -p "$APP_DIR/core-banking/logs"
mkdir -p "$APP_DIR/video-streaming/logs"
mkdir -p "$APP_DIR/frequency-extraction/logs"
mkdir -p "$APP_DIR/deepfake-detection/logs"

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
    # Check if it's a consumer service
    elif is_consumer_service "$service_path"; then
        run_consumer_service "$service_name" "$service_path"
    # Check if it's a Python service
    elif is_python_service "$service_path"; then
        run_python_service "$service_name" "$service_path"
    fi
done

echo -e "${GREEN}=== All Services Started ===${NC}"
echo -e "${GREEN}Summary:${NC}"
echo -e "  - Running services: ${#SERVICE_NAMES[@]} (Go + Python + Consumers)"
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
