#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Get the script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
APP_DIR="$PROJECT_ROOT/src/app"

# Service configurations
declare -A SERVICE_PORTS=(
    ["core-banking"]=8090
    ["video-streaming"]=8091
    ["frequency-extraction"]=8092
    ["api-gateway"]=8096
)

declare -A DOCKER_SERVICES=(
    ["openface-extraction"]="openface-extraction"
    ["video-streaming"]="video-streaming"
)

# Function to print header
print_header() {
    clear
    echo -e "${CYAN}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║${NC}          ${GREEN}Service Monitoring Dashboard${NC}                      ${CYAN}║${NC}"
    echo -e "${CYAN}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║${NC} Updated: $(date '+%Y-%m-%d %H:%M:%S')                              ${CYAN}║${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════════╝${NC}\n"
}

# Function to check if port is in use
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        return 0
    fi
    return 1
}

# Function to get process info by port
get_process_info() {
    local port=$1
    local pid=$(lsof -t -i:$port 2>/dev/null | head -n 1)
    if [ -n "$pid" ]; then
        local cpu=$(ps -p $pid -o %cpu= 2>/dev/null | tr -d ' ')
        local mem=$(ps -p $pid -o %mem= 2>/dev/null | tr -d ' ')
        local uptime=$(ps -p $pid -o etime= 2>/dev/null | tr -d ' ')
        echo "$pid|$cpu|$mem|$uptime"
    else
        echo "---"
    fi
}

# Function to monitor Go services
monitor_go_services() {
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  Go Services${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    printf "%-20s %-8s %-10s %-8s %-8s %-12s\n" "SERVICE" "PORT" "STATUS" "PID" "CPU%" "MEM%" "UPTIME"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    for service in "${!SERVICE_PORTS[@]}"; do
        local port=${SERVICE_PORTS[$service]}
        local info=$(get_process_info $port)
        
        if [ "$info" != "---" ]; then
            IFS='|' read -r pid cpu mem uptime <<< "$info"
            echo -e "${GREEN}✓${NC} %-18s %-8s ${GREEN}%-10s${NC} %-8s %-8s %-8s %-12s" \
                "$service" "$port" "RUNNING" "$pid" "$cpu%" "$mem%" "$uptime"
        else
            echo -e "${RED}✗${NC} %-18s %-8s ${RED}%-10s${NC} %-8s %-8s %-8s %-12s" \
                "$service" "$port" "STOPPED" "---" "---" "---" "---"
        fi
    done
    echo ""
}

# Function to monitor Docker services
monitor_docker_services() {
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  Docker Services${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    printf "%-25s %-15s %-15s %-20s\n" "SERVICE" "STATUS" "CONTAINERS" "IMAGE"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    for service in "${!DOCKER_SERVICES[@]}"; do
        local service_dir="$APP_DIR/$service"
        
        if [ -d "$service_dir" ] && [ -f "$service_dir/docker-compose.yml" ]; then
            cd "$service_dir"
            
            # Get container info
            local containers=$(docker compose ps -q 2>/dev/null | wc -l)
            local running=$(docker compose ps --filter "status=running" -q 2>/dev/null | wc -l)
            local images=$(docker compose images -q 2>/dev/null | head -n 1)
            
            if [ "$running" -gt 0 ]; then
                echo -e "${GREEN}✓${NC} %-24s ${GREEN}%-15s${NC} %-15s %-20s" \
                    "$service" "RUNNING" "$running/$containers" "$(docker images --format '{{.Repository}}:{{.Tag}}' | grep $service | head -n 1 | cut -c1-20)"
            elif [ "$containers" -gt 0 ]; then
                echo -e "${YELLOW}⚠${NC} %-24s ${YELLOW}%-15s${NC} %-15s %-20s" \
                    "$service" "PARTIAL" "$running/$containers" "---"
            else
                echo -e "${RED}✗${NC} %-24s ${RED}%-15s${NC} %-15s %-20s" \
                    "$service" "STOPPED" "0/0" "---"
            fi
        fi
    done
    echo ""
}

# Function to show logs tail
show_recent_logs() {
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  Recent Logs (Last 5 lines per service)${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    for service in "${!SERVICE_PORTS[@]}"; do
        local log_file="$APP_DIR/$service/logs/$service.log"
        if [ -f "$log_file" ]; then
            echo -e "${YELLOW}[$service]${NC}"
            tail -n 3 "$log_file" 2>/dev/null | sed 's/^/  /'
            echo ""
        fi
    done
}

# Function to show system resources
show_system_resources() {
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  System Resources${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    # CPU usage
    local cpu_usage=$(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1}')
    echo -e "CPU Usage:    ${CYAN}${cpu_usage}%${NC}"
    
    # Memory usage
    local mem_info=$(free -m | awk 'NR==2{printf "%.1f%%", $3*100/$2 }')
    echo -e "Memory Usage: ${CYAN}${mem_info}${NC}"
    
    # Disk usage
    local disk_usage=$(df -h / | awk 'NR==2{print $5}')
    echo -e "Disk Usage:   ${CYAN}${disk_usage}${NC}"
    
    echo ""
}

# Function to show menu
show_menu() {
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}Commands:${NC} [q]uit | [r]efresh | [l]ogs detail | [s]top all | [h]elp"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

# Main monitoring loop
main() {
    local refresh_interval=5
    local mode="auto"
    
    echo -e "${GREEN}Starting Service Monitor...${NC}"
    echo -e "${YELLOW}Press 'q' to quit, 'r' to refresh manually${NC}\n"
    sleep 2
    
    while true; do
        print_header
        monitor_go_services
        monitor_docker_services
        show_system_resources
        show_recent_logs
        show_menu
        
        if [ "$mode" = "auto" ]; then
            echo -e "\n${CYAN}Auto-refreshing in ${refresh_interval}s...${NC}"
            
            # Read user input with timeout
            read -t $refresh_interval -n 1 key
            
            case "$key" in
                q|Q)
                    echo -e "\n${GREEN}Exiting monitor...${NC}"
                    exit 0
                    ;;
                r|R)
                    continue
                    ;;
                l|L)
                    clear
                    echo -e "${YELLOW}Detailed Logs:${NC}\n"
                    for service in "${!SERVICE_PORTS[@]}"; do
                        local log_file="$APP_DIR/$service/logs/$service.log"
                        if [ -f "$log_file" ]; then
                            echo -e "${CYAN}=== $service ===${NC}"
                            tail -n 20 "$log_file"
                            echo ""
                        fi
                    done
                    read -p "Press Enter to continue..." 
                    ;;
                s|S)
                    echo -e "\n${YELLOW}Stopping all services...${NC}"
                    $SCRIPT_DIR/clean-all.sh
                    echo -e "${GREEN}All services stopped${NC}"
                    exit 0
                    ;;
                h|H)
                    clear
                    echo -e "${CYAN}Help:${NC}"
                    echo -e "  q - Quit monitoring"
                    echo -e "  r - Refresh now"
                    echo -e "  l - Show detailed logs"
                    echo -e "  s - Stop all services"
                    echo -e "  h - Show this help"
                    echo ""
                    read -p "Press Enter to continue..."
                    ;;
            esac
        fi
    done
}

# Run main function
main
