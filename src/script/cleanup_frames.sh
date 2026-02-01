#!/bin/bash
# Frame Storage Cleanup Script
#
# Cleans up old frame storage directories to prevent disk space issues.
# Run periodically via cron or systemd timer.
#
# Usage:
#   ./cleanup_frames.sh [max_age_hours]
#
# Default: Remove frames older than 24 hours

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Configuration
STORAGE_DIR="${FRAME_STORAGE_DIR:-Storage/frames}"
MAX_AGE_HOURS="${1:-24}"
DRY_RUN="${DRY_RUN:-false}"

echo -e "${GREEN}=== Frame Storage Cleanup ===${NC}\n"
echo "Storage directory: $STORAGE_DIR"
echo "Max age: $MAX_AGE_HOURS hours"
echo "Dry run: $DRY_RUN"
echo ""

# Check if storage directory exists
if [ ! -d "$STORAGE_DIR" ]; then
    echo -e "${YELLOW}Storage directory does not exist: $STORAGE_DIR${NC}"
    exit 0
fi

# Find and remove old directories
echo -e "${YELLOW}Finding directories older than $MAX_AGE_HOURS hours...${NC}"

FOUND_COUNT=0
DELETED_COUNT=0
FREED_BYTES=0

while IFS= read -r -d '' session_dir; do
    # Get directory age in hours
    AGE_SECONDS=$(( $(date +%s) - $(stat -c %Y "$session_dir") ))
    AGE_HOURS=$(( AGE_SECONDS / 3600 ))
    
    if [ $AGE_HOURS -ge $MAX_AGE_HOURS ]; then
        FOUND_COUNT=$((FOUND_COUNT + 1))
        
        # Calculate size
        DIR_SIZE=$(du -sb "$session_dir" 2>/dev/null | cut -f1)
        DIR_SIZE_MB=$(( DIR_SIZE / 1048576 ))
        
        if [ "$DRY_RUN" = "true" ]; then
            echo -e "${YELLOW}[DRY RUN]${NC} Would delete: $session_dir (${DIR_SIZE_MB}MB, ${AGE_HOURS}h old)"
        else
            echo "Deleting: $session_dir (${DIR_SIZE_MB}MB, ${AGE_HOURS}h old)"
            rm -rf "$session_dir"
            
            if [ $? -eq 0 ]; then
                DELETED_COUNT=$((DELETED_COUNT + 1))
                FREED_BYTES=$((FREED_BYTES + DIR_SIZE))
            else
                echo -e "${RED}Failed to delete: $session_dir${NC}"
            fi
        fi
    fi
done < <(find "$STORAGE_DIR" -mindepth 1 -maxdepth 1 -type d -print0)

# Summary
FREED_MB=$(( FREED_BYTES / 1048576 ))
FREED_GB=$(( FREED_BYTES / 1073741824 ))

echo ""
echo -e "${GREEN}=== Cleanup Summary ===${NC}"
echo "Directories found (> ${MAX_AGE_HOURS}h old): $FOUND_COUNT"

if [ "$DRY_RUN" = "true" ]; then
    echo "Directories that would be deleted: $FOUND_COUNT"
    echo "Space that would be freed: ${FREED_MB}MB (${FREED_GB}GB)"
else
    echo "Directories deleted: $DELETED_COUNT"
    echo "Space freed: ${FREED_MB}MB (${FREED_GB}GB)"
fi

# Check remaining storage
if [ -d "$STORAGE_DIR" ]; then
    TOTAL_SIZE=$(du -sh "$STORAGE_DIR" 2>/dev/null | cut -f1)
    echo "Current storage size: $TOTAL_SIZE"
fi

exit 0
