#!/bin/bash
# CPU Guard - Monitors CPU usage and kills process if it exceeds threshold
# Usage: bash scripts/cpu_guard.sh <max_cpu_percent> <command...>
# Example: bash scripts/cpu_guard.sh 200 python main.py --tag BTS_larger ...
#
# This script:
# 1. Checks CPU usage BEFORE starting (pre-flight)
# 2. Launches the command in background
# 3. Monitors CPU every 2 seconds
# 4. Kills the process tree if CPU% exceeds threshold for 3 consecutive checks
# 5. Reports final status

set -euo pipefail

MAX_CPU=${1:?Usage: cpu_guard.sh <max_cpu_percent> <command...>}
shift
CMD="$@"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() { echo -e "${GREEN}[cpu_guard]${NC} $*"; }
warn() { echo -e "${YELLOW}[cpu_guard]${NC} $*"; }
err() { echo -e "${RED}[cpu_guard]${NC} $*"; }

# Get total CPU usage across all cores (sum of all processes)
get_total_cpu() {
    # top -bn1 gives snapshot, sum all %CPU values
    top -bn1 | grep "Cpu(s)" | awk '{print $2 + $4}' | cut -d'.' -f1
}

# Get CPU usage of specific process tree
get_process_cpu() {
    local pid=$1
    if [ -d "/proc/$pid" ]; then
        # Sum CPU of process and all children
        ps -p "$pid" -o %cpu= 2>/dev/null | tr -d ' ' || echo "0"
    else
        echo "0"
    fi
}

# Pre-flight check
log "=== CPU Guard Pre-flight Check ==="
CURRENT_CPU=$(get_total_cpu)
log "Current system CPU usage: ${CURRENT_CPU}%"
log "Max allowed: ${MAX_CPU}%"

if [ "$CURRENT_CPU" -gt "$MAX_CPU" ]; then
    err "ABORT: Current CPU (${CURRENT_CPU}%) already exceeds limit (${MAX_CPU}%)!"
    err "Please wait for other processes to finish or increase the limit."
    exit 1
fi
log "Pre-flight check PASSED"

# Launch command
log "Launching: $CMD"
eval "$CMD" &
CMD_PID=$!
log "Process started with PID: $CMD_PID"

# Monitor loop
VIOLATION_COUNT=0
MAX_VIOLATIONS=3  # Kill after 3 consecutive violations
CHECK_INTERVAL=2  # seconds

cleanup() {
    if kill -0 "$CMD_PID" 2>/dev/null; then
        warn "Killing process tree..."
        kill -TERM -- -"$CMD_PID" 2>/dev/null || true
        pkill -P "$CMD_PID" 2>/dev/null || true
        kill "$CMD_PID" 2>/dev/null || true
        sleep 1
        kill -9 "$CMD_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

while kill -0 "$CMD_PID" 2>/dev/null; do
    sleep "$CHECK_INTERVAL"

    # Get process-specific CPU (not total system)
    PROC_CPU=$(ps -p "$CMD_PID" -o %cpu= 2>/dev/null | tr -d ' ' | cut -d'.' -f1 || echo "0")
    TOTAL_CPU=$(get_total_cpu)

    if [ "${PROC_CPU:-0}" -gt "$MAX_CPU" ]; then
        VIOLATION_COUNT=$((VIOLATION_COUNT + 1))
        warn "CPU violation ${VIOLATION_COUNT}/${MAX_VIOLATIONS}: Process CPU=${PROC_CPU}% (limit=${MAX_CPU}%)"

        if [ "$VIOLATION_COUNT" -ge "$MAX_VIOLATIONS" ]; then
            err "KILLED: Process exceeded CPU limit ${MAX_VIOLATIONS} times consecutively"
            err "Process CPU: ${PROC_CPU}%, System CPU: ${TOTAL_CPU}%"
            exit 137
        fi
    else
        VIOLATION_COUNT=0  # Reset on good check
    fi

    # Log periodically (every 30 seconds)
    if [ $((SECONDS % 30)) -lt "$CHECK_INTERVAL" ]; then
        log "Status: Process CPU=${PROC_CPU}%, System CPU=${TOTAL_CPU}%"
    fi
done

# Wait for process to finish and get exit code
wait "$CMD_PID" 2>/dev/null
EXIT_CODE=$?

if [ "$EXIT_CODE" -eq 0 ]; then
    log "Process completed successfully (exit code: 0)"
else
    warn "Process exited with code: $EXIT_CODE"
fi

exit "$EXIT_CODE"
