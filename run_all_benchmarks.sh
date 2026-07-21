#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${VENV:-$SCRIPT_DIR/.venv}"
PY="$VENV/bin/python"

if [[ ! -f "$PY" ]]; then
    echo "Virtual environment not found at $VENV"
    exit 1
fi

echo "================================================================"
echo " LLaDA-MoE Correctness Benchmark Automation"
echo "================================================================"

for BACKEND in ours ours_kv hf; do
    echo "Starting API server with backend: $BACKEND ..."
    
    # Start the server in the background
    $PY -m src.server --backend "$BACKEND" --port 8000 &
    SERVER_PID=$!

    # Wait for the server to be ready
    echo "Waiting for server to become ready on port 8000..."
    for i in {1..30}; do
        if curl -s http://localhost:8000/health > /dev/null; then
            break
        fi
        sleep 2
    done

    echo "Server is ready! Running correctness benchmark..."
    
    # Run the benchmark and save output to a log file
    LOG_FILE="correctness_${BACKEND}.log"
    $PY -m eval.correctness.run_correctness --base-url http://localhost:8000 | tee "$LOG_FILE"
    
    echo "Correctness benchmark finished. Log saved to $LOG_FILE"
    
    # Kill the server
    echo "Shutting down server (PID $SERVER_PID)..."
    kill $SERVER_PID
    wait $SERVER_PID 2>/dev/null || true
    sleep 2
    echo "================================================================"
done

echo "All benchmarks completed!"
