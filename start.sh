#!/usr/bin/env bash
# Start the LLaDA-MoE inference server.
#
# Usage:
#   bash start.sh
#   bash start.sh --weight-dir /path/to/weights --port 8000
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEIGHT_DIR="$SCRIPT_DIR/weights"
PORT=8000
HOST="0.0.0.0"
DEVICE="cuda:0"
VENV="${VENV:-$SCRIPT_DIR/.venv}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --weight-dir) WEIGHT_DIR="$2"; shift 2 ;;
        --port)       PORT="$2";       shift 2 ;;
        --host)       HOST="$2";       shift 2 ;;
        --device)     DEVICE="$2";     shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

PY="$VENV/bin/python"
if [[ ! -f "$PY" ]]; then
    echo "Venv not found at $VENV — run: bash setup.sh first"
    exit 1
fi

echo "================================================================"
echo " LLaDA-MoE-7B-A1B-Instruct Inference Server"
echo "================================================================"
echo "  Weights : $WEIGHT_DIR"
echo "  Listen  : http://$HOST:$PORT"
echo "  Device  : $DEVICE"
echo ""

cd "$SCRIPT_DIR"
exec "$PY" -m src.server \
    --weight-dir "$WEIGHT_DIR" \
    --port "$PORT" \
    --host "$HOST" \
    --device "$DEVICE"
