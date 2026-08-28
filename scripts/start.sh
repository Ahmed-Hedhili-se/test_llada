#!/usr/bin/env bash
# Start the LLaDA-MoE inference server.
#
# Usage:
#   bash start.sh
#   bash start.sh --weight-dir /path/to/weights --port 8000
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# The scripts live in scripts/, so the repo root is one level up.
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WEIGHT_DIR="$REPO_ROOT/weights"
PORT=8000
HOST="0.0.0.0"
DEVICE="cuda:0"
VENV="${VENV:-$REPO_ROOT/.venv}"

BACKEND="ours"
TP_SIZE=1
QUANT_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --weight-dir) WEIGHT_DIR="$2"; shift 2 ;;
        --port)       PORT="$2";       shift 2 ;;
        --host)       HOST="$2";       shift 2 ;;
        --device)     DEVICE="$2";     shift 2 ;;
        --backend)    BACKEND="$2";    shift 2 ;;
        --tp-size)    TP_SIZE="$2";    shift 2 ;;
        # Quantization is served by the optional LLaDA_Quant toolkit; these
        # pass straight through to dminfr.serving.server, which imports it lazily.
        --quantize)          QUANT_ARGS+=(--quantize "$2");          shift 2 ;;
        --quant-group-size)  QUANT_ARGS+=(--quant-group-size "$2");  shift 2 ;;
        --quant-mode)        QUANT_ARGS+=(--quant-mode "$2");        shift 2 ;;
        --no-fused-quant)    QUANT_ARGS+=(--no-fused-quant);         shift 1 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ ${#QUANT_ARGS[@]} -gt 0 && "$BACKEND" != "fast_dense" ]]; then
    echo "Quantization needs --backend fast_dense (got '$BACKEND')."
    echo "Only that backend builds the fused expert blocks the quantizer targets."
    exit 1
fi

if [[ ${#QUANT_ARGS[@]} -gt 0 && "$TP_SIZE" -gt 1 ]]; then
    # The torchrun branch below does not forward QUANT_ARGS, and under TP each
    # rank holds only NE//tp_size experts, so quantization is rank-specific.
    # Erroring beats launching an unquantized server that looks quantized.
    echo "Quantization with --tp-size > 1 is not supported."
    echo "Each rank holds a different expert shard, so the quantized state is"
    echo "rank-specific. Run single-GPU, or use DP replicas."
    exit 1
fi

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
echo "  TP Size : $TP_SIZE"
echo ""

cd "$REPO_ROOT"

if [ "$TP_SIZE" -gt 1 ]; then
    exec "$VENV/bin/torchrun" --nproc_per_node="$TP_SIZE" -m dminfr.serving.server \
        --weight-dir "$WEIGHT_DIR" \
        --port "$PORT" \
        --host "$HOST" \
        --backend "$BACKEND"
else
    exec "$PY" -m dminfr.serving.server \
        --weight-dir "$WEIGHT_DIR" \
        --port "$PORT" \
        --host "$HOST" \
        --device "$DEVICE" \
        --backend "$BACKEND" \
        "${QUANT_ARGS[@]}"
fi
