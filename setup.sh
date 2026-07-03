#!/usr/bin/env bash
# One-shot setup for LLaDA-MoE-7B-A1B-Instruct inference engine.
# Creates a Python venv at ~/venv (reuses if it exists from prior Qwen setup).
#
# Usage:
#   bash setup.sh                        # install deps + download weights
#   bash setup.sh --skip-weights         # install deps only
#   bash setup.sh --weight-dir /path     # custom weight dir (default: ./weights)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEIGHT_DIR="$SCRIPT_DIR/weights"
SKIP_WEIGHTS=0
VENV="${VENV:-$HOME/venv}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-weights) SKIP_WEIGHTS=1; shift ;;
        --weight-dir)   WEIGHT_DIR="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "================================================================"
echo " LLaDA-MoE-7B-A1B-Instruct Inference Engine — Setup"
echo "================================================================"
echo "  Script dir : $SCRIPT_DIR"
echo "  Weight dir : $WEIGHT_DIR"
echo "  Venv       : $VENV"
echo ""

# ── 1. Python venv + deps ─────────────────────────────────────────────────────
echo "[1/3] Setting up Python venv and dependencies..."

if [[ ! -f "$VENV/bin/python" ]]; then
    echo "  Creating venv at $VENV..."
    python3 -m venv "$VENV"
fi
PY="$VENV/bin/python"
PIP="$VENV/bin/pip"

# Detect CUDA version for torch wheel selection
CUDA_VER=$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+' || echo "12.4")
CUDA_MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)
CUDA_MINOR=$(echo "$CUDA_VER" | cut -d. -f2)

# CUDA 13+ (B300 driver 580+): cu130; otherwise fall back
if   [[ "$CUDA_MAJOR" -ge 13 ]]; then WHL_IDX="cu130"
elif [[ "$CUDA_MAJOR" -eq 12 && "$CUDA_MINOR" -ge 6 ]]; then WHL_IDX="cu126"
elif [[ "$CUDA_MAJOR" -eq 12 ]]; then WHL_IDX="cu124"
else WHL_IDX="cu121"
fi

echo "  Detected CUDA $CUDA_VER → torch wheel: $WHL_IDX"

TORCH_OK=$("$PY" -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")
if [[ "$TORCH_OK" != "True" ]]; then
    echo "  Installing PyTorch ($WHL_IDX)..."
    "$PIP" install -q torch --index-url "https://download.pytorch.org/whl/${WHL_IDX}"
else
    echo "  PyTorch already available"
fi

"$PIP" install -q \
    safetensors \
    "transformers>=4.45" \
    accelerate \
    fastapi \
    "uvicorn[standard]" \
    pydantic \
    "jinja2>=3.1.0" \
    huggingface_hub \
    "lm-eval[api]>=0.4.4" \
    aiohttp \
    tqdm

echo "  Done."
echo ""

# ── 2. Verify CUDA ────────────────────────────────────────────────────────────
echo "[2/3] Verifying CUDA..."
"$PY" -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available!'
n = torch.cuda.device_count()
print(f'  {n} GPU(s) available:')
for i in range(n):
    p = torch.cuda.get_device_properties(i)
    print(f'    [{i}] {p.name}  {p.total_memory//1024**3} GB')
"
echo ""

# ── 3. Download weights ───────────────────────────────────────────────────────
if [[ "$SKIP_WEIGHTS" -eq 1 ]]; then
    echo "[3/3] Skipping weight download (--skip-weights)"
else
    echo "[3/3] Downloading inclusionAI/LLaDA-MoE-7B-A1B-Instruct weights (~15 GB)..."
    echo "  Destination: $WEIGHT_DIR"
    "$PY" "$SCRIPT_DIR/download_weights.py" --dest "$WEIGHT_DIR"
fi

echo ""
echo "================================================================"
echo " Setup complete."
echo ""
echo " Next steps:"
echo "   Start server : bash start.sh --weight-dir $WEIGHT_DIR"
echo "   Check server : $PY -m eval.check_server --base-url http://localhost:8000"
echo "   Compare      : $PY compare_models.py --weight-dir $WEIGHT_DIR"
echo "   Correctness  : $PY -m eval.correctness.run_correctness --base-url http://localhost:8000"
echo "   Throughput   : $PY -m eval.throughput.run_throughput --base-url http://localhost:8000"
echo "================================================================"
