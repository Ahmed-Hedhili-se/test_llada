#!/usr/bin/env bash
# One-shot setup for LLaDA-MoE-7B-A1B-Instruct inference engine.
# Creates a dedicated venv at $SCRIPT_DIR/.venv
# Requires transformers==4.53.2 (5.x removed ROPE_INIT_FUNCTIONS['default'])
#
# Usage:
#   bash setup.sh                        # install deps + download weights
#   bash setup.sh --skip-weights         # install deps only
#   bash setup.sh --weight-dir /path     # custom weight dir (default: ./weights)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEIGHT_DIR="$SCRIPT_DIR/weights"
SKIP_WEIGHTS=0
VENV="${VENV:-$SCRIPT_DIR/.venv}"

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

# Upgrade pip first
"$PIP" install --upgrade pip

# Detect CUDA version (informational)
CUDA_VER=$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' \
        || nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+' \
        || echo "12.1")

echo "  Detected CUDA driver: $CUDA_VER"

# Use cu121 for every CUDA 12.x system.
# Only CUDA 13.x uses cu130.
if [[ "$CUDA_VER" =~ ^13\. ]]; then
    WHL_IDX="cu130"
else
    WHL_IDX="cu121"
fi

echo "  Using PyTorch wheel: $WHL_IDX"

# Check whether a compatible CUDA-enabled PyTorch already exists
TORCH_OK=$("$PY" - <<'PY'
try:
    import torch
    ok = (
        torch.cuda.is_available()
        and torch.__version__.startswith("2.5.1")
    )
    print(ok)
except Exception:
    print(False)
PY
)

if [[ "$TORCH_OK" != "True" ]]; then
    echo "  Installing PyTorch..."

    "$PIP" uninstall -y torch torchvision torchaudio >/dev/null 2>&1 || true

    "$PIP" install \
        torch==2.5.1 \
        torchvision==0.20.1 \
        torchaudio==2.5.1 \
        --index-url https://download.pytorch.org/whl/${WHL_IDX}
else
    echo "  Compatible PyTorch already installed."
fi

echo "  Installing project dependencies..."
"$PIP" install -r "$SCRIPT_DIR/requirements.txt"

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
echo "================================================================"