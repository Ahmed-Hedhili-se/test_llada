#!/usr/bin/env bash
# One-shot setup for LLaDA-MoE-7B-A1B-Instruct inference engine.
# Creates a dedicated venv at the repo root (.venv)
# Requires transformers==4.53.2 (5.x removed ROPE_INIT_FUNCTIONS['default'])
#
# Usage:
#   bash setup.sh                        # install deps + download weights
#   bash setup.sh --skip-weights         # install deps only
#   bash setup.sh --weight-dir /path     # custom weight dir (default: ./weights)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# The scripts live in scripts/, so the repo root is one level up.
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WEIGHT_DIR="$REPO_ROOT/weights"
SKIP_WEIGHTS=0
VENV="${VENV:-$REPO_ROOT/.venv}"

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
echo "  Repo root  : $REPO_ROOT"
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

# Wheel line for the pinned torch 2.5.1.
#
# NOT simply "newest index the driver supports": torch 2.5.1 predates CUDA 13
# and has *no* cu130 wheels, so a CUDA 13 driver (H100 boxes ship them) sent us
# to an index where the pinned version does not exist and the install died.
# CUDA drivers are backward compatible with older runtimes, so a 13.x driver
# runs the cu124 build fine -- that is the newest line 2.5.1 actually ships.
#
# If the torch pin is ever raised past 2.9, revisit: cu130 becomes real then.
if [[ "$CUDA_VER" =~ ^13\. ]]; then
    WHL_IDX="cu124"
    echo "  CUDA 13 driver: using cu124 (torch 2.5.1 has no cu130 build)"
elif [[ "$CUDA_VER" =~ ^12\.[4-9] ]]; then
    WHL_IDX="cu124"
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
"$PIP" install -r "$REPO_ROOT/requirements.txt"

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
    "$PY" "$REPO_ROOT/tools/download_weights.py" --dest "$WEIGHT_DIR"
fi

echo ""
echo "================================================================"
echo " Setup complete."
echo "================================================================"