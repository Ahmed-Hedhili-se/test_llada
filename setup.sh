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