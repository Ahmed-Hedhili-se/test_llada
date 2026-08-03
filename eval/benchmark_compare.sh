#!/usr/bin/env bash
# ============================================================
#  benchmark_compare.sh
#
#  Compares inference time of:
#    1) Baseline  — src/ model, single GPU, no TP
#    2) Optimized — model_update/ model, 2 GPUs, Tensor Parallelism
#
#  Usage:
#    bash eval/benchmark_compare.sh [--weight-dir ./weights] \
#         [--gen-length 64] [--steps 64] [--block-length 32] \
#         [--num-warmup 1] [--num-runs 3]
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
VENV="${ROOT_DIR}/.venv"
PY="${VENV}/bin/python"
TORCHRUN="${VENV}/bin/torchrun"

# ── defaults ────────────────────────────────────────────────
WEIGHT_DIR="${ROOT_DIR}/weights"
GEN_LENGTH=64
STEPS=64
BLOCK_LENGTH=32
NUM_WARMUP=1
NUM_RUNS=3
NPROC=2

while [[ $# -gt 0 ]]; do
    case $1 in
        --weight-dir)   WEIGHT_DIR="$2";  shift 2 ;;
        --gen-length)   GEN_LENGTH="$2";  shift 2 ;;
        --steps)        STEPS="$2";       shift 2 ;;
        --block-length) BLOCK_LENGTH="$2";shift 2 ;;
        --num-warmup)   NUM_WARMUP="$2";  shift 2 ;;
        --num-runs)     NUM_RUNS="$2";    shift 2 ;;
        --nproc)        NPROC="$2";       shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

BENCH="${SCRIPT_DIR}/check_time_inference.py"
COMMON_ARGS=(
    --weight-dir "$WEIGHT_DIR"
    --gen-length "$GEN_LENGTH"
    --steps      "$STEPS"
    --block-length "$BLOCK_LENGTH"
    --num-warmup "$NUM_WARMUP"
    --num-runs   "$NUM_RUNS"
)

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Inference Time Comparison: Baseline vs Optimized+TP        ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Weights     : $WEIGHT_DIR"
echo "║  Gen Length  : $GEN_LENGTH tokens"
echo "║  Steps       : $STEPS"
echo "║  Block Length: $BLOCK_LENGTH"
echo "║  Warmup / Runs: $NUM_WARMUP / $NUM_RUNS"
echo "║  TP Size     : $NPROC GPUs"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ── STEP 1: Baseline (single GPU, no TP) ────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  [1/2] BASELINE  (src/, single GPU cuda:0, no TP)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

BASELINE_OUT=$("$PY" "$BENCH" "${COMMON_ARGS[@]}" --mode baseline --device cuda:0 2>&1)
echo "$BASELINE_OUT"

# Extract: "Mean: X.XXs | ..."  →  X.XX
BL_MEAN=$(echo "$BASELINE_OUT" | grep -oP 'Mean:\s+\K[0-9]+\.[0-9]+' | head -1)
BL_TPS=$(echo "$BASELINE_OUT"  | grep -oP '[0-9]+\.[0-9]+\s+tok/s' | grep -oP '[0-9]+\.[0-9]+' | head -1)

echo ""

# ── STEP 2: Optimized + TP (multi-GPU, torchrun) ─────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  [2/2] OPTIMIZED  (model_update/, ${NPROC}x GPU, Tensor Parallelism)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

OPT_OUT=$("$TORCHRUN" --nproc_per_node="$NPROC" "$BENCH" "${COMMON_ARGS[@]}" --mode optimized 2>&1)
echo "$OPT_OUT"

OPT_MEAN=$(echo "$OPT_OUT" | grep -oP 'Mean:\s+\K[0-9]+\.[0-9]+' | head -1)
OPT_TPS=$(echo "$OPT_OUT"  | grep -oP '[0-9]+\.[0-9]+\s+tok/s' | grep -oP '[0-9]+\.[0-9]+' | head -1)
OPT_OUTPUT_TEXT=$(echo "$OPT_OUT" | grep -oP 'Optimized output:.*' | head -1)

echo ""

# ── SUMMARY TABLE ────────────────────────────────────────────
if [[ -n "$BL_MEAN" && -n "$OPT_MEAN" ]]; then
    SPEEDUP=$(python3 -c "print(f'{float(\"$BL_MEAN\")/float(\"$OPT_MEAN\"):.2f}')" 2>/dev/null || echo "?")
    echo "╔══════════════════════════════════════════════════════════════════════════╗"
    echo "║                         FINAL COMPARISON                               ║"
    echo "╠══════════════════════════════╦═══════════╦════════════╦════════════════╣"
    echo "║  Configuration               ║  Time (s) ║  Tok/s     ║  Speedup       ║"
    echo "╠══════════════════════════════╬═══════════╬════════════╬════════════════╣"
    printf "║  %-28s  ║  %7s  ║  %8s  ║  %12s  ║\n" \
        "Baseline (src/, 1 GPU)"    "${BL_MEAN}s"  "${BL_TPS}"  "1.00x"
    printf "║  %-28s  ║  %7s  ║  %8s  ║  %12s  ║\n" \
        "Optimized+TP (${NPROC}xGPU)"  "${OPT_MEAN}s" "${OPT_TPS}" "${SPEEDUP}x"
    echo "╚══════════════════════════════╩═══════════╩════════════╩════════════════╝"
    echo ""
    if python3 -c "exit(0 if float('$SPEEDUP') > 1 else 1)" 2>/dev/null; then
        echo "  ✅ Optimized+TP is ${SPEEDUP}x faster than the single-GPU baseline!"
    else
        echo "  ⚠️  Optimized+TP is slower than baseline — check GPU utilization."
    fi
else
    echo "⚠️  Could not extract timing numbers for comparison."
    echo "   BL_MEAN='$BL_MEAN'  OPT_MEAN='$OPT_MEAN'"
fi
echo ""
