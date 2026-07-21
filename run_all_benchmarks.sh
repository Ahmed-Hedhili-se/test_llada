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
echo " LLaDA-MoE Correctness Benchmark: Option A vs HuggingFace"
echo "================================================================"
echo ""
echo "This script compares:"
echo "  1. Option A (fast_dense cached + conservative dynamic experts)"
echo "  2. HuggingFace reference implementation"
echo ""
echo "================================================================"

# Create results directory
mkdir -p results/correctness

for BACKEND in fast_dense hf; do
    echo ""
    echo "================================================================"
    echo " Starting API server with backend: $BACKEND"
    echo "================================================================"

    # Start the server in the background
    $PY -m src.server --backend "$BACKEND" --port 8000 &
    SERVER_PID=$!

    # Wait for the server to be ready
    echo "Waiting for server to become ready on port 8000..."
    for i in {1..60}; do
        if curl -s http://localhost:8000/health > /dev/null 2>&1; then
            echo "Server is ready!"
            break
        fi
        if [[ $i -eq 60 ]]; then
            echo "ERROR: Server failed to start within 120 seconds"
            kill $SERVER_PID 2>/dev/null || true
            exit 1
        fi
        sleep 2
    done

    echo ""
    echo "Running correctness benchmark ($BACKEND)..."
    echo "----------------------------------------------------------------"

    # Run the benchmark
    OUTPUT_FILE="results/correctness/${BACKEND}_summary.json"
    LOG_FILE="results/correctness/${BACKEND}.log"

    if [[ "$BACKEND" == "fast_dense" ]]; then
        # Option A: save as baseline for comparison
        $PY -m eval.correctness.run_correctness \
            --base-url http://localhost:8000 \
            --limit 200 \
            --output "$OUTPUT_FILE" \
            --config-name "Option A (Fast Dense Cached)" \
            2>&1 | tee "$LOG_FILE"
    else
        # HuggingFace: compare to Option A baseline
        BASELINE_FILE="results/correctness/fast_dense_summary.json"
        if [[ -f "$BASELINE_FILE" ]]; then
            $PY -m eval.correctness.run_correctness \
                --base-url http://localhost:8000 \
                --limit 200 \
                --baseline "$BASELINE_FILE" \
                --output "$OUTPUT_FILE" \
                --config-name "HuggingFace Reference" \
                2>&1 | tee "$LOG_FILE"
        else
            echo "WARNING: Baseline not found at $BASELINE_FILE"
            echo "Running without baseline comparison..."
            $PY -m eval.correctness.run_correctness \
                --base-url http://localhost:8000 \
                --limit 200 \
                --output "$OUTPUT_FILE" \
                --config-name "HuggingFace Reference" \
                2>&1 | tee "$LOG_FILE"
        fi
    fi

    echo ""
    echo "Benchmark finished. Results saved to:"
    echo "  Summary: $OUTPUT_FILE"
    echo "  Log:     $LOG_FILE"

    # Kill the server
    echo ""
    echo "Shutting down server (PID $SERVER_PID)..."
    kill $SERVER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
    sleep 3

    # Verify server is down
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "WARNING: Server still responding, forcing kill..."
        kill -9 $SERVER_PID 2>/dev/null || true
        sleep 2
    fi

    echo "================================================================"
done

echo ""
echo "================================================================"
echo " ALL BENCHMARKS COMPLETED"
echo "================================================================"
echo ""

# Print comparison if both results exist
BASELINE="results/correctness/fast_dense_summary.json"
HF_RESULT="results/correctness/hf_summary.json"

if [[ -f "$BASELINE" && -f "$HF_RESULT" ]]; then
    echo "Comparison Summary:"
    echo "----------------------------------------------------------------"

    # Extract accuracies using Python
    $PY -c "
import json, sys

def load_acc(path):
    with open(path) as f:
        d = json.load(f)
    return d.get('accuracy', None)

opt_a = load_acc('$BASELINE')
hf = load_acc('$HF_RESULT')

if opt_a is None or hf is None:
    print('Could not extract accuracy from results.')
    sys.exit(1)

diff = hf - opt_a
print(f'  Option A (fast_dense):  {opt_a*100:.2f}%')
print(f'  HuggingFace (hf):       {hf*100:.2f}%')
print(f'  Difference (HF - OptA): {diff*100:+.2f}%')
print()
if abs(diff) <= 0.01:
    print('  ✅ Option A matches HuggingFace within 1%')
elif abs(diff) <= 0.02:
    print('  ⚠️  Option A within 2% of HuggingFace (acceptable)')
else:
    print('  ❌ Option A differs from HuggingFace by >2%')
    print('     Consider tuning: min_k=5, expert_threshold=0.02')
print()
"
    echo "----------------------------------------------------------------"
else
    echo "Could not find both result files for comparison."
    echo "  Expected: $BASELINE"
    echo "  Expected: $HF_RESULT"
fi

echo ""
echo "Full results saved in: results/correctness/"
echo ""