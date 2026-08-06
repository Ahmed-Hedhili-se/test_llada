#!/usr/bin/env bash
# ============================================================
#  profile_kernel_occupancy.sh
#
#  Packages the Nsight Compute (ncu) invocation used to deeply profile
#  fused_moe_kernel (Triton grouped-GEMM) and flash_fwd_kernel (PyTorch's
#  bundled flash-attention) -- memory bandwidth, occupancy, and warp stall
#  reasons -- into one repeatable command, at a chosen --batch-size.
#
#  This is NOT a pass/fail test. ncu requires sudo (GPU performance
#  counters are root-restricted by the NVIDIA driver) and the standalone
#  Nsight Compute package, neither of which is something to assume is
#  present in a CI environment -- so this stays a manual diagnostic script,
#  not something eval/test_*.py's automated-check pattern applies to.
#  Thresholds for "good" occupancy/bandwidth also depend on batch size,
#  hardware, and Triton's autotuned config, so a fixed pass/fail bar here
#  would be more likely to flag false failures than catch real ones.
#
#  See README.md's "Kernel-Level Validation" section under Server-Side
#  Request Batching for the numbers this produced on a single A6000 at
#  batch sizes 1/32/64, and the conclusions drawn from them.
#
#  Prerequisites (one-time setup, see README if ncu is missing):
#    - NVIDIA Nsight Compute (ncu) installed and on PATH
#    - sudo access (GPU perf counters need root; the NVIDIA driver blocks
#      non-root reads with ERR_NVGPUCTRPERM otherwise)
#
#  Usage:
#    bash eval/profile_kernel_occupancy.sh [--batch-size 1] \
#         [--gen-length 16] [--steps 16] [--block-length 16] \
#         [--launch-count 16] [--kernel-name "fused_moe_kernel|flash_fwd_kernel"] \
#         [--master-port 29501] [--out-dir ~/test_llada/ncu_reports] \
#         [--weight-dir ./weights]
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

# ── defaults ────────────────────────────────────────────────
WEIGHT_DIR="${ROOT_DIR}/weights"
BATCH_SIZE=1
GEN_LENGTH=16
STEPS=16
BLOCK_LENGTH=16
LAUNCH_COUNT=16
KERNEL_NAME="fused_moe_kernel|flash_fwd_kernel"
MASTER_PORT=29501
OUT_DIR="${ROOT_DIR}/ncu_reports"

while [[ $# -gt 0 ]]; do
    case $1 in
        --weight-dir)    WEIGHT_DIR="$2";    shift 2 ;;
        --batch-size)    BATCH_SIZE="$2";    shift 2 ;;
        --gen-length)    GEN_LENGTH="$2";    shift 2 ;;
        --steps)         STEPS="$2";         shift 2 ;;
        --block-length)  BLOCK_LENGTH="$2";  shift 2 ;;
        --launch-count)  LAUNCH_COUNT="$2";  shift 2 ;;
        --kernel-name)   KERNEL_NAME="$2";   shift 2 ;;
        --master-port)   MASTER_PORT="$2";   shift 2 ;;
        --out-dir)       OUT_DIR="$2";       shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ── locate ncu ──────────────────────────────────────────────
NCU_BIN="$(command -v ncu || true)"
if [[ -z "$NCU_BIN" ]]; then
    for candidate in /usr/local/cuda*/bin/ncu /opt/nvidia/nsight-compute/*/ncu; do
        if [[ -x "$candidate" ]]; then NCU_BIN="$candidate"; break; fi
    done
fi
if [[ -z "$NCU_BIN" ]]; then
    echo "ncu not found. Install it (Ubuntu example, match your CUDA/driver version):"
    echo "  wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb"
    echo "  sudo dpkg -i cuda-keyring_1.1-1_all.deb && sudo apt-get update"
    echo "  apt-cache search nsight-compute   # pick the version matching your CUDA install"
    echo "  sudo apt-get install -y <package-name>"
    exit 1
fi
echo "Using ncu: $NCU_BIN"

# ── locate venv python ──────────────────────────────────────
PY="${ROOT_DIR}/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
    echo "Expected venv python at $PY -- adjust ROOT_DIR/venv layout or edit this script."
    exit 1
fi

mkdir -p "$OUT_DIR"
STAMP=$(date +%s)
REP_BASE="${OUT_DIR}/kernel_profile_b${BATCH_SIZE}_${STAMP}"

echo ""
echo "======================================================================"
echo "  Kernel occupancy/bandwidth/stall profile -- batch_size=${BATCH_SIZE}"
echo "======================================================================"
echo "  Kernel filter : $KERNEL_NAME"
echo "  Launch count  : $LAUNCH_COUNT"
echo "  Gen/Steps/Blk : ${GEN_LENGTH}/${STEPS}/${BLOCK_LENGTH}"
echo "  Report        : ${REP_BASE}.ncu-rep / .txt"
echo "======================================================================"
echo ""
echo "This runs the full ncu metric set (--set full) -- expect a few minutes,"
echo "not seconds. Requires sudo for GPU performance counter access."
echo ""

MASTER_ADDR=localhost MASTER_PORT="$MASTER_PORT" RANK=0 WORLD_SIZE=1 TOKENIZERS_PARALLELISM=false \
    sudo -E env "PATH=$PATH" "$NCU_BIN" \
    --target-processes all \
    --kernel-name "regex:${KERNEL_NAME}" \
    --launch-count "$LAUNCH_COUNT" \
    --set full \
    -f -o "$REP_BASE" \
    "$PY" "${SCRIPT_DIR}/check_time_inference.py" \
    --mode optimized --weight-dir "$WEIGHT_DIR" \
    --batch-size "$BATCH_SIZE" --gen-length "$GEN_LENGTH" --steps "$STEPS" --block-length "$BLOCK_LENGTH" \
    --num-warmup 0 --num-runs 1

sudo -E env "PATH=$PATH" "$NCU_BIN" --import "${REP_BASE}.ncu-rep" --page details > "${REP_BASE}.txt"

echo ""
echo "======================================================================"
echo "  Summary (Speed of Light / Occupancy / Scheduler, per kernel launch)"
echo "======================================================================"
printf "%-28s %10s %8s %8s %8s %10s\n" "Kernel" "Duration" "MemBW%" "Compute%" "Occup%" "NoElig%"
printf "%-28s %10s %8s %8s %8s %10s\n" "------" "--------" "------" "--------" "------" "-------"

awk '
  /^  [a-zA-Z]/ && !/Section:/ { name = $0; sub(/ \(.*/, "", name); if (length(name) > 26) name = substr(name, 1, 26); n=1 }
  /Duration / && n==1 { dur = $(NF-1) " " $NF }
  /Memory Throughput  *%/ && n==1 { membw = $(NF) }
  /Compute \(SM\) Throughput/ && n==1 { compute = $(NF) }
  /No Eligible  *%/ && n==1 { noelig = $(NF) }
  /Achieved Occupancy/ && n==1 {
    occ = $(NF)
    printf "%-28s %10s %8s %8s %8s %10s\n", name, dur, membw, compute, occ, noelig
    n=0
  }
' "${REP_BASE}.txt"

echo ""
echo "Full report: ${REP_BASE}.txt  (binary: ${REP_BASE}.ncu-rep)"
echo "To pull the full report to a local machine:"
echo "  scp <user>@<host>:${REP_BASE}.txt ."
echo ""
