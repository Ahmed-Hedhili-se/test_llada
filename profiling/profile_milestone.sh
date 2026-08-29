#!/usr/bin/env bash
# Profile one historical milestone of the DMInfr engine under a fixed workload.
#
#   ./profiling/profile_milestone.sh <label> [--worktree <commit>] [--no-nsys]
#
# The ONLY thing that may differ between runs is the code version. Everything
# else -- model, input shape, batch size, generation length, step count, block
# length, decoding config, GPU, environment, tuned kernel configs -- is pinned
# below and recorded at run time.
#
# Two ways to select the code version:
#
#   --worktree <commit>   RECOMMENDED. Materialises the commit in a private git
#                         worktree and profiles that, leaving your checkout (and
#                         this script) untouched.
#   (no flag)             Profiles the current working tree as it stands.
#
# Why worktree is recommended: `git checkout <old-commit>` DELETES profiling/,
# because that directory does not exist in history. The naive
# checkout-then-run workflow removes the very script it is about to run.
set -uo pipefail

# ─── FIXED WORKLOAD ───────────────────────────────────────────────────────────
# Chosen as the intersection of flags available at EVERY selected milestone.
# --mode exists only from bff9b41 (2026-07-29) and --batch-size only from
# b954121 (2026-08-16), so batch size stays at its default of 1 and nothing
# here may change without re-checking availability across the whole range.
GEN_LENGTH=128
STEPS=128
BLOCK_LENGTH=32
NUM_RUNS=3
NUM_WARMUP=1
MODE=both          # runs the frozen baseline AND the optimized path, so the
                   # baseline arm acts as a per-milestone control (see README)
# ──────────────────────────────────────────────────────────────────────────────

die() { echo "error: $*" >&2; exit 1; }

LABEL=""; WORKTREE_COMMIT=""; USE_NSYS=1
while [[ $# -gt 0 ]]; do
    case $1 in
        --worktree) WORKTREE_COMMIT="${2:-}"; shift 2 ;;
        --no-nsys)  USE_NSYS=0; shift ;;
        -h|--help)  sed -n '2,20p' "$0"; exit 0 ;;
        -*)         die "unknown flag: $1" ;;
        *)          LABEL="$1"; shift ;;
    esac
done
[[ -n "$LABEL" ]] || die "usage: $0 <label> [--worktree <commit>] [--no-nsys]"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAIN_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUT_DIR="$MAIN_ROOT/profiling/results/$LABEL"
mkdir -p "$OUT_DIR"

# ─── select the tree to profile ───────────────────────────────────────────────
if [[ -n "$WORKTREE_COMMIT" ]]; then
    WT="$MAIN_ROOT/.profiling_worktrees/$LABEL"
    if [[ -d "$WT" ]]; then
        echo "reusing existing worktree $WT"
    else
        echo "creating worktree for $WORKTREE_COMMIT ..."
        git -C "$MAIN_ROOT" worktree add --detach "$WT" "$WORKTREE_COMMIT" >/dev/null \
            || die "could not create worktree for $WORKTREE_COMMIT"
    fi
    RUN_ROOT="$WT"
else
    RUN_ROOT="$MAIN_ROOT"
fi

COMMIT=$(git -C "$RUN_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)
COMMIT_DATE=$(git -C "$RUN_ROOT" log -1 --format=%ad --date=short 2>/dev/null || echo unknown)
COMMIT_MSG=$(git -C "$RUN_ROOT" log -1 --format=%s 2>/dev/null || echo unknown)

# ─── layout detection ─────────────────────────────────────────────────────────
# The tree was restructured at 1fca44f (eval/ + model_update/ became
# benchmarks/ + dminfr/engine/). Milestones straddle that boundary, so resolve
# the entry point instead of assuming one.
if   [[ -f "$RUN_ROOT/benchmarks/check_time_inference.py" ]]; then
    BENCH_MODULE="benchmarks.check_time_inference"; LAYOUT="dminfr (post-1fca44f)"
elif [[ -f "$RUN_ROOT/eval/check_time_inference.py" ]]; then
    BENCH_MODULE="eval.check_time_inference";       LAYOUT="legacy (eval/ + model_update/)"
else
    die "no check_time_inference.py in $RUN_ROOT -- this commit predates the benchmark harness and cannot be profiled with this workload"
fi

# ─── pinned inputs ────────────────────────────────────────────────────────────
WEIGHT_DIR="${WEIGHT_DIR:-$MAIN_ROOT/weights}"
[[ -d "$WEIGHT_DIR" ]] || die "weights not found at $WEIGHT_DIR (override with WEIGHT_DIR=...)"

VENV="${VENV:-$MAIN_ROOT/.venv}"
PY="$VENV/bin/python"
[[ -x "$PY" ]] || PY="$(command -v python3 || true)"
[[ -x "$PY" ]] || die "no python interpreter found"

# Tuned-kernel-config normalisation. THIS MATTERS. HEAD prefers
# moe_tune_config.device_name=<GPU>.json; every earlier milestone reads the
# unkeyed moe_tune_config.json. With only one of them present, the older
# milestones silently fall back to hardcoded default tile shapes while HEAD
# uses tuned ones -- which would attribute an autotuning difference to whatever
# else changed between the commits.
GPU_TAG=$("$PY" -c "import torch;print(torch.cuda.get_device_name(0).replace(' ','_').replace('/','_'))" 2>/dev/null || echo unknown)
KEYED="$MAIN_ROOT/moe_tune_config.device_name=${GPU_TAG}.json"
PLAIN="$MAIN_ROOT/moe_tune_config.json"
CONFIG_NOTE="none found -- every milestone uses hardcoded fallback configs (still internally comparable)"
if [[ -f "$KEYED" ]]; then
    cp -f "$KEYED" "$PLAIN"
    CONFIG_NOTE="device-keyed config for $GPU_TAG, mirrored to the unkeyed name"
elif [[ -f "$PLAIN" ]]; then
    CONFIG_NOTE="unkeyed moe_tune_config.json only"
fi
if [[ "$RUN_ROOT" != "$MAIN_ROOT" ]]; then
    for f in "$PLAIN" "$KEYED"; do
        [[ -f "$f" ]] && cp -f "$f" "$RUN_ROOT/$(basename "$f")"
    done
fi

# ─── environment record ───────────────────────────────────────────────────────
META="$OUT_DIR/run_info.txt"
PYINFO=$("$PY" -c "
import torch
out=[f'torch            : {torch.__version__}', f'cuda             : {torch.version.cuda}']
try:
    import triton; out.append(f'triton           : {triton.__version__}')
except Exception: out.append('triton           : missing')
for i in range(torch.cuda.device_count()):
    p=torch.cuda.get_device_properties(i)
    out.append(f'  gpu[{i}]        : {p.name} sm_{p.major}{p.minor} {p.total_memory/2**30:.0f}GiB {p.multi_processor_count}SM')
print(chr(10).join(out))
" 2>/dev/null || echo "torch            : unavailable")

{
  echo "label            : $LABEL"
  echo "commit           : $COMMIT"
  echo "commit_date      : $COMMIT_DATE"
  echo "commit_subject   : $COMMIT_MSG"
  echo "layout           : $LAYOUT"
  echo "bench_module     : $BENCH_MODULE"
  echo "run_root         : $RUN_ROOT"
  echo "profiled_at      : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "hostname         : $(hostname)"
  echo "gpu              : $GPU_TAG"
  echo "tuned_config     : $CONFIG_NOTE"
  echo
  echo "workload         : gen_length=$GEN_LENGTH steps=$STEPS block_length=$BLOCK_LENGTH"
  echo "                   num_runs=$NUM_RUNS num_warmup=$NUM_WARMUP mode=$MODE batch_size=1(default)"
  echo
  echo "$PYINFO"
  echo
  nvidia-smi --query-gpu=driver_version,memory.total --format=csv,noheader 2>/dev/null \
      | sed 's/^/driver, vram     : /'
} | tee "$META"

# ─── run ──────────────────────────────────────────────────────────────────────
ARGS=( -m "$BENCH_MODULE"
       --weight-dir "$WEIGHT_DIR"
       --gen-length "$GEN_LENGTH" --steps "$STEPS" --block-length "$BLOCK_LENGTH"
       --num-runs "$NUM_RUNS" --num-warmup "$NUM_WARMUP" --mode "$MODE" )

NSYS_BIN="$(command -v nsys || true)"
if [[ -z "$NSYS_BIN" && -x /opt/nvidia/nsight-systems/bin/nsys ]]; then
    NSYS_BIN=/opt/nvidia/nsight-systems/bin/nsys
fi

echo
echo "=============================================================="
echo " profiling '$LABEL'   ($COMMIT, $COMMIT_DATE)"
echo "=============================================================="
cd "$RUN_ROOT" || die "cannot cd to $RUN_ROOT"

START=$(date +%s)
if [[ $USE_NSYS -eq 1 && -n "$NSYS_BIN" ]]; then
    echo "nsys: $NSYS_BIN"
    "$NSYS_BIN" profile \
        --trace=cuda,nvtx,osrt \
        --sample=cpu \
        --cudabacktrace=none \
        --force-overwrite=true \
        --output "$OUT_DIR/${LABEL}" \
        "$PY" "${ARGS[@]}" 2>&1 | tee "$OUT_DIR/stdout.txt"
    RC=${PIPESTATUS[0]}
else
    if [[ $USE_NSYS -eq 1 ]]; then
        echo "nsys not found -- running without a trace (timings are still recorded)"
    fi
    "$PY" "${ARGS[@]}" 2>&1 | tee "$OUT_DIR/stdout.txt"
    RC=${PIPESTATUS[0]}
fi
END=$(date +%s)

{
  echo
  echo "exit_code        : $RC"
  echo "wall_seconds     : $((END - START))"
} | tee -a "$META"

# Export summaries alongside the trace so the report does not need the GUI.
if [[ -n "$NSYS_BIN" && -f "$OUT_DIR/${LABEL}.nsys-rep" ]]; then
    echo "exporting kernel / API / memory summaries ..."
    for r in cuda_gpu_kern_sum cuda_api_sum cuda_gpu_mem_time_sum cuda_gpu_mem_size_sum; do
        if "$NSYS_BIN" stats --report "$r" --format csv \
              --output "$OUT_DIR/${LABEL}_${r}" \
              "$OUT_DIR/${LABEL}.nsys-rep" >/dev/null 2>&1; then
            echo "  wrote ${LABEL}_${r}.csv"
        fi
    done
fi

echo
echo "results -> $OUT_DIR"
ls -1 "$OUT_DIR"
exit $RC
