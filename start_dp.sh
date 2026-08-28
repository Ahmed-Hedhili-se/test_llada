#!/usr/bin/env bash
# Data-parallel deployment: one replica per GPU, one endpoint in front.
#
# This is the throughput topology. Tensor parallelism is NOT, because
# src/server.py disables request batching whenever tp_size > 1 -- a TP=8 server
# serialises every request through one lock and delivers roughly an eighth of
# what the hardware can do. The model is ~14 GiB, so on 80 GiB cards there is
# no capacity reason to split it either.
#
# Usage:
#   bash start_dp.sh                                  # one replica per visible GPU
#   bash start_dp.sh --gpus 8 --quantize int8
#   bash start_dp.sh --gpus 4 --port 8000 --batch-max-size 128
#
# Logs land in ./dp_logs/. Ctrl-C stops the router and every replica.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEIGHT_DIR="$SCRIPT_DIR/weights"
VENV="${VENV:-$SCRIPT_DIR/.venv}"
PORT=8000
REPLICA_PORT_BASE=8100
GPUS=""
REPLICAS=""
DRY_RUN=0
BACKEND="fast_dense"
BATCH_MAX_SIZE="${BATCH_MAX_SIZE:-}"
LOG_DIR="$SCRIPT_DIR/dp_logs"
QUANT_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --weight-dir)     WEIGHT_DIR="$2";        shift 2 ;;
        --port)           PORT="$2";              shift 2 ;;
        --replica-port)   REPLICA_PORT_BASE="$2"; shift 2 ;;
        --gpus)           GPUS="$2";              shift 2 ;;
        --replicas)       REPLICAS="$2";          shift 2 ;;
        --dry-run)        DRY_RUN=1;              shift 1 ;;
        --backend)        BACKEND="$2";           shift 2 ;;
        --batch-max-size) BATCH_MAX_SIZE="$2";    shift 2 ;;
        --log-dir)        LOG_DIR="$2";           shift 2 ;;
        --quantize)          QUANT_ARGS+=(--quantize "$2");          shift 2 ;;
        --quant-group-size)  QUANT_ARGS+=(--quant-group-size "$2");  shift 2 ;;
        --quant-mode)        QUANT_ARGS+=(--quant-mode "$2");        shift 2 ;;
        --no-fused-quant)    QUANT_ARGS+=(--no-fused-quant);         shift 1 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Every replica runs `-m src.server`, which needs the repo root on Python's
# module search path. `-m` resolves that from the CWD (absent PYTHONPATH), so
# invoking this script by path from somewhere else -- `bash /path/to/
# start_dp.sh`, or a caller like a monitoring script that lives outside the
# repo -- silently launches replicas that die with "No module named 'src'"
# the instant they start. WEIGHT_DIR/LOG_DIR are already absolute (derived
# from SCRIPT_DIR above) so only the replica launch's CWD was ever missing.
cd "$SCRIPT_DIR"

PY="$VENV/bin/python"
if [[ ! -f "$PY" ]]; then
    PY="$(command -v python3 || true)"
    [[ -n "$PY" ]] || { echo "No python found. Run: bash setup.sh"; exit 1; }
    echo "Note: venv not at $VENV, falling back to $PY"
fi

# Default to every visible GPU. Deliberately queried rather than assumed --
# getting this wrong silently starts N replicas on GPU 0 and they OOM each
# other, which looks like a memory bug rather than a launch bug.
if [[ -z "$GPUS" ]]; then
    GPUS="$("$PY" -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo 0)"
fi
if [[ "$GPUS" -lt 1 ]]; then
    echo "No CUDA GPUs detected. Set --gpus explicitly if this is wrong."
    exit 1
fi

# Replicas default to one per GPU, but the two are separable. The model is
# ~14 GiB, so an 80 GiB card holds several -- and on a single-GPU box this is
# the only way to exercise multi-replica routing at all. Replica i is pinned to
# GPU (i % GPUS), so replicas beyond GPUS share cards round-robin.
REPLICAS="${REPLICAS:-$GPUS}"
if [[ "$REPLICAS" -lt 1 ]]; then
    echo "--replicas must be >= 1 (got '$REPLICAS')."
    exit 1
fi
if [[ "$REPLICAS" -gt "$GPUS" ]]; then
    echo "Note: $REPLICAS replicas across $GPUS GPU(s) -- they will share cards."
    echo "      Each holds its own full copy of the weights; size --batch-max-size"
    echo "      for the per-GPU total, not per replica."
fi

if [[ ${#QUANT_ARGS[@]} -gt 0 && "$BACKEND" != "fast_dense" ]]; then
    echo "Quantization needs --backend fast_dense (got '$BACKEND')."
    exit 1
fi

echo "Placement plan:"
for ((i = 0; i < REPLICAS; i++)); do
    echo "  replica $i -> GPU $((i % GPUS)), port $((REPLICA_PORT_BASE + i))"
done
echo ""

if [[ "$DRY_RUN" == "1" ]]; then
    echo "--dry-run: nothing launched."
    exit 0
fi

mkdir -p "$LOG_DIR"
PIDS=()
BACKENDS=""

cleanup() {
    echo
    echo "Stopping ${#PIDS[@]} process(es)..."
    for pid in "${PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
    wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "================================================================"
echo "  LLaDA-MoE data-parallel deployment"
echo "================================================================"
echo "  Replicas : $REPLICAS across $GPUS GPU(s)"
echo "  Router   : http://0.0.0.0:$PORT"
echo "  Backend  : $BACKEND${QUANT_ARGS[*]:+  ${QUANT_ARGS[*]}}"
echo "  Logs     : $LOG_DIR"
echo ""

for ((i = 0; i < REPLICAS; i++)); do
    rport=$((REPLICA_PORT_BASE + i))
    gpu=$((i % GPUS))
    # CUDA_VISIBLE_DEVICES pins the replica; inside it the GPU is always
    # cuda:0, which is also what makes each replica's tp_size == 1 and keeps
    # the batching path enabled.
    #
    # Each replica needs its OWN rendezvous port. src.server calls
    # init_distributed() unconditionally, which stands up a TCPStore -- and its
    # default MASTER_PORT is 29500 for everyone, so on a single host only the
    # first replica binds and the rest die with EADDRINUSE. Without this, N
    # replicas on one node means N-1 dead GPUs.
    #
    # All four variables must be set together: distributed.py fills in the
    # single-process defaults only when MASTER_ADDR is absent, so setting the
    # port alone would skip RANK/WORLD_SIZE and fail inside env:// rendezvous.
    env CUDA_VISIBLE_DEVICES="$gpu" \
        MASTER_ADDR=127.0.0.1 \
        MASTER_PORT=$((29500 + i)) \
        RANK=0 \
        WORLD_SIZE=1 \
        ${BATCH_MAX_SIZE:+BATCH_MAX_SIZE="$BATCH_MAX_SIZE"} \
        "$PY" -m src.server \
            --weight-dir "$WEIGHT_DIR" \
            --port "$rport" \
            --host 127.0.0.1 \
            --device cuda:0 \
            --backend "$BACKEND" \
            "${QUANT_ARGS[@]}" > "$LOG_DIR/replica_$i.log" 2>&1 &
    PIDS+=("$!")
    BACKENDS="${BACKENDS:+$BACKENDS,}http://127.0.0.1:$rport"
    echo "  replica $i -> GPU $gpu, port $rport (pid ${PIDS[-1]})"
done

echo ""
echo "Waiting for replicas to load (a 7B checkpoint takes a few minutes)..."
for ((i = 0; i < REPLICAS; i++)); do
    rport=$((REPLICA_PORT_BASE + i))
    for _ in $(seq 240); do
        if curl -sf "http://127.0.0.1:$rport/health" >/dev/null 2>&1; then
            echo "  replica $i ready"
            break
        fi
        # Fail fast if the replica died rather than waiting out the full window
        # on a process that is never coming back.
        if ! kill -0 "${PIDS[$i]}" 2>/dev/null; then
            echo "  replica $i DIED -- last lines of $LOG_DIR/replica_$i.log:"
            tail -20 "$LOG_DIR/replica_$i.log"
            exit 1
        fi
        sleep 5
    done
done

echo ""
echo "All replicas up. Starting router..."
echo ""
# NOT exec: exec would replace this shell and discard the EXIT/INT trap above,
# so killing the router would orphan every replica -- N processes still holding
# N GPUs, with nothing left to explain why they are busy. Running it as a child
# keeps the trap, so Ctrl-C really does take the whole deployment down.
"$PY" -m src.router --backends "$BACKENDS" --port "$PORT" --host 0.0.0.0 &
ROUTER_PID=$!
PIDS+=("$ROUTER_PID")
wait "$ROUTER_PID"
