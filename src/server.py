"""
OpenAI-compatible chat completions server for LLaDA-MoE-7B-A1B-Instruct.

Backends:
  - "ours"        : Dense baseline (src.generate)
  - "ours_kv"     : Original sparse-dLLM + SparseD (src.generate_KVcache)
  - "fast_dense"  : Fast dense cached (Option A) — TP+EP, Triton fused MoE,
                    block-wise KV cache. Static top-8 experts (the model's
                    native routing config).
  - "hf"          : HuggingFace reference

Usage:
    python3 -m src.server --weight-dir ./weights --port 8000 --backend fast_dense
"""

import argparse
import os
import queue
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

import torch
import uvicorn
import threading
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

app = FastAPI(title="LLaDA-MoE Inference Server")

MODEL: Optional[torch.nn.Module] = None
TOKENIZER = None
DEVICE = "cuda:0"
BACKEND = "ours"

DEFAULT_STEPS        = 512
DEFAULT_GEN_LENGTH   = 512
DEFAULT_BLOCK_LENGTH = 32
DEFAULT_TEMPERATURE  = 0.0
DEFAULT_CFG_SCALE    = 0.0
DEFAULT_REMASKING    = "low_confidence"

request_lock = threading.Lock()

# ── Server-side request batching (fast_dense, single-GPU only) ─────────────────
#
# Scoped deliberately narrowly: only BACKEND == "fast_dense" with tp_size == 1
# (a plain single-GPU instance, e.g. one DP replica) routes through here. TP+EP
# (tp_size > 1) keeps the original one-request-at-a-time request_lock path
# below completely unchanged -- batching a request there would mean the batch
# itself has to survive dist.broadcast_object_list() to worker_loop() every
# step, which is real additional distributed-coordination surface area on top
# of code that's already the most failure-prone part of this project (see
# INVESTIGATION_LOG.md's KV-cache collapse bug). Validate batching's actual
# throughput win on the simple path first.
#
# model_update/generate.py's _generate_block_cached already handles a batch
# dimension B generically (get_num_transfer_tokens and the per-row top-k
# selection both operate per batch row already) -- no changes needed there.
# What's missing is purely server-side: collecting concurrent requests into
# one generate_cached() call instead of serializing them one at a time.
#
# Batching is restricted to requests that share an IDENTICAL
# (gen_length, steps, block_length, prompt_token_length) key. Different
# prompt lengths would need padding + an attention mask, which nothing in
# model_update/model.py supports today (F.scaled_dot_product_attention is
# always called with attn_mask=None) -- unsafe to fake without it, so
# mismatched requests simply go in separate batches rather than being forced
# together.
#
# The per-batch-row confidence top-k selection in generate.py's
# select_transfer_indices() was vectorized to a single `.item()` per step
# (not per row) as a separate fix earlier -- batching does not multiply that
# cost with B, so it is not a concern for this batching path.
#
# Default of 64 (was 8): validated on a single A6000 via
# eval/check_time_inference.py --batch-size {1,32,64} under Nsight Compute
# (occupancy/memory-bandwidth/compute-throughput all kept improving up to at
# least 64) and then confirmed end-to-end with real concurrent HTTP traffic
# (eval/throughput/run_throughput.py --concurrency 64 --n-requests 128):
# BATCH_MAX_SIZE=8 -> 144.0 tok/s, p50=56.18s; BATCH_MAX_SIZE=64 -> 199.6
# tok/s, p50=36.89s. Not a throughput/latency tradeoff -- 8 forces 16
# sequential batch rounds for 128 concurrent requests, so most requests sit
# queued behind earlier rounds; 64 needs only 2 rounds, so most requests
# start almost immediately.
#
# Collector/executor split, not a single worker: the original single-thread
# design called generate_cached() (via _run_batch) directly inside the same
# loop that collects requests -- so while one batch was generating (tens of
# seconds at production gen_length), a request arriving mid-batch got zero
# collection progress: it sat untouched in _pending_queue until the *entire*
# current batch finished, THEN had to wait out its own fresh BATCH_WAIT_S
# window on top of that. _batch_collector now only forms batches and hands
# each one to _ready_batches (a queue.Queue); _batch_executor is the sole
# consumer, the only thread that ever calls generate_cached(), so GPU work
# still runs strictly one batch at a time -- no change to generation
# semantics or single-GPU serialization. What changes is that the collector
# keeps accumulating and forming the *next* batch the entire time the
# executor is busy with the current one, so as soon as the executor finishes,
# the next batch is usually already fully formed and starts immediately,
# instead of paying a fresh collection window on top of the wait.
# True mid-generation admission (splicing a new request into an
# already-running batch, freeing a row the moment ITS block finishes rather
# than waiting for the whole batch) is a materially bigger change -- it needs
# per-row early-exit decoding first (this project uses a fixed, uniform
# step schedule per block today, so rows never finish at different times to
# begin with) plus restructuring _generate_block_cached to support a
# dynamically resized batch tensor mid-loop. Deliberately out of scope here;
# this fix only removes the dead time between batches, which is real and
# unconditional regardless of the decoding algorithm.

BATCH_MAX_SIZE = int(os.environ.get("BATCH_MAX_SIZE", 64))
BATCH_WAIT_S = float(os.environ.get("BATCH_WAIT_S", 0.05))
BATCH_POLL_INTERVAL_S = float(os.environ.get("BATCH_POLL_INTERVAL_S", 0.005))
BATCH_REQUEST_TIMEOUT_S = float(os.environ.get("BATCH_REQUEST_TIMEOUT_S", 300))
MAX_THREADPOOL_WORKERS = int(os.environ.get("MAX_THREADPOOL_WORKERS", 128))


class _PendingRequest:
    def __init__(self, input_ids, gen_length, steps, block_length, temperature,
                 confidence_threshold=None, low_confidence_threshold=0.4, no_cache=False):
        self.input_ids = input_ids
        self.gen_length = gen_length
        self.steps = steps
        self.block_length = block_length
        self.temperature = temperature
        self.confidence_threshold = confidence_threshold
        self.low_confidence_threshold = low_confidence_threshold
        self.no_cache = no_cache
        self.event = threading.Event()
        self.result = None
        self.error = None

    @property
    def key(self):
        # confidence_threshold/low_confidence_threshold/no_cache included:
        # requests must share the exact same generation config to be batched
        # into one generate_cached()/generate_dense() call, same reasoning as
        # gen_length/steps/block_length already being here.
        #
        # prompt_token_length is deliberately NOT here. It used to be, because
        # attention ran with attn_mask=None and could not tolerate padding --
        # which meant two users whose prompts tokenized to 51 and 52 tokens
        # could never share a batch. Under real traffic that fragmented the
        # queue into batches of one or two and the batching path stopped
        # mattering; every throughput figure in the README was measured with
        # --fixed-prompt for exactly that reason. generate_cached now accepts
        # left-padded prompts plus prompt_lens, so length is no longer part of
        # a request's identity.
        #
        # The no_cache (generate_dense) path still cannot pad, and is grouped
        # by length below rather than here.
        return (
            self.gen_length, self.steps, self.block_length,
            self.confidence_threshold, self.low_confidence_threshold, self.no_cache,
            # generate_dense has no prompt_lens parameter, so mixed lengths
            # would silently mis-slice its output. Keep those grouped by length.
            self.input_ids.shape[1] if self.no_cache else None,
        )


_pending_lock = threading.Lock()
_pending_queue: list["_PendingRequest"] = []
_new_request_event = threading.Event()

# Formed batches waiting for _batch_executor (the sole GPU-generation
# consumer) to get to them. Unbounded, same as _pending_queue always was --
# in practice bounded anyway by MAX_THREADPOOL_WORKERS (a request can't even
# be collected until its HTTP handler is running), so at most a couple of
# BATCH_MAX_SIZE-sized batches can ever be queued here at once.
_ready_batches: "queue.Queue[list[_PendingRequest]]" = queue.Queue()


PROFILE_BATCHES = os.environ.get("PROFILE_BATCHES", "0") == "1"
PROFILE_DIR = os.environ.get("PROFILE_DIR", "traces")


_last_batch_end_time: Optional[float] = None  # only _run_batch touches this -- single executor thread, no lock needed


def _run_batch(batch: list["_PendingRequest"]):
    global _last_batch_end_time
    if not batch:
        return
    # Visibility into what's actually forming -- this was the open question
    # from the concurrency investigation: does the collection window really
    # produce batches near BATCH_MAX_SIZE, or fall back to 1-2 under GIL
    # contention? No way to know without this line.
    #
    # gap_since_prev: the direct measurement of what the collector/executor
    # split (see _batch_collector's docstring) is meant to eliminate -- dead
    # time between one batch finishing and the next starting. Near-zero means
    # the next batch was already fully formed and ready the instant the GPU
    # freed up; a large gap means either genuinely no traffic was waiting, or
    # the pipelining isn't working as intended.
    now = time.monotonic()
    gap_str = f"{now - _last_batch_end_time:.3f}s" if _last_batch_end_time is not None else "n/a (first batch)"
    print(f"[batch] size={len(batch)} key={batch[0].key} gap_since_prev_batch_end={gap_str}", flush=True)
    try:
        from model_update.generate import generate_cached, generate_dense
        first = batch[0]

        # Mixed prompt lengths are LEFT-padded so every row's prompt ends at
        # the same column, which is what lets the block loop use one
        # block_start for the batch. prompt_lens travels alongside so the model
        # can give each row its own RoPE positions and mask the pad columns
        # out; without it a shorter prompt would be encoded as though it began
        # partway into the sequence.
        prompt_lens = None
        widths = {r.input_ids.shape[1] for r in batch}
        if len(widths) == 1:
            input_ids = torch.cat([r.input_ids for r in batch], dim=0)
        else:
            max_len = max(widths)
            pad_id = TOKENIZER.pad_token_id
            if pad_id is None:
                # Value is irrelevant -- these columns are masked out of
                # attention -- but it must be a valid embedding index.
                pad_id = TOKENIZER.eos_token_id or 0
            rows = []
            for r in batch:
                ids = r.input_ids
                pad = max_len - ids.shape[1]
                if pad:
                    ids = torch.cat(
                        [torch.full((1, pad), pad_id, dtype=ids.dtype, device=ids.device), ids],
                        dim=1,
                    )
                rows.append(ids)
            input_ids = torch.cat(rows, dim=0)
            prompt_lens = torch.tensor(
                [r.input_ids.shape[1] for r in batch], device=input_ids.device
            )
        if first.no_cache:
            # generate_dense has no confidence_threshold support (see its
            # docstring) -- ChatRequest already rejects that combination
            # before a request ever reaches here, so gen_kwargs is
            # deliberately narrower for this path.
            gen_fn = generate_dense
            gen_kwargs = dict(
                gen_length=first.gen_length,
                steps=first.steps,
                block_length=first.block_length,
                temperature=first.temperature,
            )
        else:
            gen_fn = generate_cached
            gen_kwargs = dict(
                gen_length=first.gen_length,
                steps=first.steps,
                block_length=first.block_length,
                temperature=first.temperature,
                confidence_threshold=first.confidence_threshold,
                low_confidence_threshold=first.low_confidence_threshold,
                prompt_lens=prompt_lens,
            )
        if PROFILE_BATCHES:
            # Chrome/Perfetto trace JSON -- shows actual GPU busy vs idle
            # time during this batch's generate_cached() call, not just
            # wall-clock totals. Open at chrome://tracing or
            # https://ui.perfetto.dev/ (load the file). One trace per
            # batch, named with its size so it's identifiable alongside
            # the [batch] log line above. Real overhead from the profiler
            # itself, so this is gated off by default -- one-off
            # diagnostic capture, not for production.
            # CUDA-only, not CPU+CUDA: tracing every Python-level op
            # dispatch/call-stack entry across 128 steps x 16 layers x a
            # 63-sequence batch produced 500-640MB trace files in practice
            # -- impractical to transfer and likely to hang a browser
            # viewer even if transferred. CUDA-only answers the actual
            # question (is the GPU busy or idle) directly, at a small
            # fraction of the size. Set PROFILE_CPU=1 to add CPU activity
            # back for a deeper look, at the same size cost.
            os.makedirs(PROFILE_DIR, exist_ok=True)
            activities = [torch.profiler.ProfilerActivity.CUDA]
            if os.environ.get("PROFILE_CPU", "0") == "1":
                activities.insert(0, torch.profiler.ProfilerActivity.CPU)
            with torch.no_grad(), torch.profiler.profile(activities=activities) as prof:
                out_ids = gen_fn(MODEL, input_ids, **gen_kwargs)
            trace_path = os.path.join(
                PROFILE_DIR, f"batch_size{len(batch)}_{int(time.time() * 1000)}.json"
            )
            prof.export_chrome_trace(trace_path)
            print(f"[profile] wrote {trace_path}", flush=True)
        else:
            with torch.no_grad():
                out_ids = gen_fn(MODEL, input_ids, **gen_kwargs)
        for i, req in enumerate(batch):
            req.result = out_ids[i : i + 1]
    except Exception as e:
        for req in batch:
            req.error = e
    finally:
        for req in batch:
            req.event.set()
        _last_batch_end_time = time.monotonic()


def _batch_collector():
    """Runs on a dedicated background thread, started once at server startup
    (see main()). Forms batches only -- never calls generate_cached() itself,
    that's _batch_executor's job (see its docstring for why they're split).

    Early-exit polling, not a fixed sleep-then-check: once the first request
    of a new group arrives, poll every BATCH_POLL_INTERVAL_S for either (a)
    BATCH_MAX_SIZE matching-key requests having accumulated, or (b)
    BATCH_WAIT_S total having elapsed since the group started -- whichever
    comes first. A fixed `sleep(BATCH_WAIT_S)` (the original version) always
    pays the full window even when a large burst arrives almost immediately,
    which matters a lot at higher target concurrency (e.g. 64): a burst that
    large will usually hit BATCH_MAX_SIZE well before BATCH_WAIT_S elapses,
    and there's no reason to keep those requests waiting once it has.
    Anything left over (mismatched key, or over the cap) stays queued for
    the next round with a fresh deadline."""
    while True:
        _new_request_event.wait()
        deadline = time.monotonic() + BATCH_WAIT_S
        while True:
            with _pending_lock:
                if not _pending_queue:
                    _new_request_event.clear()
                    batch = []
                    break
                key = _pending_queue[0].key
                matching = sum(1 for r in _pending_queue if r.key == key)
                if matching >= BATCH_MAX_SIZE or time.monotonic() >= deadline:
                    batch, remaining = [], []
                    for req in _pending_queue:
                        if req.key == key and len(batch) < BATCH_MAX_SIZE:
                            batch.append(req)
                        else:
                            remaining.append(req)
                    _pending_queue[:] = remaining
                    if not _pending_queue:
                        _new_request_event.clear()
                    # else: event stays set, loop continues immediately for
                    # the remainder rather than waiting for a new arrival.
                    break
            time.sleep(BATCH_POLL_INTERVAL_S)
        if batch:
            _ready_batches.put(batch)
        # Loop straight back to collecting the next group -- does NOT wait
        # for _batch_executor to consume this one. That's the whole point:
        # collection and GPU execution now overlap instead of alternating.


def _batch_executor():
    """Runs on a dedicated background thread, started once at server startup
    (see main()). The ONLY thread that ever calls _run_batch()/generate_cached()
    -- pulling one fully-formed batch at a time off _ready_batches and running
    it to completion before pulling the next. This preserves exactly the same
    single-GPU serialization as before (one generation in flight at a time);
    what's different is that _batch_collector is free to keep forming the
    NEXT batch the whole time this thread is busy with the current one, so
    there's usually no gap between one batch finishing and the next starting."""
    while True:
        batch = _ready_batches.get()
        _run_batch(batch)


@app.on_event("startup")
async def _raise_threadpool_limit():
    """anyio's thread limiter can only be fetched/set from inside a running
    event loop -- calling this synchronously in main() before uvicorn.run()
    starts one raises anyio.NoEventLoopError. A FastAPI startup hook runs
    after uvicorn's loop is live, so it works here. chat_completions() is a
    sync `def`, so FastAPI/Starlette runs each request in a worker thread via
    this threadpool; its default capacity (~40) can otherwise queue requests
    before they even reach _batch_collector -- indistinguishable from batching
    "not helping" unless raised explicitly for higher target concurrency
    (e.g. 64)."""
    from model_update.distributed import get_tp_size
    if BACKEND == "fast_dense" and get_tp_size() == 1:
        import anyio.to_thread
        anyio.to_thread.current_default_thread_limiter().total_tokens = MAX_THREADPOOL_WORKERS
        print(f"Raised ASGI threadpool capacity to {MAX_THREADPOOL_WORKERS}.")


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "inclusionAI/LLaDA-MoE-7B-A1B-Instruct"
    messages: list[Message]
    max_tokens: int = DEFAULT_GEN_LENGTH
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    steps: int = DEFAULT_STEPS
    block_length: int = DEFAULT_BLOCK_LENGTH
    cfg_scale: float = DEFAULT_CFG_SCALE
    remasking: str = DEFAULT_REMASKING
    # Opt-in hierarchical threshold-based decoding (model_update/generate.py's
    # confidence_threshold, ported from dInfer's HierarchyDecoder) instead
    # of the default fixed-per-step reveal schedule. None (default) is the
    # exact original behavior. Validated on MMLU-Pro (44.0% acc) and GSM8K
    # (see README's "Adaptive Decoding" section for current numbers) -- but
    # see generate_cached's docstring for the required steps_per_block >=
    # block_length / 2 ratio before trusting output with this set (violating
    # it causes degenerate repetition). Only applies to the fast_dense/
    # single-GPU batched path (see _PendingRequest).
    confidence_threshold: Optional[float] = None
    low_confidence_threshold: float = 0.4
    # Opt-in: use model_update/generate.py's generate_dense (same model
    # class/weights/Triton fused MoE as the normal path, but NO KV cache --
    # every step recomputes the full sequence from scratch) instead of
    # generate_cached. Added to test whether block-wise KV caching itself is
    # the source of the ~6pt residual accuracy gap vs the HF reference found
    # in the Correctness Verification investigation (README.md /
    # INVESTIGATION_LOG.md Part 2 §2.9) -- generate_dense's only difference
    # from HF's reference path at that point is caching, so if accuracy with
    # no_cache=True closes most of that gap, caching is confirmed as the
    # cause; if it doesn't, something else is. Incompatible with
    # confidence_threshold (generate_dense doesn't implement it). Only
    # applies to the fast_dense/single-GPU batched path.
    no_cache: bool = False


class CompletionRequest(BaseModel):
    """Raw text completion -- no chat template, no system message, unlike
    ChatRequest. Added specifically to test whether dInfer's own eval
    methodology (dInfer/evaluations/tasks/gsm8k/gsm8k-llada-moe.yaml's
    doc_to_text: "Question: ...\\nPlease reason step by step\\nAnswer:",
    fed as raw few-shot text completion with no chat markup at all, no
    system message, and until: ["Question:", "</s>", "<|im_end|>"] stop
    sequences) scores differently than this project's existing
    /v1/chat/completions path, which wraps the same prompt text through
    the Instruct model's chat template plus an added system-prompt
    instruction -- a real methodology gap identified by comparing this
    project's harness against dInfer's actual eval config, suspected of
    letting the model's chat-tuned formatting habits (markdown, "**bold**",
    preambles) leak into what should be a clean raw-completion answer."""
    model: str = "inclusionAI/LLaDA-MoE-7B-A1B-Instruct"
    prompt: str
    max_tokens: int = DEFAULT_GEN_LENGTH
    temperature: float = DEFAULT_TEMPERATURE
    steps: int = DEFAULT_STEPS
    block_length: int = DEFAULT_BLOCK_LENGTH
    # Truncate the generated text at the first occurrence of any of these
    # substrings, applied client-side to the decoded text (this project's
    # generation loop has no true early-stopping / causal KV cache the way
    # an AR model's until-sequence stopping criterion would use -- diffusion
    # generation always fills the full max_tokens budget regardless).
    stop: Optional[list[str]] = None
    confidence_threshold: Optional[float] = None
    low_confidence_threshold: float = 0.4


@app.get("/health")
def health():
    return {"status": "ok", "model": "LLaDA-MoE-7B-A1B-Instruct", "backend": BACKEND}


@app.get("/v1/quantization")
def quantization_status():
    """What is actually being served, so an A/B arm is verifiable over HTTP
    rather than inferred from which terminal launched it."""
    return QUANTIZATION or {"quantized": False, "label": "BF16 (unmodified)"}


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{"id": "inclusionAI/LLaDA-MoE-7B-A1B-Instruct", "object": "model"}],
    }


# ── Config switching endpoint (for automated testing) ──────────────────────────
@app.post("/v1/config")
def set_config(config: dict):
    """Switch generation config at runtime (for testing only)."""
    global BACKEND
    new_backend = config.get("backend")
    if new_backend in ["ours", "ours_kv", "fast_dense", "hf"]:
        BACKEND = new_backend
        return {"status": "ok", "backend": BACKEND}
    return JSONResponse(
        status_code=400,
        content={"error": f"Unknown backend: {new_backend}. Valid: ours, ours_kv, fast_dense, hf"}
    )


@app.get("/v1/config")
def get_config():
    return {"backend": BACKEND}


# ── Chat completions ───────────────────────────────────────────────────────────
@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    messages = [{"role": m.role, "content": m.content} for m in req.messages]

    prompt = TOKENIZER.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    input_ids = TOKENIZER(prompt, return_tensors="pt")["input_ids"].to(DEVICE)

    gen_length   = req.max_tokens
    block_length = req.block_length
    if gen_length % block_length != 0:
        gen_length = ((gen_length // block_length) + 1) * block_length

    steps = req.steps
    if steps % (gen_length // block_length) != 0:
        steps = (gen_length // block_length) * max(1, steps // (gen_length // block_length))

    from model_update.distributed import get_tp_size

    t0 = time.time()

    if BACKEND == "fast_dense" and get_tp_size() == 1:
        # Batched path -- see the "Server-side request batching" block above
        # for why this is scoped to single-GPU only.
        if req.no_cache and req.confidence_threshold is not None:
            raise HTTPException(
                status_code=400,
                detail="no_cache and confidence_threshold are incompatible -- "
                       "generate_dense (no_cache=True) doesn't implement threshold decoding.",
            )
        pending = _PendingRequest(
            input_ids, gen_length, steps, block_length, req.temperature,
            confidence_threshold=req.confidence_threshold,
            low_confidence_threshold=req.low_confidence_threshold,
            no_cache=req.no_cache,
        )
        with _pending_lock:
            _pending_queue.append(pending)
            _new_request_event.set()
        if not pending.event.wait(timeout=BATCH_REQUEST_TIMEOUT_S):
            raise HTTPException(status_code=504, detail="Timed out waiting for a batch slot")
        if pending.error is not None:
            raise HTTPException(status_code=500, detail=str(pending.error))
        out_ids = pending.result
    else:
        with torch.no_grad(), request_lock:
            if BACKEND == "ours":
                from src.generate import generate
                out_ids = generate(
                    MODEL,
                    input_ids,
                    gen_length=gen_length,
                    steps=steps,
                    block_length=block_length,
                    temperature=req.temperature,
                    cfg_scale=req.cfg_scale,
                    remasking=req.remasking,
                )
            elif BACKEND == "ours_kv":
                from src.generate_KVcache import generate_cached as generate_kv
                out_ids = generate_kv(
                    MODEL,
                    input_ids,
                    gen_length=gen_length,
                    steps=steps,
                    block_length=block_length,
                    temperature=req.temperature,
                    cfg_scale=req.cfg_scale,
                    remasking=req.remasking,
                )
            elif BACKEND == "fast_dense":
                # Only reached here when get_tp_size() > 1 (TP+EP) -- the
                # single-GPU case is routed through the batched path above.
                from model_update.model import LLaDAMoEKV
                from model_update.generate import generate_cached

                import torch.distributed as dist
                req_obj = {
                    "type": "generate_cached",
                    "input_ids": input_ids.cpu(),
                    "gen_length": gen_length,
                    "steps": steps,
                    "block_length": block_length,
                    "temperature": req.temperature,
                }
                dist.broadcast_object_list([req_obj], src=0)

                out_ids = generate_cached(
                    MODEL,
                    input_ids,
                    gen_length=gen_length,
                    steps=steps,
                    block_length=block_length,
                    temperature=req.temperature,
                )
            elif BACKEND == "hf":
                # HuggingFace's .generate() blocks diffusion models in newer versions.
                # We wrap the HF model to return logits and use our diffusion generate loop.
                from src.generate import generate
                class HFWrapper:
                    def __init__(self, m): self.m = m
                    def __call__(self, x, **kwargs): return self.m(x, **kwargs).logits

                out_ids = generate(
                    HFWrapper(MODEL),
                    input_ids,
                    gen_length=gen_length,
                    steps=steps,
                    block_length=block_length,
                    temperature=req.temperature,
                    cfg_scale=req.cfg_scale,
                    remasking=req.remasking,
                )
    elapsed = time.time() - t0

    generated = out_ids[0].tolist()
    eos_id = TOKENIZER.eos_token_id
    if eos_id in generated:
        generated = generated[: generated.index(eos_id)]
    while generated and generated[-1] == 156895:  # MASK_ID
        generated.pop()

    text = TOKENIZER.decode(generated, skip_special_tokens=True)
    tokens_generated = len(generated)

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": input_ids.shape[1],
            "completion_tokens": tokens_generated,
            "total_tokens": input_ids.shape[1] + tokens_generated,
        },
        "timing": {"generation_seconds": round(elapsed, 3)},
    }


# ── Raw text completions (no chat template) ────────────────────────────────────
@app.post("/v1/completions")
def completions(req: CompletionRequest):
    """See CompletionRequest's docstring. Scoped to the fast_dense/single-GPU
    batched path only, same restriction confidence_threshold already has --
    this is an experimental A/B path, not meant to cover every backend."""
    from model_update.distributed import get_tp_size
    if not (BACKEND == "fast_dense" and get_tp_size() == 1):
        raise HTTPException(
            status_code=400,
            detail="/v1/completions only supports backend=fast_dense with tp_size=1",
        )

    input_ids = TOKENIZER(req.prompt, return_tensors="pt")["input_ids"].to(DEVICE)

    gen_length   = req.max_tokens
    block_length = req.block_length
    if gen_length % block_length != 0:
        gen_length = ((gen_length // block_length) + 1) * block_length

    steps = req.steps
    if steps % (gen_length // block_length) != 0:
        steps = (gen_length // block_length) * max(1, steps // (gen_length // block_length))

    t0 = time.time()
    pending = _PendingRequest(
        input_ids, gen_length, steps, block_length, req.temperature,
        confidence_threshold=req.confidence_threshold,
        low_confidence_threshold=req.low_confidence_threshold,
    )
    with _pending_lock:
        _pending_queue.append(pending)
        _new_request_event.set()
    if not pending.event.wait(timeout=BATCH_REQUEST_TIMEOUT_S):
        raise HTTPException(status_code=504, detail="Timed out waiting for a batch slot")
    if pending.error is not None:
        raise HTTPException(status_code=500, detail=str(pending.error))
    out_ids = pending.result
    elapsed = time.time() - t0

    generated = out_ids[0].tolist()
    eos_id = TOKENIZER.eos_token_id
    if eos_id in generated:
        generated = generated[: generated.index(eos_id)]
    while generated and generated[-1] == 156895:  # MASK_ID
        generated.pop()

    text = TOKENIZER.decode(generated, skip_special_tokens=True)
    tokens_generated = len(generated)

    finish_reason = "length"
    if req.stop:
        cut = min((i for i in (text.find(s) for s in req.stop) if i != -1), default=None)
        if cut is not None:
            text = text[:cut]
            finish_reason = "stop"

    return {
        "id": f"cmpl-{uuid.uuid4().hex[:8]}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {"index": 0, "text": text, "finish_reason": finish_reason}
        ],
        "usage": {
            "prompt_tokens": input_ids.shape[1],
            "completion_tokens": tokens_generated,
            "total_tokens": input_ids.shape[1] + tokens_generated,
        },
        "timing": {"generation_seconds": round(elapsed, 3)},
    }


# ── Startup ────────────────────────────────────────────────────────────────────
#: Set by apply_quantization(); reported by /v1/quantization so a served model
#: can be identified over HTTP. Without it there is no way to tell a quantized
#: server from a BF16 one, and forgetting to restart between A/B arms silently
#: produces two identical result files that look like "quantization changed
#: nothing".
QUANTIZATION: Optional[dict] = None


def apply_quantization(bits: int = 8, group_size: int = 128,
                       mode: str = "packed", fused: bool = True) -> dict:
    """Quantize the loaded model's MoE experts using the LLaDA_Quant toolkit.

    LLaDA_Quant is an **optional** dependency -- deliberately not in
    requirements.txt, because this engine is BF16 and everything it claims is
    measured that way. It is imported lazily, here, so a server that never
    passes --quantize neither needs it installed nor pays for importing it.

    Order matters and is why this runs after load_model() rather than inside
    it: PACKED mode replaces w1/w2 with properties that dequantize on read, so
    any in-place write into them afterwards (which is exactly what
    load_state_dict_from_unfused does) would land in a temporary and vanish.
    The experts must be fully populated first.
    """
    global QUANTIZATION

    if BACKEND != "fast_dense":
        raise SystemExit(
            f"--quantize needs --backend fast_dense; got '{BACKEND}'. The other "
            "backends do not build fused expert blocks (w1/w2), so there is "
            "nothing for the expert adapter to match."
        )

    try:
        from LLaDA_Quant.api import quantize_model
        from LLaDA_Quant.config import QuantConfig
    except ImportError as exc:
        raise SystemExit(
            "--quantize needs the LLaDA_Quant toolkit, which is an optional "
            "dependency and is not installed.\n"
            "  pip install -e /path/to/LLaDA_Quant\n"
            f"(import failed: {exc})"
        ) from exc

    print(f"Quantizing MoE experts: INT{bits} g{group_size} {mode}...")
    config = QuantConfig(
        bits=bits,
        group_size=group_size,
        targets=("expert",),
        execution_mode=mode,
        expect_expert_blocks=MODEL.cfg.NL,
    )
    result = quantize_model(MODEL, config)

    fused_blocks = []
    if fused and mode == "packed" and bits == 8:
        from LLaDA_Quant.runtime import fused_block

        # strict: a run that quietly fell back to dequantize-per-access would
        # still serve, ~6x slower, and be written up as a fused number.
        fused_blocks = fused_block.install(MODEL, strict=True)
    elif fused and bits == 4:
        print("  note: INT4 has no fused kernel; using dequantize-per-access.")

    QUANTIZATION = {
        "quantized": True,
        "bits": bits,
        "group_size": group_size,
        "execution_mode": mode,
        "fused_kernel": bool(fused_blocks),
        "fused_blocks": len(fused_blocks),
        "expert_blocks": len(result.expert_blocks),
    }
    print(f"  {QUANTIZATION}")
    if mode == "reference":
        print("  REFERENCE mode uses MORE memory than not quantizing. "
              "Accuracy evaluation only -- never report it as a saving.")
    return QUANTIZATION


def load_model(weight_dir: str, device: str, backend: str):
    global MODEL, TOKENIZER, DEVICE, BACKEND
    DEVICE = device
    BACKEND = backend
    
    if "cuda" in device:
        torch.cuda.set_device(device)

    print(f"Loading tokenizer from {weight_dir}...")
    TOKENIZER = AutoTokenizer.from_pretrained(weight_dir, trust_remote_code=True)

    print(f"Loading model with backend '{backend}'...")
    if backend == "ours":
        from src.model import LLaDAMoE, load_weights
        MODEL = LLaDAMoE().to(torch.bfloat16).to(device).eval()
        load_weights(MODEL, weight_dir, verbose=True)
    elif backend == "fast_dense":
        from model_update.model import LLaDAMoEKV, TritonFusedMoEBlock
        from model_update.distributed import load_weights_tp, get_tp_rank

        print(f"Instantiating unfused model to load weights on Rank {get_tp_rank()}...")
        MODEL = LLaDAMoEKV(use_fused_moe=False).to(torch.bfloat16).eval()
        load_weights_tp(MODEL, weight_dir, verbose=True)

        print("Converting to Fused MoE blocks...")
        for i, layer in enumerate(MODEL.layers):
            fused_mlp = TritonFusedMoEBlock(layer.mlp.cfg).to(torch.bfloat16)
            fused_mlp.load_state_dict_from_unfused(layer.mlp)
            layer.mlp = fused_mlp

        MODEL = MODEL.to(device)
    elif backend == "ours_kv":
        from src.Model_KVcache import LLaDAMoEKV
        MODEL = LLaDAMoEKV().to(torch.bfloat16).to(device).eval()
        load_weights(MODEL, weight_dir, verbose=True)
    elif backend == "hf":
        from transformers import AutoModelForCausalLM
        MODEL = AutoModelForCausalLM.from_pretrained(
            weight_dir,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
        ).to(device).eval()
    else:
        raise ValueError(f"Unknown backend: {backend}")
    print("Model ready.\n")


def main():
    from model_update.distributed import init_distributed, get_tp_rank, get_tp_size
    init_distributed()
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight-dir", default="weights")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="0.0.0.0")
    # In distributed mode, we override device with local rank
    ap.add_argument("--device", default=f"cuda:{get_tp_rank()}" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--backend", choices=["ours", "ours_kv", "fast_dense", "hf"], default="ours")
    ap.add_argument("--quantize", choices=["int8", "int4"], default=None,
                    help="Quantize the MoE experts via the LLaDA_Quant toolkit "
                         "(optional dependency, not installed by requirements.txt). "
                         "Requires --backend fast_dense. int8 gets the fused W8A16 "
                         "kernel; int4 has none and runs dequantize-per-access.")
    ap.add_argument("--quant-group-size", type=int, default=128)
    ap.add_argument("--quant-mode", choices=["packed", "reference"], default="packed",
                    help="packed: real memory reduction (deploy). reference: same "
                         "numerics at BF16 speed but MORE memory than not "
                         "quantizing -- accuracy evaluation only.")
    ap.add_argument("--no-fused-quant", action="store_true",
                    help="Skip the fused W8A16 kernel and dequantize on every "
                         "weight access instead. ~6x slower; for A/B only.")
    args = ap.parse_args()

    # Override device if TP is active
    if get_tp_size() > 1:
        args.device = f"cuda:{get_tp_rank()}"

    load_model(args.weight_dir, args.device, args.backend)

    if args.quantize:
        apply_quantization(
            bits=8 if args.quantize == "int8" else 4,
            group_size=args.quant_group_size,
            mode=args.quant_mode,
            fused=not args.no_fused_quant,
        )

    if get_tp_size() > 1 and get_tp_rank() != 0:
        print(f"Rank {get_tp_rank()} waiting for generation tasks...")
        worker_loop()
    else:
        if args.backend == "fast_dense" and get_tp_size() == 1:
            print(f"Starting batch collector + executor (max_size={BATCH_MAX_SIZE}, "
                  f"wait={BATCH_WAIT_S}s, poll={BATCH_POLL_INTERVAL_S}s)...")
            threading.Thread(target=_batch_collector, daemon=True).start()
            threading.Thread(target=_batch_executor, daemon=True).start()
        uvicorn.run(app, host=args.host, port=args.port)


def worker_loop():
    import torch.distributed as dist
    while True:
        objs = [None]
        dist.broadcast_object_list(objs, src=0)
        req = objs[0]
        if req is None:
            continue
        
        if req["type"] == "generate_cached":
            from model_update.generate import generate_cached
            generate_cached(
                MODEL,
                req["input_ids"].to(DEVICE),
                gen_length=req["gen_length"],
                steps=req["steps"],
                block_length=req["block_length"],
                temperature=req["temperature"],
            )



if __name__ == "__main__":
    main()
