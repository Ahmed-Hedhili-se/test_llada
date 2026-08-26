<h1 align="center"><img src="assets/logo/dminfr-mark-inline.svg" alt="" height="60" hspace="8">DMInfr</h1>

Self-contained PyTorch reimplementation of [inclusionAI/LLaDA-MoE-7B-A1B-Instruct](https://huggingface.co/inclusionAI/LLaDA-MoE-7B-A1B-Instruct), a masked-diffusion LLM. Triton fused MoE, block-wise KV caching, fused RMSNorm/decode kernels, variable-length batching, an OpenAI-compatible server, a data-parallel router, and an end-to-end-aware autotuner.

**8.70× single-request and 103× total-pipeline throughput vs the unoptimized baseline** (single RTX A6000).

> The mark is an 8×10 token lattice shaped as a **D**: masked in the upper left, resolved in the lower right, with four tokens stepping violet→blue through the counter — this engine's own decoding process, drawn. Full identity in [`assets/logo/`](assets/logo/README.md).

---

## Contents

- [Architecture](#architecture) · [Optimizations](#optimizations) · [Benchmarks](#benchmarks)
- [Quantization](#quantization-int8-experts) · [Autotuner](#triton-autotuner) · [Correctness](#correctness)
- [Decoding](#adaptive-decoding) · [Multi-GPU](#multi-gpu) · [Getting Started](#getting-started) · [Structure](#project-structure)

---

## Architecture

Generation is iterative unmasking, not next-token prediction. Every forward pass sees the entire sequence; `[MASK]` tokens are progressively replaced over N denoising steps, highest-confidence first.

| | |
|---|---|
| Layers / hidden / heads | 16 (all MoE) / 2048 / 16 (MHA, head dim 128) |
| Experts | 64, top-8, inner dim 1024 |
| Vocabulary | 157,184 |
| Attention | **Bidirectional** (non-causal), RoPE θ=50,000 |

Each expert is a SwiGLU FFN: `W1 = [gate ; up]` (2048 → 2×1024), `SiLU(gate) ⊙ up`, then `W2` (1024 → 2048). Expert weights are ~92% of the model.

> Static top-8 throughout. Reduced expert activation (top-5, step ramps, adaptive thresholds) was evaluated and dropped — this checkpoint's router is near-uniform, so fewer experts cost accuracy without a compensating win.

---

## Optimizations

`src/` is the unoptimized reference. `model_update/` is the optimized engine.

**1. Triton fused MoE** — the baseline loops all 64 experts sequentially. One grouped-GEMM kernel replaces 64 launches: tokens are sorted by expert, padded to block boundaries, processed together. `moe_align_block_size` is fully vectorized (the original made 128 host-device syncs per call).

**2. Block-wise KV caching** — prompt and finalized blocks have fixed content, so their post-RoPE K/V never change. Each step computes only the active block against `[cached ; fresh]`. A block's K/V is committed only after it is fully unmasked, and priming runs over the *full* mask-filled sequence — the model was trained to always see one, and truncating it causes premature EOS collapse.

**3. Fused SiLU epilogue** — `w1` packs `[gate ; up]` along N, so each program computes two B tiles and applies `SiLU(gate)*up` in registers, never materializing the 2·EI-wide intermediate (~1.2 GB of round-trip traffic per forward). **Bit-exact**: accumulators are rounded to bf16 *before* the activation, reproducing the unfused op order exactly.

**4. Narrowed `lm_head`** — `lm_head` is the widest GEMM (2048 → 157,184, ~644 MB of weights) and callers discarded most of its output: the cache-prime and block-finalize passes discard *all* of it. `num_logits` restricts it to the rows actually consumed. **Bit-exact** (GEMM rows are independent; RMSNorm reduces along the feature axis).

**5. Fused RMSNorm + decode tail** — eager RMSNorm is 8 launches, and the model runs 65 per forward (~30% of all kernel launches). The decode tail did a full fp32 copy → 157k-wide softmax → gather of one value. One Triton kernel each: **9.1–11.4×** and **3.1–6.3×** respectively. Not bit-exact (both reassociate a reduction); disable with `LLADA_FUSE_RMSNORM=0` / `LLADA_FUSE_DECODE=0`.

**6. Variable-length batching** — prompts of different lengths are left-padded to a common width, with per-row RoPE positions and an additive attention mask. Before this, requests could only batch if their prompts tokenized to *exactly* the same length.

**Rejected: fused QKV.** One GEMM instead of three measured 1.62× at M=32, 0.99× at M=256, **0.96× at M=1024**, 1.17× at M=2048. This engine runs at M = batch × suffix_length, so batch 32 at block 32 sits near M=1024 — exactly where it loses. `eval/test_fusions.py` still measures it, since the answer is cuBLAS- and shape-dependent.

---

## Benchmarks

### Single request vs baseline — RTX A6000

```bash
python -m eval.check_time_inference --weight-dir weights \
    --gen-length 128 --steps 128 --block-length 32 --mode both --num-runs 3
```

| | Time | Tok/s | Speedup |
|---|---:|---:|---:|
| Baseline (`src/`, unfused, no cache) | 46.67 s | 2.74 | 1.00× |
| **Optimized** | **5.37 s** | **23.85** | **8.70×** |

> The ratio is **not hardware-portable**. `src/`'s MoE loops 64 experts in Python, so the baseline is CPU-dispatch-bound while the optimized path is GPU-bound. An earlier measurement on a different box (A40-24Q) gave 6.46× with a 29.69 s baseline — the difference is mostly the host CPU, not the engine.

### Total pipeline: 103×

Both arms through the same HTTP client and load harness, so this is a deployment-to-deployment number:

| Pipeline | Tok/s |
|---|---:|
| `src/` over HTTP, serialized | **2.7** |
| `model_update/`, `BATCH_MAX_SIZE=32`, concurrency 32 | **278.3** |
| | **103×** |

It decomposes as **8.70× engine × 11.67× batching = 101.5**, within 1.5% of the direct ratio — the check that makes the figure trustworthy rather than a headline. The baseline is single-request by construction: `src/generate.py` hardcodes the batch dimension to 1.

### Batched throughput — A6000, `BATCH_MAX_SIZE=32`, concurrency 32

| | Tok/s |
|---|---:|
| Fixed prompts, fusions off | 232.1 |
| **Fixed prompts, fusions on** | **278.3 (+19.7%)** |
| **Varied prompts** (8 prompts, 6 lengths) | **240.0** |

The fusion gain is a **batched** gain; the same fusions on single-request GSM8K measure ~2%, because MoE weight streaming dominates at batch 1.

> Figures marked *fixed prompts* send an identical prompt for every request. Before variable-length batching that was the only way to fill a batch — with varied lengths the queue fragmented into batches of ~4. Treat fixed-prompt numbers as a ceiling, not as expected traffic.

### Throughput vs batch size

| `BATCH_MAX_SIZE` | Tok/s | Δ throughput | Δ batch | p50 |
|---:|---:|---:|---:|---:|
| 8 | 150.3 | — | — | 6.77 s |
| 16 | 204.8 | +36.3% | +100% | 9.93 s |
| 24 | 224.8 | +9.8% | +50% | 13.56 s |
| **32** | **243.2** | +8.2% | +33% | 16.71 s |
| 48 | **0/96 succeeded** | — | — | — |

Throughput is past its knee by 32: 4× the batch buys 1.62× the throughput. The default of 64 was tuned on a 48 GB A6000 and **kills a 24 GB card** — set it per deployment. Rough sizing per sequence at length `L`, block `b`: KV `128 KB × L`, MoE transients `~48 KB × L`, logits `307 KB × b`.

### Kernel profile

`fused_moe_kernel` is **58.75% of GPU time** at **81% of theoretical weight-streaming bandwidth** — the expert weights (805 MB/layer) stream every forward regardless of token count. There is no kernel headroom left there; the levers are batching, quantization, or faster memory.

> Nsight's "Memory Throughput ~96%" is **L2**, not DRAM (DRAM sits at 66–68%). Read as a DRAM ceiling it says "nothing to gain"; read as an L2 ceiling it says "remove intermediate traffic" — which is what produced optimization 3.

---

## Quantization (INT8 experts)

**This engine runs BF16.** Nothing in `model_update/` quantizes anything — the `use_fp8_w8a8` / `use_int8_w8a16` parameters in `fused_moe_kernel` are inherited from vLLM, always `False`, never referenced.

INT8 is available through the optional [`LLaDA_Quant`](https://github.com/Ahmed-Hedhili-se/LLaDA_Quant) toolkit, imported lazily and absent from `requirements.txt`:

```bash
bash start.sh --backend fast_dense --weight-dir weights --quantize int8
```

`MEASURED` on one A6000, 128-token generation:

| Load | BF16 | INT8 PACKED | INT8 PACKED + fused W8A16 |
|---|---:|---:|---:|
| GSM8K, per question (anchored) | 7.2 s | — | **5.8 s (1.24×)** |
| Single request | 11.88 s | 60.57 s (0.20×) | **10.71 s (1.11×)** |
| Concurrency 32 | 230.5 tok/s | — | 227.6 tok/s (0.99×) |
| Concurrency 64 | 230.8 tok/s | 98.3 (0.43×) | 114.4 tok/s (0.50×) |
| Resident | 13.70 GiB | 7.89 GiB | **7.89 GiB (0.58×)** |

Memory drops 42% at every load; speed **inverts with batch size** — faster than BF16 at batch 1, parity at 32, half the speed at 64. The concurrency-64 collapse is not explained by the roofline (BF16 is flat from 32→64 while the quantized arm halves) and remains open.

**Recommendation:** on a card that fits the model, stay BF16 for throughput. Quantization earns its place when memory is the binding constraint, or for single-request latency.

---

## Triton Autotuner

`tuning_fused_moe_triton.py` generates a hardware-specific `moe_tune_config.json`, loaded at import time. **Run it on every new GPU** — it was the single largest speed win found in this project (2.2× on the MoE pipeline at M=2048).

```bash
python tuning_fused_moe_triton.py --model FULL_CFG          # single GPU
python tuning_fused_moe_triton.py --model FULL_CFG --tp-size 8
```

It scores the *complete* pipeline (`GEMM1 → activation → GEMM2 → sum`, following `FUSE_SILU`), not one GEMM, and penalizes configs that inflate padding: `score = latency × (1 + penalty × padding_ratio)`. Shared-memory limits are queried from the device. The winner is validated at `cos_sim > 0.999` — **not** bit-exactness, so retuning can legitimately shift generated tokens.

---

## Correctness

Against the HuggingFace reference at the logit level: **3,219/3,219 weights mapped**, logit cosine **0.9781**, top-1 token match **91.0%**.

A residual ~6pt fixed-schedule gap vs HF was investigated and closed as **inherent, not a bug**: this checkpoint's router is near-uniform (top-1 weight ~1.7–5%), so bf16-level noise flips top-8 expert membership for 43–90% of positions per layer. A 2×2 kernel-isolation matrix exonerated the Triton MoE kernel entirely. See `INVESTIGATION_LOG.md` §2.9–2.11.

Regression tests (`python -m eval.<name>`):

| | |
|---|---|
| `test_fusions` | RMSNorm / decode / QKV numerics + speed, and token identity |
| `test_variable_length_batch` | a padded row reproduces its solo run **exactly** |
| `test_num_logits_slice` | narrowed `lm_head` is bit-exact |
| `test_moe_align_block_size` | vectorized alignment vs a frozen reference |
| `test_select_transfer_indices` | vectorized per-row top-k vs a frozen reference |
| `test_router` | dispatch, health, failover, recovery (no GPU needed) |

> **Accuracy figures predating `_stable_subset` are not comparable across machines.** `random.shuffle` over a HuggingFace dataset whose row order varies by version meant the same `--seed` selected *different questions* on different boxes — two runs shared only 6 of their first 10. Fixed by sorting on a content hash before shuffling. Re-measure before comparing any historical accuracy number.

---

## Adaptive Decoding

Opt-in threshold decoding (`confidence_threshold`), ported from dInfer's `HierarchyDecoder`: at most one reveal per contiguous run of selectable positions, floored by `low_confidence_threshold`, unioned with positions clearing the threshold outright.

GSM8K, n=50, seed 42, chat-templated:

`ANCHORED` — A6000, GSM8K n=50 seed=42, `max_tokens=1024 steps=512 block_length=64`, threshold 0.9/low 0.4, under the fixed question selection:

| Config | Accuracy | s/question |
|---|---:|---:|
| **BF16** | **74.0%** (37/50) | **7.2** |
| **INT8 + fused W8A16** | **76.0%** (38/50) | **5.8** |

Same 50 questions, same box — the first genuinely comparable accuracy pair in this project. INT8 is 19% faster *and* one question ahead, which at n=50 is noise on the accuracy axis and a real win on the time axis.

> `HISTORICAL, NOT REPRODUCIBLE AS STATED` — earlier runs reported 88.0% (threshold 0.9/0.4) against an HF reference at 82.0%, and 76.0% for the fixed schedule. Those predate the `_stable_subset` fix, so `--seed 42` selected a *different* 50 questions on a different machine. They are not wrong for the set they saw, and they are not comparable to the table above. Re-run before quoting.

> **`steps_per_block >= block_length / 2` is mandatory when a threshold is set.** At ratio 0.25 generation collapses into degenerate repetition (0% accuracy); at 0.5 it recovers fully. A runtime warning fires below the guideline. A `remask_threshold` mechanism was tried for the residual repetition and made things categorically worse (0%) — reverted.

> `_grade_gsm8k` falls back to "last number anywhere in the response" when the extracted span has no digits; 3 of 44 correct answers in one run rest on that fallback. Comparisons *within* this harness hold; the margin over the paper's 82.41% is softer than it looks.

---

## Multi-GPU

**Use data parallelism, not tensor parallelism, for throughput.** `src/server.py` disables request batching whenever `tp_size > 1` — above one rank every request serializes through `request_lock`, so TP=8 delivers roughly an eighth of what the hardware can do. The model is ~14 GiB against an 80 GiB H100, so there is no capacity reason to shard it either.

```bash
bash start_dp.sh --gpus 8 --weight-dir weights          # one replica per GPU + router
bash start_dp.sh --gpus 8 --quantize int8
```

`src/router.py` routes **least-outstanding**, not round-robin: a replica runs a whole batch to completion, so one that just accepted 32 requests is busy for ~35 s while an idle one answers immediately. Unhealthy replicas leave rotation and are retried in the background. `GET /v1/replicas` shows live per-replica load.

TP+EP remains available (`start.sh --tp-size N`) and is the right choice for **single-request latency** only: 2× A6000 measured 6.15× vs a single-GPU baseline at 128 tokens.

---

## Getting Started

```bash
bash setup.sh                                        # venv + torch 2.5.1 + ~15 GB weights
python tuning_fused_moe_triton.py --model FULL_CFG   # do not skip
bash start.sh --backend fast_dense --weight-dir weights
```

Requires an NVIDIA GPU with Triton, ≥24 GB VRAM (bf16), CUDA 12.x, `transformers==4.53.2` (5.x removed `ROPE_INIT_FUNCTIONS['default']`, which this model needs).

```bash
# benchmark
python -m eval.check_time_inference --weight-dir weights --mode both

# throughput
python -m eval.throughput.run_throughput --base-url http://localhost:8000 \
    --concurrency 32 --n-requests 96 --max-tokens 128 --steps 128 --block-length 32

# accuracy
python -m eval.correctness.run_math_reasoning_code --task gsm8k \
    --limit 50 --seed 42 --max-tokens 1024 --steps 512 --block-length 64 \
    --confidence-threshold 0.9 --low-confidence-threshold 0.4
```

The server is OpenAI-compatible (`/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/v1/quantization`).

---

## Project Structure

```
src/                        reference implementation + server + DP router
  model.py  generate.py     unoptimized baseline (batch dimension hardcoded to 1)
  server.py                 OpenAI-compatible API, request batching, --quantize
  router.py                 data-parallel router (least-outstanding)
model_update/               the optimized engine
  model.py                  KV cache, RoPE, attention, fused MoE block
  generate.py               block-wise cached diffusion decoding
  fused_moe_triton.py       grouped-GEMM MoE kernel + SiLU epilogue + autotune loader
  fused_ops.py              fused RMSNorm and decode tail
  distributed.py            TP/EP helpers and weight loading
eval/                       benchmarks, regression tests, diagnostics
  correctness/              GSM8K / BBH / CRUX-O / MMLU-Pro harnesses
  throughput/               concurrent-request benchmark
tuning_fused_moe_triton.py  end-to-end-aware MoE autotuner
start.sh  start_dp.sh       single-GPU and data-parallel launchers
```

---

## Dependencies

`torch>=2.0` (tested 2.5.1), `triton>=2.1`, `transformers==4.53.2`, `safetensors`, `fastapi`, `uvicorn`, `aiohttp`, `lm-eval[api]>=0.4.4`. Full list in `requirements.txt`. `LLaDA_Quant` is optional and deliberately not listed.
