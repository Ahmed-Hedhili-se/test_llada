<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo/dminfr-lockup.svg">
    <img src="assets/logo/dminfr-lockup-light.svg" alt="DMInfr" width="340">
  </picture>
</p>

Self-contained PyTorch reimplementation of [inclusionAI/LLaDA-MoE-7B-A1B-Instruct](https://huggingface.co/inclusionAI/LLaDA-MoE-7B-A1B-Instruct), a masked-diffusion LLM. Triton fused MoE, block-wise KV caching, fused RMSNorm/decode kernels, variable-length batching, an OpenAI-compatible server, a data-parallel router, and an end-to-end-aware autotuner.

**~955 tok/s on 2× H100 PCIe** · **10.2× single-request and ~179× single-GPU total-pipeline throughput** vs the unoptimized baseline.

Measured on 2× H100 PCIe (full log: [`h100x2_bench.md`](h100x2_bench.md)) and on a single RTX A6000. Ratios are **not hardware-portable** — the baseline is CPU-dispatch-bound, so the same engine scores 8.70× on A6000 and 10.2× on H100 with no code difference.

> The mark is an 8×10 token lattice shaped as a **D**: masked in the upper left, resolved in the lower right, with four tokens stepping violet→blue through the counter — this engine's own decoding process, drawn. Full identity in [`assets/logo/`](assets/logo/README.md).

---

## Contents

- [Architecture](#architecture) · [Optimizations](#optimizations) · [Benchmarks](#benchmarks)
- [Quantization](#quantization-int8-experts) · [Autotuner](#triton-autotuner) · [Correctness](#correctness)
- [Decoding](#adaptive-decoding) · [Multi-GPU](#multi-gpu) · [Getting Started](#getting-started) · [Structure](#project-structure)
- **[`h100x2_bench.md`](h100x2_bench.md)** — full 2× H100 validation log: every measurement, the bugs found, and the hypotheses that were tested and rejected

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

`src/` is the unoptimized reference. `dminfr/engine/` is the optimized engine.

**1. Triton fused MoE** — the baseline loops all 64 experts sequentially. One grouped-GEMM kernel replaces 64 launches: tokens are sorted by expert, padded to block boundaries, processed together. `moe_align_block_size` is fully vectorized (the original made 128 host-device syncs per call).

**2. Block-wise KV caching** — prompt and finalized blocks have fixed content, so their post-RoPE K/V never change. Each step computes only the active block against `[cached ; fresh]`. A block's K/V is committed only after it is fully unmasked, and priming runs over the *full* mask-filled sequence — the model was trained to always see one, and truncating it causes premature EOS collapse.

**3. Fused SiLU epilogue** — `w1` packs `[gate ; up]` along N, so each program computes two B tiles and applies `SiLU(gate)*up` in registers, never materializing the 2·EI-wide intermediate (~1.2 GB of round-trip traffic per forward). **Bit-exact**: accumulators are rounded to bf16 *before* the activation, reproducing the unfused op order exactly.

**4. Narrowed `lm_head`** — `lm_head` is the widest GEMM (2048 → 157,184, ~644 MB of weights) and callers discarded most of its output: the cache-prime and block-finalize passes discard *all* of it. `num_logits` restricts it to the rows actually consumed. **Bit-exact** (GEMM rows are independent; RMSNorm reduces along the feature axis).

**5. Fused RMSNorm + decode tail** — eager RMSNorm is 8 launches, and the model runs 65 per forward (~30% of all kernel launches). The decode tail did a full fp32 copy → 157k-wide softmax → gather of one value. One Triton kernel each: **9.1–11.4×** and **3.1–6.3×** respectively. Not bit-exact (both reassociate a reduction); disable with `LLADA_FUSE_RMSNORM=0` / `LLADA_FUSE_DECODE=0`.

**6. Fused RoPE** — `rotate_half` was `torch.cat([-x2, x1])`, materializing a
full rotated copy of q and k every layer, every step, purely to express a
permutation of the head dimension. ncu attributed ~4% of GPU time to the
resulting `aten::cat` + `aten::neg`. One kernel reads the partner element
directly and never builds the copy: **2.55×** on the op, and **bit-exact** —
it reproduces ATen's three-bf16-op rounding order rather than taking the more
accurate fully-fp32 route, so q/k are identical downstream and accuracy is
provably unchanged. Disable with `LLADA_FUSE_ROPE=0`.

**7. Variable-length batching** — prompts of different lengths are left-padded to a common width, with per-row RoPE positions and an additive attention mask. Before this, requests could only batch if their prompts tokenized to *exactly* the same length.

**Rejected on A6000, flips on H100: fused QKV.** One GEMM instead of three measured 1.62× at M=32, 0.99× at M=256, **0.96× at M=1024**, 1.17× at M=2048 on A6000 — and this engine runs at M = batch × suffix_length, so batch 32 at block 32 sits near M=1024, exactly where it lost. On **H100 the same test measures 1.87× / 3.76× / 1.18× / 1.12×** — faster at every size, and bit-exact at M≥256. It remains unfused because `lm_head` dominates the cuBLAS time, so the end-to-end gain is ~0.5%, below the measurement noise floor, and enabling it would need a shape-dependent bit-exactness rule. `tests/test_fusions.py` measures it on whatever card you run: the answer is cuBLAS- and shape-dependent, not universal.

---

## Benchmarks

> Two hardware sets below. **H100 numbers are 3 repetitions per point**; A6000
> numbers are single samples and should be read as approximate — the measured
> run-to-run noise floor on this harness is **~±5%** (up to ±15% at high batch),
> so anything under ~10% is not a result. See `h100x2_bench.md` §11.

### 2× H100 PCIe — current

**Throughput, single GPU** (fused RoPE on, 3 reps/point)

| `BATCH_MAX_SIZE` | Tok/s | sd | p50 |
|---:|---:|---:|---:|
| **32** ← optimum | **656.9** | 3.5 | **5.33 s** |
| 64 | 585.0 | 34.7 | 12.07 s |
| 128 | 607.2 | 54.4 | 23.04 s |

**Throughput, DP=2 across both GPUs**

| Concurrency | Tok/s | sd | p50 |
|---:|---:|---:|---:|
| 32 | 871.6 | 23.9 | **4.18 s** |
| **64** ← peak | **954.5** | 52.8 | 5.48 s |
| 128 | 599.3 | 129.3 | 16.17 s |

Past concurrency 64 throughput *drops* while p50 triples — 64 total (32/replica)
is the operating point, not a floor to push past.

**Speedup** (5 runs, 0/128 token divergence vs baseline)

| | Tok/s | vs baseline |
|---|---:|---:|
| Baseline `src/`, single request | 3.67 | 1.00× |
| Optimized, single request | 37.29 | **10.2×** |
| 1 GPU batched (`BATCH_MAX_SIZE=32`) | 656.9 | **~179×** |
| 2 GPUs (DP=2, concurrency 64) | 954.5 | **~260×** |

Decomposition cross-checks (10.16 × 17.62 = 179.0), but the `src/` baseline
drifted 7.8% between two runs of identical code — **quote a range: ~165–180×
on one GPU, ~225–260× on two**, not a single digit.

### Single request vs baseline — RTX A6000

```bash
python -m benchmarks.check_time_inference --weight-dir weights \
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
| `dminfr/engine/`, `BATCH_MAX_SIZE=32`, concurrency 32 | **278.3** |
| | **103×** |

It decomposes as **8.70× engine × 11.67× batching = 101.5**, within 1.5% of the direct ratio — the check that makes the figure trustworthy rather than a headline. The baseline is single-request by construction: `dminfr/reference/generate.py` hardcodes the batch dimension to 1.

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

`fused_moe_kernel` is the dominant kernel — **58.75% of GPU time on A6000, 55.3% on H100**. Where the bottleneck sits is **hardware-dependent**, so the levers differ per card.

On A6000 it is close to a weight-streaming wall: the expert weights (805 MB/layer) stream every forward regardless of token count.

On H100 it is **not**. Nsight Compute at the production shape measures **DRAM throughput 14.5%, L2 79.7%, compute 49.6%** — ncu names the L2 as the bottleneck outright, and DRAM is nowhere near saturated. An earlier revision of this section claimed "81% of theoretical weight-streaming bandwidth" and "no kernel headroom left"; that is an A6000 figure and does not transfer. See `h100x2_bench.md` §10.

> Nsight's "Memory Throughput ~96%" is **L2**, not DRAM (DRAM sits at 66–68%). Read as a DRAM ceiling it says "nothing to gain"; read as an L2 ceiling it says "remove intermediate traffic" — which is what produced optimization 3.

---

## Quantization (INT8 / FP8 experts)

**This engine runs BF16.** Nothing in `dminfr/engine/` quantizes anything. (The
`use_fp8_w8a8` / `use_int8_w8a16` parameters that used to sit in
`fused_moe_kernel` were inherited from vLLM, always `False`, never referenced —
they have since been removed.)

INT8 is available through the optional [`LLaDA_Quant`](https://github.com/Ahmed-Hedhili-se/LLaDA_Quant) toolkit, imported lazily and absent from `requirements.txt`:

```bash
bash start.sh --backend fast_dense --weight-dir weights --quantize int8
bash start.sh --backend fast_dense --weight-dir weights --quantize fp8   # E4M3
```

### Measured on 2× H100

Resident memory, tensor-level accounting (`LLaDA_Quant.memory.resident_memory` —
**not** `nvidia-smi`, which reports the allocator's peak reserve and made
quantization look like it *grew* the model):

| | Resident | vs BF16 |
|---|---:|---:|
| BF16 | 14032 MiB | — |
| INT8 packed | 8080 MiB | **−42.4%** |
| FP8-E4M3 packed | 8080 MiB | **−42.4%** |

Identical, as they must be — both are 1 byte/weight over the same shapes.

Accuracy, each arm **paired against BF16 on its own identical questions**,
McNemar:

| Arm | n | BF16 on *those* n | quantized | **Δ vs own BF16** | McNemar p |
|---|---:|---:|---:|---:|---:|
| INT8 (packed + fused W8A16) | 200 | 71.0% | 68.5% | **−2.5 pt** | 0.50 |
| FP8-E4M3 (packed, no fused kernel) | 100 | 73.0% | 70.0% | **−3.0 pt** | 0.61 |

> ⚠️ **Do not compare the two `quantized` percentages to each other.** The rows
> use *different question sets* — INT8 was scored on the first 200 of the seeded
> subset, FP8 on the first 100, and the first 100 are easier (visible in the
> BF16 column: 73.0% vs 71.0%). FP8's 70.0% is **not** "better than" INT8's
> 68.5%. Only the Δ column, each row against its own BF16 baseline, is
> meaningful — and by that measure FP8 is marginally *worse*, not better.
>
> FP8's arm stops at n=100 because without a fused kernel it runs ~3× slower per
> question; n=1000 would be ~3.5 h. A direct INT8-vs-FP8 pairing on one shared
> question set has not been run.

**Neither Δ is significant** on its own. But pooling every accuracy
measurement this project has taken of the two formats — including the A6000
historical figure below — separates them:

| | measurements | pattern | read |
|---|---|---|---|
| **INT8** | A6000 n=50 **+2.0**, H100 n=50 **0.0**, H100 n=200 **−2.5** | scatters around zero, across two GPUs | **no measurable accuracy cost** |
| **FP8** | H100 n=50 **−4.0**, H100 n=100 **−3.0** | both negative, stable estimate | small real cost **not excluded** |

Three INT8 samples landing positive, zero and negative is what "no effect"
looks like — the A6000 run that put INT8 *ahead* was noise in the same way our
n=200 run that puts it behind is. FP8's two both point the same way, which is
weak evidence rather than proof, but it is a different pattern.

So the defensible statement is **"INT8 is measurably free; FP8 is probably
slightly lossy but unconfirmed"** — not that the two are interchangeable.
Settling FP8 needs n≈1000, which it cannot reach without a fused kernel.

**On speed the two are not close:** INT8 has a fused W8A16 kernel and runs at
parity with BF16; FP8 has none and runs ~3× slower per question. **INT8 is the
deployable format**; FP8 is a memory play only until it gets a kernel.

### Historical — one A6000, 128-token generation

| Load | BF16 | INT8 PACKED | INT8 PACKED + fused W8A16 |
|---|---:|---:|---:|
| GSM8K, per question (anchored) | 7.2 s | — | **5.8 s (1.24×)** |
| Single request | 11.88 s | 60.57 s (0.20×) | **10.71 s (1.11×)** |
| Concurrency 32 | 230.5 tok/s | — | 227.6 tok/s (0.99×) |
| Concurrency 64 | 230.8 tok/s | 98.3 (0.43×) | 114.4 tok/s (0.50×) |
| Resident | 13.70 GiB | 7.89 GiB | **7.89 GiB (0.58×)** |

Memory drops 42% at every load; speed **inverts with batch size** — faster than BF16 at batch 1, parity at 32, half the speed at 64. The concurrency-64 collapse is not explained by the roofline (BF16 is flat from 32→64 while the quantized arm halves) and remains open.

> ### The A6000 speed win does not transfer to H100 — and the profiling says why
>
> On A6000, `INT8 + fused W8A16` was **19% faster** than BF16 on GSM8K
> (5.8 vs 7.2 s/question). On H100 the same configuration is **24% slower**
> (4.2 vs 3.4 s/question). Same code, opposite sign.
>
> | | DRAM throughput | What INT8 does |
> |---|---:|---|
> | A6000 | 66–68% | close to the bandwidth wall, so halving the weight bytes relieves the **binding** constraint → faster |
> | H100 | **14.5%** | DRAM was never the constraint (L2 is, at 79.7%) → fewer bytes buys nothing, while the W8A16 dequantize work still costs → slower |
>
> **INT8 buys speed only when you are bandwidth-starved.** The A6000 is; the
> H100 is not. This is the same measurement that overturned the "no kernel
> headroom" claim in [Kernel profile](#kernel-profile) — once you know DRAM sits
> at 14.5% on H100, INT8 losing there is the predictable outcome rather than a
> surprise. Memory savings (−42.4%) are hardware-independent and hold on both.

**Recommendation:** on a card that fits the model, stay BF16 for throughput.
Quantization earns its place when memory is the binding constraint, or for
single-request latency. On H100 specifically, using the freed memory to run
*more replicas per GPU* was tested and is a **net loss** — the MoE kernel is
bandwidth-bound, so co-located replicas compete for bytes rather than adding
throughput (`h100x2_bench.md` §8d).

---

## Triton Autotuner

`dminfr.tuning.autotune_moe.py` generates a hardware-specific `moe_tune_config.json`, loaded at import time. **Run it on every new GPU** — it was the single largest speed win found in this project (2.2× on the MoE pipeline at M=2048).

```bash
python dminfr.tuning.autotune_moe.py --model FULL_CFG          # single GPU
python dminfr.tuning.autotune_moe.py --model FULL_CFG --tp-size 8
```

It scores the *complete* pipeline (`GEMM1 → activation → GEMM2 → sum`, following `FUSE_SILU`), not one GEMM, and penalizes configs that inflate padding: `score = latency × (1 + penalty × padding_ratio)`. Shared-memory limits are queried from the device. The winner is validated at `cos_sim > 0.999` — **not** bit-exactness, so retuning can legitimately shift generated tokens.

---

## Correctness

**GSM8K on 2× H100: 75.2% at n=1000** (`steps=1024 block_length=64 threshold 0.9/0.4`).
Regression suite **9/9**, `LLaDA_Quant` **276/276**, autotuner validates
`cos_sim = 1.000000` at every M, optimized output is **character-identical** to
the baseline (0/128 divergence).

> **Sample size matters more than anything else here.** The same engine and config
> scores 74.0% at n=50, 71.0% at n=200 and **75.2% at n=1000** — the small
> subsets are simply unrepresentative. Worse, **~17% of GSM8K questions flip
> outcome under *any* numerical perturbation** on this checkpoint (measured: a
> router-precision change with no accuracy effect still churned 167/1000
> questions). So n=200 carries ±3pt of intrinsic noise, marginal comparison at
> that scale is invalid, and **paired McNemar is required** — resolving a 3-point
> effect needs n≈1000. Every accuracy figure below n=1000 in this repo's history
> inherits this. See `h100x2_bench.md` §9.

Against the HuggingFace reference at the logit level: **3,219/3,219 weights mapped**, logit cosine **0.9781**, top-1 token match **91.0%**.

A residual ~6pt fixed-schedule gap vs HF was investigated and closed as **inherent, not a bug**: this checkpoint's router is near-uniform (top-1 weight ~1.7–5%), so bf16-level noise flips top-8 expert membership for 43–90% of positions per layer. A 2×2 kernel-isolation matrix exonerated the Triton MoE kernel entirely. See `INVESTIGATION_LOG.md` §2.9–2.11.

Regression tests (`python -m tests.<name>`):

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

> **`steps_per_block >= block_length / 2` is the floor, not the recommended setting.** At ratio 0.25 generation collapses into degenerate repetition (0% accuracy); at 0.5 it recovers fully and a runtime warning fires below that. But **0.5 is still on the edge**: measured at n=200 on H100, ratio 0.5 produced 5 deterministic degenerate-repetition failures (2.5% of questions) that **ratio 1.0 eliminates entirely**, worth +3.5pt accuracy for +6% time — the step budget is a ceiling, not a floor, so early exit absorbs most of the extra. **Prefer `steps_per_block >= block_length`.** See `h100x2_bench.md` §8c. A `remask_threshold` mechanism was tried for the residual repetition and made things categorically worse (0%) — reverted.

> `_grade_gsm8k` falls back to "last number anywhere in the response" when the extracted span has no digits; 3 of 44 correct answers in one run rest on that fallback. Comparisons *within* this harness hold; the margin over the paper's 82.41% is softer than it looks.

---

## Multi-GPU

**Use data parallelism, not tensor parallelism, for throughput.** `dminfr/serving/server.py` disables request batching whenever `tp_size > 1` — above one rank every request serializes through `request_lock`, so TP=8 delivers roughly an eighth of what the hardware can do. The model is ~14 GiB against an 80 GiB H100, so there is no capacity reason to shard it either.

```bash
bash start_dp.sh --gpus 8 --weight-dir weights          # one replica per GPU + router
bash start_dp.sh --gpus 8 --quantize int8
```

`dminfr/serving/router.py` routes **least-outstanding**, not round-robin: a replica runs a whole batch to completion, so one that just accepted 32 requests is busy for ~35 s while an idle one answers immediately. Unhealthy replicas leave rotation and are retried in the background. `GET /v1/replicas` shows live per-replica load.

TP+EP remains available (`start.sh --tp-size N`), but **do not reach for it on latency grounds without measuring on your own hardware.** The 6.15× figure previously quoted here was 2× A6000 against the *unoptimized* baseline, not against `dminfr/engine/` on one GPU. Measured on 2× H100 PCIe (no NVLink):

- At concurrency 1 with 128-token generation, TP+EP=2 and a single replica are **statistically tied** (24.8 vs 23.3 tok/s).
- At realistic GSM8K length it **inverts**: 19.4 s/question against a single GPU's 3.4–3.8 s, because TP's per-layer NCCL all-reduce cost is paid *per diffusion step* and GSM8K runs far more steps than the throughput benchmark does.

On an NVLink'd node the picture may differ; on PCIe it does not favour TP. See `h100x2_bench.md` §7, §8e.

---

## Getting Started

```bash
bash setup.sh                                        # venv + torch 2.5.1 + ~15 GB weights
python dminfr.tuning.autotune_moe.py --model FULL_CFG   # do not skip
bash start.sh --backend fast_dense --weight-dir weights
```

Requires an NVIDIA GPU with Triton, ≥24 GB VRAM (bf16), CUDA 12.x, `transformers==4.53.2` (5.x removed `ROPE_INIT_FUNCTIONS['default']`, which this model needs).

```bash
# benchmark
python -m benchmarks.check_time_inference --weight-dir weights --mode both

# throughput
python -m benchmarks.throughput.run_throughput --base-url http://localhost:8000 \
    --concurrency 32 --n-requests 96 --max-tokens 128 --steps 128 --block-length 32

# accuracy
python -m benchmarks.correctness.run_math_reasoning_code --task gsm8k \
    --limit 50 --seed 42 --max-tokens 1024 --steps 512 --block-length 64 \
    --confidence-threshold 0.9 --low-confidence-threshold 0.4
```

The server is OpenAI-compatible (`/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/v1/quantization`).

---

## Project Structure

```
dminfr/                     the package
  engine/                   the optimized engine — what runs in production
    model.py                KV cache, RoPE, attention, fused MoE block
    generate.py             block-wise cached diffusion decoding
    fused_moe_triton.py     grouped-GEMM MoE kernel + SiLU epilogue + autotune loader
    fused_ops.py            fused RMSNorm, decode tail, RoPE
    distributed.py          TP/EP helpers and weight loading
  serving/                  how it is served
    server.py               OpenAI-compatible API, request batching, --quantize
    router.py               data-parallel router (least-outstanding)
  reference/                the unoptimized baseline every speedup is measured
    model.py  generate.py   against. Loops 64 experts in Python; not for deployment.
  tuning/
    autotune_moe.py         end-to-end-aware MoE autotuner

tests/                      regression suite (python -m tests.<name>)
benchmarks/                 latency, throughput and accuracy harnesses
  correctness/              GSM8K / BBH / CRUX-O / MMLU-Pro
  throughput/               concurrent-request benchmark
scripts/                    setup.sh, start.sh, start_dp.sh
tools/                      download_weights.py
docs/                       INVESTIGATION_LOG.md, h100x2_bench.md
archive/investigations/     one-off scripts behind INVESTIGATION_LOG's findings
```

The `engine` / `reference` split was previously `model_update/` vs `src/` —
which also left the production server sitting inside the directory named after
the slow reference path. `dminfr.engine` is the engine; `dminfr.reference` is
the thing it is faster *than*.

---

## Dependencies

`torch>=2.0` (tested 2.5.1), `triton>=2.1`, `transformers==4.53.2`, `safetensors`, `fastapi`, `uvicorn`, `aiohttp`, `lm-eval[api]>=0.4.4`. Full list in `requirements.txt`. `LLaDA_Quant` is optional and deliberately not listed.
