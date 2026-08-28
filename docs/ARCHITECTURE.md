# Architecture & Internals

How the model works, what was optimized, and how the tree is laid out.
Benchmark numbers live in the [README](../README.md); the full measurement
record is in [`h100x2_bench.md`](h100x2_bench.md).

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


`dminfr/reference/` is the unoptimized reference. `dminfr/engine/` is the optimized engine.

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

## Triton Autotuner


`dminfr/tuning/autotune_moe.py` generates a hardware-specific `moe_tune_config.json`, loaded at import time. **Run it on every new GPU** — it was the single largest speed win found in this project (2.2× on the MoE pipeline at M=2048).

```bash
python -m dminfr.tuning.autotune_moe --model FULL_CFG          # single GPU
python -m dminfr.tuning.autotune_moe --model FULL_CFG --tp-size 8
```

It scores the *complete* pipeline (`GEMM1 → activation → GEMM2 → sum`, following `FUSE_SILU`), not one GEMM, and penalizes configs that inflate padding: `score = latency × (1 + penalty × padding_ratio)`. Shared-memory limits are queried from the device. The winner is validated at `cos_sim > 0.999` — **not** bit-exactness, so retuning can legitimately shift generated tokens.

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

