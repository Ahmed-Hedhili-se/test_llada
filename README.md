# LLaDA-MoE-7B-A1B-Instruct — Inference Engine

Self-contained PyTorch reimplementation of [inclusionAI/LLaDA-MoE-7B-A1B-Instruct](https://huggingface.co/inclusionAI/LLaDA-MoE-7B-A1B-Instruct) with an OpenAI-compatible server and eval suite. Tested on B300 SXM6 (SM10.3, torch 2.12.1+cu130).

## Architecture

- 16 layers, all MoE — hidden 2048, 16 heads (MHA), head\_dim 128
- 64 experts/layer, top-8 routing, expert inner dim 1024
- Bidirectional attention (non-causal) — diffusion LM, not autoregressive
- Generation: iterative masked diffusion — start with all `[MASK]` tokens, denoise over N steps
- vocab 157184, RoPE θ=50000, QK RMSNorm per head
- Requires `transformers==4.53.2` (5.x removed `ROPE_INIT_FUNCTIONS['default']`)

## Optimized Inference (`model_update/`)

Block-wise KV-cached generation with dynamic expert routing:

- **KV caching**: Prompt + finalized blocks are cached once; each denoising step only runs over the active block.
- **Dynamic expert ramping (`min_k`)**: During denoising, the number of active experts ramps from `min_k` at step 0 up to the full `base_k=8` at the final step. Early denoising steps operate on noisy `[MASK]` tokens where fewer experts suffice.
- **Default: `min_k=5`** — 1.14x speedup with 0% token divergence vs dense baseline.

> **Note:** Expert threshold pruning was evaluated and removed — LLaDA-MoE's 64-expert softmax routing produces diffuse gate weights (uniform ≈ 1/64 ≈ 0.016), causing any static threshold (e.g., 0.03) to zero out >65% of expert contributions, particularly in early/late layers.

## Verified results

- Weight mapping: 3219/3219 (100%)
- Logit cosine vs HF reference: **avg 0.9706** across 256 masked positions
- Top-1 token match: **91.4%** (234/256)
- Generations match HF on factual, code, math prompts

### Dynamic Expert Speedup (32-token benchmark)

| Configuration                   | Speedup | Token Divergence |
|---------------------------------|---------|------------------|
| Dense Baseline                  | 1.00x   | —                |
| Cache Only (Block-wise)         | 1.06x   | 0.00%            |
| Cache + Dynamic Experts (min_k=4) | 1.20x | 6.25%            |
| Cache + Dynamic Experts (min_k=5) | **1.14x** | **0.00%**     |
| Cache + Dynamic Experts (min_k=6) | 1.11x | 0.00%            |

## Usage

```bash
bash setup.sh                                   # create .venv, install deps, download weights
bash setup.sh --skip-weights --weight-dir /path # skip download if weights already present
bash start.sh --weight-dir ./weights            # start server on :8000
.venv/bin/python -m eval.check_server           # smoke test
.venv/bin/python compare_models.py --weight-dir ./weights   # verify vs HF
.venv/bin/python -m eval.correctness.run_correctness        # GSM8K-CoT (200 problems)
.venv/bin/python -m eval.throughput.run_throughput          # throughput benchmark
.venv/bin/python eval/check_time_inference.py               # speedup + divergence benchmark
.venv/bin/python eval/diagnose_dynamic_experts.py           # routing diagnostic
```

## Files

```
src/model.py                  — dense model implementation (baseline)
src/model_small.py            — ~195M scaled-down variant (random weights, same architecture)
src/generate.py               — masked diffusion decode loop (baseline)
src/server.py                 — OpenAI-compatible chat completions server

model_update/model.py         — KV-cached model with dynamic expert routing
model_update/generate.py      — block-wise KV-cached generation (min_k ramping)
model_update/kv_cache.py      — standalone KV cache model variant

compare_models.py             — logit + generation comparison vs HF reference
download_weights.py
setup.sh / start.sh

eval/check_server.py                      — server smoke test
eval/check_time_inference.py              — speedup + divergence benchmark
eval/diagnose_dynamic_experts.py          — token divergence + routing weight diagnostic
eval/diagnose_real_activation_pruning.py  — per-layer routing distribution on real weights
eval/correctness/run_correctness.py       — GSM8K-CoT (200 problems)
eval/throughput/run_throughput.py          — throughput benchmark
```