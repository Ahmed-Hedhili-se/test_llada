# LLaDA-MoE-7B-A1B-Instruct — Inference Engine

Self-contained PyTorch reimplementation of [inclusionAI/LLaDA-MoE-7B-A1B-Instruct](https://huggingface.co/inclusionAI/LLaDA-MoE-7B-A1B-Instruct) with an OpenAI-compatible server and eval suite. Tested on B300 SXM6 (SM10.3, torch 2.12.1+cu130).

## Architecture

- 16 layers, all MoE — hidden 2048, 16 heads (MHA), head\_dim 128
- 64 experts/layer, top-8 routing, expert inner dim 1024
- Bidirectional attention (non-causal) — diffusion LM, not autoregressive
- Generation: iterative masked diffusion — start with all `[MASK]` tokens, denoise over N steps
- vocab 157184, RoPE θ=50000, QK RMSNorm per head
- Requires `transformers==4.53.2` (5.x removed `ROPE_INIT_FUNCTIONS['default']`)

## Verified results

- Weight mapping: 3219/3219 (100%)
- Logit cosine vs HF reference: **avg 0.9706** across 256 masked positions
- Top-1 token match: **91.4%** (234/256)
- Generations match HF on factual, code, math prompts

## Usage

```bash
bash setup.sh                                   # create .venv, install deps, download weights
bash setup.sh --skip-weights --weight-dir /path # skip download if weights already present
bash start.sh --weight-dir ./weights            # start server on :8000
.venv/bin/python -m eval.check_server           # smoke test
.venv/bin/python compare_models.py --weight-dir ./weights   # verify vs HF
.venv/bin/python -m eval.correctness.run_correctness        # GSM8K-CoT (200 problems)
.venv/bin/python -m eval.throughput.run_throughput          # throughput benchmark
```

## Files

```
src/model.py       — model implementation
src/model_small.py — ~195M scaled-down variant (random weights, same architecture)
src/generate.py    — masked diffusion decode loop
src/server.py      — OpenAI-compatible chat completions server
compare_models.py  — logit + generation comparison vs HF reference
download_weights.py
setup.sh / start.sh
eval/check_server.py
eval/correctness/run_correctness.py
eval/throughput/run_throughput.py
```