# LLaDA-MoE-7B-A1B-Instruct — Self-Contained Inference Engine

A clean, self-contained PyTorch reimplementation of [inclusionAI/LLaDA-MoE-7B-A1B-Instruct](https://huggingface.co/inclusionAI/LLaDA-MoE-7B-A1B-Instruct) with an OpenAI-compatible server and evaluation suite.

## Architecture

LLaDA-MoE is a **Mixture-of-Experts Diffusion Language Model** — fundamentally different from autoregressive models:

| Parameter | Value |
|---|---|
| Layers | 16 |
| Hidden size | 2048 |
| Attention heads | 16 (head_dim=128, MHA) |
| Experts per layer | 64 (all layers are MoE) |
| Active experts | 8 (top-8 routing) |
| Expert inner dim | 1024 |
| Vocab size | 157,184 |
| RoPE theta | 50,000 |
| QK LayerNorm | Yes (per-head RMSNorm) |
| Attention type | **Bidirectional** (non-causal) |
| Generation | **Masked diffusion** (not autoregressive) |

### Key difference from AR models

Generation starts with all response tokens replaced by `[MASK]` (token id `156895`). Multiple denoising forward passes iteratively unmask the highest-confidence tokens until all positions are filled.

## Setup

```bash
# On a CUDA machine (B300 / H100 / A100):
bash setup.sh

# Skip weight download if already present:
bash setup.sh --skip-weights --weight-dir /path/to/weights
```

## Start server

```bash
bash start.sh --weight-dir ./weights --port 8000
```

## Check server

```bash
python3 -m eval.check_server --base-url http://localhost:8000
```

## Compare vs HF model

```bash
python3 compare_models.py --weight-dir ./weights
```

## Correctness eval (GSM8K-CoT)

```bash
# Start server first, then:
python3 -m eval.correctness.run_correctness --base-url http://localhost:8000 --limit 200
```

## Throughput benchmark

```bash
python3 -m eval.throughput.run_throughput --base-url http://localhost:8000
```

## Files

```
src/
  model.py      — self-contained LLaDA-MoE PyTorch implementation
  generate.py   — masked diffusion generation algorithm
  server.py     — OpenAI-compatible FastAPI server
eval/
  check_server.py              — health check + smoke test
  correctness/run_correctness.py — GSM8K-CoT eval via lm-eval
  throughput/run_throughput.py   — async throughput benchmark
compare_models.py  — logit + generation comparison vs HF reference
download_weights.py — HuggingFace weight download (always from HF directly)
setup.sh           — one-shot environment setup
start.sh           — server launcher
```