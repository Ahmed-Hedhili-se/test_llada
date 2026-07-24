# LLaDA-MoE-7B-A1B-Instruct — Optimized Inference Engine

Self-contained PyTorch reimplementation of [inclusionAI/LLaDA-MoE-7B-A1B-Instruct](https://huggingface.co/inclusionAI/LLaDA-MoE-7B-A1B-Instruct) with three stacked inference optimizations: **Triton Fused MoE**, **Block-wise KV Caching**, and **Reduced Expert Activation (top-k)**. Includes an OpenAI-compatible API server and a full evaluation suite.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Optimizations](#optimizations)
- [Benchmark Results](#benchmark-results)
- [Correctness Verification](#correctness-verification)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Project Structure](#project-structure)

---

## Architecture Overview

LLaDA-MoE is a **Masked Diffusion Language Model** — it generates text by iteratively unmasking `[MASK]` tokens rather than autoregressive next-token prediction.

| Parameter | Value |
|---|---|
| Layers | 16 (all MoE) |
| Hidden dim | 2048 |
| Attention | 16 heads, MHA, head dim 128 |
| Experts per layer | 64 |
| Default active experts (top-k) | 8 |
| Expert inner dim | 1024 |
| Vocabulary | 157,184 |
| Positional encoding | RoPE (θ = 50,000) |
| Attention type | Bidirectional (non-causal) |
| Generation method | Iterative masked diffusion |

> **Key difference from autoregressive LLMs**: Each forward pass sees the *entire* sequence (prompt + masked positions). Generation starts with all target tokens as `[MASK]` and progressively unmasks them over N denoising steps, selecting the highest-confidence predictions at each step.

### Expert Architecture (SwiGLU FFN)

Each of the 64 experts is a small feed-forward network with a **SwiGLU** activation:

```
Input x (dim 2048)
   │
   ├──► W1 = [gate_proj ; up_proj]  (dim 2048 → 2×1024)
   │         │
   │    split into gate, up
   │         │
   │    SiLU(gate) ⊙ up     ← SwiGLU activation
   │         │
   └──► W2 = down_proj       (dim 1024 → 2048)
            │
       Output (dim 2048)
```

The router selects the **top-k** experts per token, runs each through this FFN, and produces a weighted sum of their outputs.

---

## Optimizations

The `model_update/` directory implements three stacked optimizations over the baseline (`src/`):

### 1. Triton Fused MoE (`fused_moe_triton.py`)

**Problem**: The baseline loops over all 64 experts sequentially, calling small matmuls one at a time.

**Solution**: A single Triton grouped-GEMM kernel fuses all active expert computations into one GPU kernel launch. Tokens are sorted by expert assignment, padded to block boundaries, and processed together.

```
Baseline (src/):     for expert in 64_experts: expert(tokens)   → 64 kernel launches
Optimized:           fused_moe_kernel(all_tokens, all_experts)  → 1 kernel launch
```

### 2. Block-wise KV Caching

**Problem**: In masked diffusion, each denoising step re-runs the full model over the entire sequence (prompt + all generated tokens), even though the prompt tokens never change.

**Solution**: Cache the Key/Value tensors from finalized tokens (prompt + already-generated blocks). Each denoising step only computes over the *active block*, attending against `[cached K/V ; fresh K/V]`.

```
Baseline:    model(full_sequence)         every step
Optimized:   model(active_block, cache)   each step only processes the current block
```

> **Correctness rule**: A block's K/V is only pushed to the permanent cache *after* it is fully unmasked. Caching mid-denoising K/V would bake in stale masked-token representations.

### 3. Reduced Expert Activation (top-k = 5)

**Problem**: The default top-8 routing activates 8 of 64 experts per token, but early denoising steps operate on noisy `[MASK]` tokens where full expert capacity is unnecessary.

**Solution**: Use fewer experts (e.g., top-5 instead of top-8) to reduce compute per token by **~37%** with minimal quality loss. Configurable via `--topk` at inference time. A `dynamic_k` mechanism also supports ramping from `min_k` to `base_k` across denoising steps.

---

## Benchmark Results

32-token generation benchmark on CPU (PyTorch 2.11.0+cu130). All three optimizations are stacked in the optimized model.

```
python -m eval.check_time_inference --weight-dir weights --topk <K>
```

| Configuration | top-k | Time (s) | Tok/s | Speedup | Token Divergence |
|---|:---:|---:|---:|:---:|:---:|
| **Baseline** (`src/`, unfused, no cache) | 8 | 30.59 | 1.05 | 1.00× | — |
| **Optimized** (`model_update/`) | 8 | 20.83 | 1.54 | **1.43×** | 9.38% |
| **Optimized** (`model_update/`) | 5 | 15.96 | 2.01 | **1.92×** | 9.38% |
| **Optimized** (`model_update/`) | 4 | 13.95 | 2.29 | **2.11×** | 9.38% |

### Speedup Breakdown

```
            Speedup vs Baseline (32 tokens, CPU)
  
  topk=8  ██████████████░░░░░░░░░░░░░░░░  1.43×   (KV cache + fused MoE only)
  topk=5  ███████████████████░░░░░░░░░░░  1.92×   (+ reduced experts)
  topk=4  █████████████████████░░░░░░░░░  2.11×   (+ further reduction)
  
  ──────────────────────────────────────
  1.0×          1.5×          2.0×     2.5×
```

> **Note**: These benchmarks ran on CPU due to a CUDA driver mismatch on the test server. On GPU, the Triton fused MoE kernel provides **additional** speedup since it replaces 64 sequential expert calls with a single grouped-GEMM kernel — the CPU benchmark cannot benefit from this Triton fusion.

---

## Correctness Verification

Verified against the HuggingFace reference implementation:

| Metric | Result |
|---|---|
| Weight mapping | 3,219 / 3,219 (100%) |
| Logit cosine similarity | avg **0.9706** across 256 masked positions |
| Top-1 token match | **91.4%** (234/256) |
| Generation quality | Matches HF on factual, code, and math prompts |

---

## Getting Started

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (for Triton fused MoE; CPU fallback available)
- ~15 GB disk space for model weights

### Installation

```bash
# Option 1: Automated setup (creates venv, installs deps, downloads weights)
bash setup.sh

# Option 2: Manual setup
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python download_weights.py --dest weights
```

### Quick Start

```bash
# Run the inference time benchmark
python -m eval.check_time_inference --weight-dir weights --topk 5

# Start the OpenAI-compatible API server
bash start.sh --weight-dir ./weights
```

---

## Usage

### Inference Time Benchmark

Compare baseline vs optimized model at different top-k values:

```bash
python -m eval.check_time_inference --weight-dir weights --topk 5    # recommended
python -m eval.check_time_inference --weight-dir weights --topk 8    # full experts
python -m eval.check_time_inference --weight-dir weights --topk 4    # fastest
```

**Options:**

| Flag | Default | Description |
|---|---|---|
| `--weight-dir` | `weights` | Path to model weight files |
| `--topk` | `5` | Number of active experts for optimized model |
| `--gen-length` | `32` | Number of tokens to generate |
| `--steps` | `32` | Number of denoising steps |
| `--block-length` | `16` | Block size for KV caching |
| `--num-warmup` | `1` | Warmup iterations (excluded from timing) |
| `--num-runs` | `3` | Timed benchmark iterations |
| `--device` | `cuda:0` | Device (falls back to `cpu` if CUDA unavailable) |

### API Server

```bash
bash start.sh --weight-dir ./weights --port 8000
```

Compatible with OpenAI chat completions API:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "inclusionAI/LLaDA-MoE-7B-A1B-Instruct",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "max_tokens": 128,
    "temperature": 0.0
  }'
```

### Evaluation Suite

```bash
python -m eval.check_server                     # server smoke test
python -m eval.correctness.run_correctness      # GSM8K-CoT accuracy (200 problems)
python -m eval.throughput.run_throughput         # concurrent request throughput
python compare_models.py --weight-dir ./weights  # logit comparison vs HF reference
```

---

## Project Structure

```
.
├── src/                                    ← Baseline implementation
│   ├── model.py                            Dense model (unfused MoE, no KV cache, top-8)
│   ├── model_small.py                      ~195M scaled-down variant (same architecture)
│   ├── generate.py                         Masked diffusion decode loop
│   └── server.py                           OpenAI-compatible API server
│
├── model_update/                           ← Optimized implementation
│   ├── model.py                            KV-cached model with configurable top-k routing
│   ├── generate.py                         Block-wise KV-cached generation loop
│   └── fused_moe_triton.py                 Triton grouped-GEMM kernel for fused MoE
│
├── eval/                                   ← Evaluation & benchmarks
│   ├── check_time_inference.py             Baseline vs optimized speedup benchmark
│   ├── check_server.py                     Server smoke test
│   ├── diagnose_dynamic_experts.py         Token divergence & routing diagnostic
│   ├── diagnose_real_activation_pruning.py Per-layer routing distribution analysis
│   ├── correctness/
│   │   └── run_correctness.py              GSM8K-CoT accuracy (200 problems)
│   └── throughput/
│       └── run_throughput.py               Concurrent request throughput benchmark
│
├── compare_models.py                       Logit + generation comparison vs HF reference
├── download_weights.py                     Download weights from HuggingFace Hub
├── requirements.txt                        Python dependencies (torch, triton, transformers, ...)
├── setup.sh                                One-shot environment setup script
├── start.sh                                Start inference server
└── README.md
```

---

## Dependencies

Key dependencies (see `requirements.txt` for full list):

| Package | Version | Purpose |
|---|---|---|
| `torch` | ≥ 2.0.0 | Core deep learning framework |
| `triton` | ≥ 2.1.0 | Fused MoE Triton kernel |
| `transformers` | == 4.53.2 | Tokenizer & config loading |
| `safetensors` | latest | Weight file format |
| `fastapi` + `uvicorn` | latest | API server |

> ⚠️ **transformers 5.x is not supported** — it removed `ROPE_INIT_FUNCTIONS['default']` which this model relies on.