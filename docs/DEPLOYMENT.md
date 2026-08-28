# Deployment & Configuration

How to install, launch, and configure the engine — including the decoding
settings that materially affect accuracy, and the multi-GPU topology choice.

Results for every claim here are in the [README](../README.md) and
[`h100x2_bench.md`](h100x2_bench.md).

---

## Getting Started


```bash
bash scripts/setup.sh                                        # venv + torch 2.5.1 + ~15 GB weights
python -m dminfr.tuning.autotune_moe --model FULL_CFG   # do not skip
bash scripts/start.sh --backend fast_dense --weight-dir weights
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

## Multi-GPU


**Use data parallelism, not tensor parallelism, for throughput.** `dminfr/serving/server.py` disables request batching whenever `tp_size > 1` — above one rank every request serializes through `request_lock`, so TP=8 delivers roughly an eighth of what the hardware can do. The model is ~14 GiB against an 80 GiB H100, so there is no capacity reason to shard it either.

```bash
bash scripts/start_dp.sh --gpus 8 --weight-dir weights          # one replica per GPU + router
bash scripts/start_dp.sh --gpus 8 --quantize int8
```

`dminfr/serving/router.py` routes **least-outstanding**, not round-robin: a replica runs a whole batch to completion, so one that just accepted 32 requests is busy for ~35 s while an idle one answers immediately. Unhealthy replicas leave rotation and are retried in the background. `GET /v1/replicas` shows live per-replica load.

TP+EP remains available (`scripts/start.sh --tp-size N`), but **do not reach for it on latency grounds without measuring on your own hardware.** The 6.15× figure previously quoted here was 2× A6000 against the *unoptimized* baseline, not against `dminfr/engine/` on one GPU. Measured on 2× H100 PCIe (no NVLink):

- At concurrency 1 with 128-token generation, TP+EP=2 and a single replica are **statistically tied** (24.8 vs 23.3 tok/s).
- At realistic GSM8K length it **inverts**: 19.4 s/question against a single GPU's 3.4–3.8 s, because TP's per-layer NCCL all-reduce cost is paid *per diffusion step* and GSM8K runs far more steps than the throughput benchmark does.

On an NVLink'd node the picture may differ; on PCIe it does not favour TP. See [`docs/h100x2_bench.md`](h100x2_bench.md) §7, §8e.

---

## Decoding configuration


Opt-in threshold decoding (`confidence_threshold`), ported from dInfer's `HierarchyDecoder`: at most one reveal per contiguous run of selectable positions, floored by `low_confidence_threshold`, unioned with positions clearing the threshold outright.

GSM8K, n=50, seed 42, chat-templated:

`ANCHORED` — A6000, GSM8K n=50 seed=42, `max_tokens=1024 steps=512 block_length=64`, threshold 0.9/low 0.4, under the fixed question selection:

| Config | Accuracy | s/question |
|---|---:|---:|
| **BF16** | **74.0%** (37/50) | **7.2** |
| **INT8 + fused W8A16** | **76.0%** (38/50) | **5.8** |

Same 50 questions, same box — the first genuinely comparable accuracy pair in this project. INT8 is 19% faster *and* one question ahead, which at n=50 is noise on the accuracy axis and a real win on the time axis.

> `HISTORICAL, NOT REPRODUCIBLE AS STATED` — earlier runs reported 88.0% (threshold 0.9/0.4) against an HF reference at 82.0%, and 76.0% for the fixed schedule. Those predate the `_stable_subset` fix, so `--seed 42` selected a *different* 50 questions on a different machine. They are not wrong for the set they saw, and they are not comparable to the table above. Re-run before quoting.

> **`steps_per_block >= block_length / 2` is the floor, not the recommended setting.** At ratio 0.25 generation collapses into degenerate repetition (0% accuracy); at 0.5 it recovers fully and a runtime warning fires below that. But **0.5 is still on the edge**: measured at n=200 on H100, ratio 0.5 produced 5 deterministic degenerate-repetition failures (2.5% of questions) that **ratio 1.0 eliminates entirely**, worth +3.5pt accuracy for +6% time — the step budget is a ceiling, not a floor, so early exit absorbs most of the extra. **Prefer `steps_per_block >= block_length`.** See [`docs/h100x2_bench.md`](h100x2_bench.md) §8c. A `remask_threshold` mechanism was tried for the residual repetition and made things categorically worse (0%) — reverted.

> `_grade_gsm8k` falls back to "last number anywhere in the response" when the extracted span has no digits; 3 of 44 correct answers in one run rest on that fallback. Comparisons *within* this harness hold; the margin over the paper's 82.41% is softer than it looks.

---

## Dependencies


`torch>=2.0` (tested 2.5.1), `triton>=2.1`, `transformers==4.53.2`, `safetensors`, `fastapi`, `uvicorn`, `aiohttp`, `lm-eval[api]>=0.4.4`. Full list in `requirements.txt`. `LLaDA_Quant` is optional and deliberately not listed.
