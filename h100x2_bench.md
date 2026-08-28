# 2× H100 PCIe validation log

Full record of provisioning and validating DMInfr end-to-end on a fresh
2× NVIDIA H100 PCIe box, following the order in `README.md`: correctness
gates first, then the autotuner, then latency, then accuracy, then
throughput. One genuine finding surfaced along the way — a reproducible
decoding-collapse failure mode invisible at every `n` the project has
tested before now — documented in full below.

---

## 0. Environment

| | |
|---|---|
| GPUs | 2× NVIDIA H100 PCIe, 80 GB each |
| Interconnect | **PHB** (PCIe host bridge) — no NVLink |
| Driver / CUDA | 580.126.09 / CUDA 13.0 |
| CPU / RAM | 40 cores / 251 GB |
| torch | 2.5.1+cu124 |
| triton | 3.1.0 |
| transformers | 4.53.2 |
| SM capability | sm_90 (9,0), both GPUs |

### `setup.sh` bug found and fixed

`setup.sh` routed any CUDA 13.x driver to the `cu130` wheel index, then
pinned `torch==2.5.1`. **Torch 2.5.1 has no `cu130` build at all** — the
install fails outright on any CUDA-13 machine, which every current H100
rental is. Fixed to route CUDA 13 drivers to `cu124` instead (drivers are
backward-compatible with older runtimes, and cu124 is the newest line 2.5.1
actually ships for). Committed as `ada8a8b`, pushed to both remotes before
provisioning this box.

---

## 1. Correctness gates (`eval.test_*`)

All 9 regression tests, run individually. **9/9 PASS.**

| Test | Result |
|---|---|
| `test_fusions` | PASS — RMSNorm 4.55–12.47× (rel_L2 ≤1.1e-5), decode tail 1.89–5.98× (100% argmax match), QKV 1.12–3.76× (bit-exact at M≥256), end-to-end token identity 95.3–100% |
| `test_fused_silu_epilogue` | PASS — bit-exact vs unfused at every M from 1 to 2048, all `generate_cached` token sequences identical |
| `test_variable_length_batch` | PASS — a padded row reproduces its solo run exactly |
| `test_num_logits_slice` | PASS — CPU and CUDA both bit-exact |
| `test_moe_align_block_size` | PASS — 35 randomized cases + edge cases |
| `test_select_transfer_indices` | PASS — 48 randomized + hierarchy invariants |
| `test_threshold_decoding` | PASS — bounded/complete at threshold 0.0/0.9/1.1 |
| `test_router` | PASS — dispatch, health, failover, recovery |
| `test_oom_retry` | PASS — halves 16→8→4, reassembles identically |

No regressions from the A6000 baseline. Engine is sound on sm_90.

## 2. Autotuner

```
python tuning_fused_moe_triton.py --model FULL_CFG
```

Completed cleanly. `cos_sim = 1.000000` at every M tested (1 through 2048).
`moe_tune_config.json` written to repo root — **not currently tracked in
git** (never has been; see open items).

> **Anomaly noted, not blocking:** every config line reports
> `shmem=NoneB occ=None%`. The tuner's occupancy/shared-memory introspection
> isn't returning data on this box — likely a Triton-internals call that
> changed shape between versions. Didn't affect the `cos_sim` correctness
> check or the winning configs, but it means register/shared-mem pressure
> is currently unmeasured here. Worth a follow-up.

Best configs shifted meaningfully from the A6000 tuning, as expected —
H100's larger `shared_memory_per_block_optin` (~94 KB usable here) and
different SM count change the optimal tile shapes at every M.

## 3. Single-request latency vs baseline

```
python -m eval.check_time_inference --weight-dir weights \
    --gen-length 128 --steps 128 --block-length 32 --mode both --num-runs 3
```

| | Time | Tok/s | Speedup |
|---|---:|---:|---:|
| Baseline (`src/`, unfused, no cache) | 32.38 s | 3.95 | 1.00× |
| **Optimized** (`model_update/`) | **3.60 s** | **35.57** | **9.00×** |

**Token divergence: 0/128 (0.00%)** — optimized output is character-identical
to the baseline. Consistent with the A6000's 8.70× (README explicitly notes
this ratio is not hardware-portable, since the baseline is CPU-dispatch-bound
— 9.00× here is in the same neighborhood, as expected).

## 4. GSM8K correctness

All runs: `--seed 42 --max-tokens 1024 --steps 512 --block-length 64 --confidence-threshold 0.9 --low-confidence-threshold 0.4`, chat-templated, against `bash start.sh --backend fast_dense`.

### 4a. n=50 (clean, isolated)

| | |
|---|---|
| Accuracy | **74.0% (37/50)** |
| Time | 189.8 s total, **3.8 s/question** |

Matches the README's A6000 BF16 anchored figure (74.0%, 37/50) **exactly** —
confirms the `_stable_subset` fix makes this comparable cross-machine, as
designed. Per-question latency is **~1.9× faster than the A6000's 7.2 s.**

### 4b. n=200 and n=300 — first attempt, INVALID (see below), then confirmed clean

Two runs (`n=200`, `n=300`) were launched against the server in close
succession; a manual server restart between them **failed silently** (port
already held, `start.sh` exited 1, but the eval client didn't error because
a server was still listening — just the old one). Both runs ended up
querying the **same server process concurrently**:

| Run | Accuracy | s/question |
|---|---:|---:|
| n=200 (concurrent w/ n=300) | 67.5% (135/200) | 6.2 |
| n=300 (concurrent w/ n=200) | 68.0% (136/200) | ~4.1 |

Initial hypothesis was GPU/server contention corrupting shared state. That
hypothesis was **wrong** — disproven directly, see 4c.

### 4c. n=200, re-run fully isolated (single client, idle server, GPU0 alone)

```
python -m eval.correctness.run_math_reasoning_code --task gsm8k \
    --limit 200 --seed 42 --max-tokens 1024 --steps 512 --block-length 64 \
    --confidence-threshold 0.9 --low-confidence-threshold 0.4
```

| | |
|---|---|
| Accuracy | **67.50% (135/200)** — **exact match** to the contaminated run |
| Time | 666.7 s total, **3.3 s/question** (faster, no contention) |

**Identical result, identical failures, at identical question indices**, with
zero concurrency. This proves the 67.5% figure is real: the earlier
contention slowed the run down but did not change its outcome. The result
generalizes: `n=50` accuracy (74.0%) does not predict `n=200` accuracy
(67.5%) — a 6.5-point gap that was invisible until this run, because no
prior benchmark in this project has evaluated past question 50.

**By 50-question bucket** (clean n=200 run):

| Questions | Correct | Accuracy |
|---|---:|---:|
| 1–50 | 37/50 | 74.0% |
| 51–100 | 32/50 | 64.0% |
| 101–150 | 34/50 | 68.0% |
| 151–200 | 33/50 | 66.0% |

The first bucket is not representative of the rest — it happens to be the
only one anyone had ever run before.

### 4d. Finding: reproducible decoding collapse on 5/200 questions (2.5%)

Five questions — global stable-subset indices **96, 122, 139, 168, 200** —
produce catastrophic degenerate-repetition output instead of an answer:
single tokens or short fragments repeated hundreds of times
(`balloons balloons balloons...`, endless `5`s, `drive drive drive...`,
runs of `+`, `sters sters sters...`), 946 to 7,578 characters long against a
normal response of a few dozen. All five are counted wrong.

This is **fully deterministic**, confirmed two ways:
- Identical garbage text at identical positions in the two *concurrent*
  n=200/n=300 runs (ruling out a race condition — a race would not
  reproduce byte-for-byte).
- Identical garbage, same 5 indices, in the *isolated* re-run (ruling out
  contention entirely).

This is a genuine decoding failure mode, not a concurrency bug, and it was
never visible before because it doesn't occur in the first 50 questions of
the stable-sorted set — the only range this project has ever evaluated.

**Hypothesis, not yet verified:** the run configuration sits exactly on the
README's documented stability boundary. `steps_per_block = steps /
(max_tokens / block_length) = 512 / (1024/64) = 32`; `block_length / 2 =
32`. The ratio is `32/32 = 1.0`× the guideline minimum stated in the
README (`steps_per_block ≥ block_length/2`) — i.e. exactly at the boundary
the README already flags as where "generation collapses into degenerate
repetition" below it. This may be the same failure mode surfacing at the
edge of the safe region rather than comfortably inside it, on specific hard
long-form problems. **Not confirmed** — needs a run at a higher
`steps_per_block` ratio (e.g. `--steps 768`) on the same 5 questions to
test whether it resolves them.

Excluding these 5 purely-mechanical failures changes accuracy only
marginally (135/195 = 69.2%) — the interesting result is qualitative (a
distinct, previously-unknown failure class), not that it explains most of
the n=50→n=200 gap. The rest of the gap looks like ordinary sample
variance, consistent with the README's own note that accuracy deltas at
n=50 are noise-level.

---

## 5. Throughput — batch-size sweep, single GPU

```
BATCH_MAX_SIZE=$B python -m src.server --backend fast_dense --weight-dir weights
python -m eval.throughput.run_throughput --base-url http://localhost:8000 \
    --concurrency $B --n-requests $((B*3)) --max-tokens 128 --steps 128 --block-length 32
```

Server restarted per `BATCH_MAX_SIZE` (read once at startup). Fixed prompts,
GPU 0 only, tuned `moe_tune_config.json` from §2 in effect. The README's own
sweep stops at 32 — the A6000's practical ceiling — so this pushes into
territory the project has never measured.

| `BATCH_MAX_SIZE` | Tok/s | p50 latency | p95 latency |
|---:|---:|---:|---:|
| 32 | 590.2 | 5.93 s | 5.95 s |
| **64** | **618.9** | 11.50 s | 11.55 s |
| 96 | 495.2 ⚠️ | 22.62 s | 22.68 s |
| 128 | 568.1 | 24.61 s | 25.59 s |
| 192 | 569.0 | 35.26 s | 39.80 s |
| 256 | 566.2 | 48.93 s | 49.87 s |

### Finding: throughput plateaus at B=64 and never rises again

Unlike the A6000 — still climbing at its VRAM-limited ceiling of B=32 — this
H100 **peaks at B=64 and is flat within noise from 64 through 256**
(566–619 tok/s throughout). Pushing `BATCH_MAX_SIZE` past 64 buys nothing:
p50 latency scales roughly linearly with B (11.5 s → 48.9 s, a 4.3×
increase from 64→256) while throughput is unchanged. **On this hardware,
64 is the production setting, not 256** — the A6000's memory ceiling and
this GPU's actual throughput ceiling are two different numbers, and the
"push past 32 since we have 80 GB" premise this sweep started from turned
out to be the wrong lever: headroom here is VRAM, not throughput.

**The B=96 dip (495.2, ~20% below both neighbors) is suspicious and likely
not noise.** `get_best_config`'s config selection (`fused_moe_triton.py`)
picks the *nearest* tuned M from `moe_tune_config.json` by `min(abs(k - M))`;
the tuner in §2 tuned M∈{64, 128, ...}, so M=96 sits exactly equidistant
between two tuned points and `min()` breaks the tie by dict-iteration
order — an arbitrary choice, not a measured one. That would explain a
config mismatched to the actual shape at exactly this M and nowhere else.
**Not confirmed** — needs a print of which config `get_best_config` actually
picks at runtime for M=96 vs M=64/128 to verify.

### vs the A6000

At matched `BATCH_MAX_SIZE=32`: 590.2 tok/s here vs 243.2 tok/s on the
README's A6000 sweep — **2.43× higher throughput on the same batch size**,
consistent with the HBM2e-vs-GDDR6 bandwidth ratio (§ hardware notes,
~2.65×) given `fused_moe_kernel` is bandwidth-bound. The gap between the
2.43× measured and the 2.65× bandwidth ratio is consistent with fixed
per-request scheduling overhead (Python/asyncio in `src/server.py`) making
up a larger fraction of each request as raw kernel time drops.

---

## 6. FP8 (E4M3) weight quantization

`LLaDA_Quant` had no FP8 support at all going in — confirmed by an
exhaustive grep before starting; everything was built around a uniform
integer grid (`W_q = clamp(round(W/s), qmin, qmax)`, storage always int8,
int4 packed two-per-byte). Added `algorithms/fp8.py`: the same per-group
symmetric-scale contract, but the round/clamp step is replaced by a direct
cast to `torch.float8_e4m3fn`. Same 1 byte/weight footprint as INT8, a
different error distribution — E4M3's step widens with magnitude instead of
staying uniform, so which format wins on a given weight distribution is a
measurement question, not something derivable from the format alone.

Reused the existing PACKED-mode residency machinery end to end rather than
building a parallel system: `QuantResult` gained a `qtype` field that
dispatches `dequantize()` between the int and fp8 paths, so
`attach_packed_buffers`/`install_packed_expert_access`/
`materialize_expert_params` needed no changes beyond threading `qtype`
through. `QuantConfig` gained a `dtype` field (`"int"` default,
`"fp8_e4m3"` opt-in). Ships at the **same maturity tier INT4 already
established** — PACKED, dequantize-per-access, no fused kernel — rather
than a lesser-effort shortcut invented for fp8 specifically; a fused
FP8×BF16 (or native FP8×FP8 tensor-core) kernel is the same kind of
follow-up step INT8's fused W8A16 was, not something skipped here.

12 new unit tests (round-trip error budget, scale-to-format-max mapping,
storage-byte parity with INT8, an end-to-end PACKED-residency check
through a fake expert block). Full `LLaDA_Quant` suite: **276/276 pass**,
including the pre-existing fused-W8A16 integration tests, run with
`LLADA_INFERENCE_REPO` pointed at this repo. Wired into `test_llada` as
`--quantize fp8` (`src/server.py`), alongside the existing `int8`/`int4`.
Committed to `LLaDA_Quant` as `cc81002`.

### Memory: identical to INT8, and the naive measurement was wrong

First pass measured `nvidia-smi --query-gpu=memory.used` before/after
quantizing a *live server process* and got INT8 = FP8 = 18285 MiB, both
**higher** than BF16's 14621 MiB — i.e. quantization appearing to grow the
model. That's a real gotcha, not a real result: PyTorch's CUDA caching
allocator doesn't return freed BF16 buffers to the driver after a
quantize-then-free sequence, so `nvidia-smi` was reporting the allocator's
peak reserved footprint (BF16 + quantized briefly coexisting) rather than
the live resident set.

Re-measured with `LLaDA_Quant.memory.resident_memory` — tensor-level
accounting over the live module tree, keyed by storage pointer so tied
weights aren't double-counted, built for exactly this failure mode (its own
docstring: *"a run that grew the model by 52% was reported as a 47%
saving"*):

| | Resident | vs BF16 |
|---|---:|---:|
| BF16 | 14032.14 MiB | — |
| INT8 (packed) | 8080.14 MiB | **−42.4%** |
| FP8-E4M3 (packed) | 8080.14 MiB | **−42.4%** |

Identical to three decimal places, which is exactly what should happen:
both formats are 1 byte/weight over the same expert shapes and the same
`group_size=128`. **Lesson for this project going forward: measure
quantization memory in-process with `resident_memory`/
`compare_resident_memory`, never with `nvidia-smi` on a server that already
did the quantize-then-free dance.**

### GSM8K accuracy and latency — same box, same harness, n=50

All three arms measured fresh on this H100 (server restarted per arm),
`--quant-mode packed`, same GSM8K config as everywhere else in this log
(seed 42, `steps=512 block_length=64 confidence_threshold=0.9/0.4`):

| Arm | Accuracy | s/question | Fused kernel |
|---|---:|---:|:---:|
| BF16 | 74.0% (37/50) | 3.4 | n/a |
| INT8 (packed, fused W8A16) | 74.0% (37/50) | 4.2 | yes |
| FP8-E4M3 (packed, no fused kernel) | **70.0% (35/50)** | **12.5** | no |

**Latency is not a fair fight yet.** FP8 has no fused kernel and no
`torch.compile` path in this pass (both deliberately deferred, matching
where INT4 already sits) — it's paying the same "dequantize on every
weight access, eager, four kernel launches" cost INT8 paid before its
fused W8A16 kernel existed. INT8's *own* unfused number isn't in this
table (not re-measured this session), so 12.5s vs 4.2s overstates FP8's
ceiling; it does not overstate today's reality.

**Accuracy — 70.0% vs 74.0% is a 2-question difference at n=50, and this
project has already learned the hard way that deltas at this n are
noise-level** (§4's own n=50→n=200 investigation). This is a real,
measured result, not dismissed — but it is *reported*, not *concluded*.
Confirming whether FP8 is genuinely 4 points worse or within sampling
noise needs the same n=200 treatment §4 already applied to BF16.

### Open questions this raises, not yet answered

- Is the 70.0% vs 74.0% gap real or noise? Needs an n=200 FP8 run.
- With a fused kernel, does FP8 latency undercut INT8's 4.2s/question
  (H100 has native FP8 tensor cores; INT8 does too, so there's no a priori
  reason FP8 should win the *fused* race, but it hasn't been measured).
- Does per-channel or finer-group scaling narrow FP8's accuracy gap, given
  its error is concentrated where a group's weights vary widely in
  magnitude — the opposite failure mode from INT4's outlier problem that
  `scale_search="mse"` was built for (not applicable to fp8_e4m3 as
  implemented; `QuantConfig` currently rejects it there).

---

## 7. Parallelism under load: TP+EP=2 vs DP=2

Both topologies exercised end to end on the 2 GPUs, at rising concurrency,
same request shape throughout (`max_tokens=128 steps=128 block_length=32`).

### A real bug found launching this: `start_dp.sh` assumed its own CWD

`start_dp.sh` never `cd`s into its own directory before backgrounding
replica processes. `WEIGHT_DIR`/`LOG_DIR` are resolved to absolute paths up
front so weight loading was never affected, but the replica launch itself
is `"$PY" -m src.server ...`, and `-m` resolves the `src` package from the
process's CWD when nothing else sets `PYTHONPATH`. Invoked from inside the
repo (the README's own examples) this is invisible; invoked from anywhere
else — exactly what an orchestration script outside the repo does — every
replica dies instantly with `ModuleNotFoundError: No module named 'src'`.
Fixed with a `cd "$SCRIPT_DIR"` mirroring what `start.sh` already does.
Committed as `c45fe6c`.

### TP+EP=2: first real end-to-end run this project has done

`load_weights_tp` at `tp_size>1` had never executed — flagged earlier this
session as genuinely unexercised code. It worked cleanly on the first real
attempt: both ranks mapped their shard (1683 tensors each — not exactly
half of the 3,219 total, because embeddings/norms/`lm_head` replicate on
every rank while q/k/v/o_proj and expert weights shard), fused MoE blocks
built, server came up, served requests correctly.

**Correctness sanity** (n=10, same GSM8K config as everywhere else in this
log — too small to be more than a smoke test): **70.0% (7/10)**, in the
same range as BF16's measured 74.0% at n=50. Not a substitute for a real
n≥50 TP accuracy run, which hasn't been done — just confirmation that TP=2
is not obviously broken.

### The concurrency sweep

| | TP+EP=2 | | DP=2 | |
|---:|---:|---:|---:|---:|
| Concurrency | Tok/s | p50 | Tok/s | p50 |
| 1 | 24.8 | 4.14s | 23.3 | 3.57s |
| 8 | 24.5 | 35.98s | 220.8 | 3.96s |
| 32 | 25.1 | 138.05s | 814.0 | 4.58s |
| 64 | *(not measured — see below)* | | **877.8** (peak) | 5.92s |
| 128 | *(not measured)* | | 728.5 | 14.54s |

TP+EP's 64/128 rows were deliberately skipped: the trend through 1→8→32 is
already unambiguous — throughput dead flat regardless of concurrency,
latency climbing linearly with it (4.14s → 138.05s, a 33× increase for
zero throughput gain) — and finishing the sweep would have cost roughly 45
more minutes to reconfirm the same serialization with no new information.
This is `src/server.py`'s documented behavior working exactly as designed:
`request_lock` serializes every request when `tp_size > 1`, so concurrent
clients queue behind each other one at a time rather than batching.

**At concurrency 32, DP delivers 32.4× TP's throughput at 1/30th the
latency** (814.0 vs 25.1 tok/s; 4.58s vs 138.05s p50). This is the same
conclusion the README's Multi-GPU section already states from first
principles — measuring it end to end on real hardware is the point of this
section, not a surprise result.

### A finding that revises the README's own guidance: TP has no latency edge here either

The README currently frames TP+EP as *"the right choice for single-request
latency"*, backed by an A6000 figure: *"2× A6000 measured 6.15× vs a
single-GPU baseline at 128 tokens."* That comparison was against the
**unoptimized `src/` baseline**, not against `model_update/`'s own
single-GPU path.

At concurrency 1 here — the actual single-request case — **TP+EP=2 and a
single DP replica are statistically tied**: 24.8 vs 23.3 tok/s, 4.14s vs
3.57s p50. TP is not faster; if anything the single GPU edges it out. This
is consistent with §0's hardware note: these are **H100 PCIe with PHB
interconnect, no NVLink** (~25–50 GB/s between cards vs NVLink's ~900
GB/s), and TP's attention *and* MoE blocks each do a SUM all-reduce every
layer, every diffusion step — 32 all-reduces per layer-stack per step, all
over that link. On this box, TP's communication cost appears to fully
cancel out whatever compute-splitting benefit it would otherwise offer at
batch size 1. **The README's "TP for latency" framing needs an NVLink
caveat, or re-measurement on hardware that has one** — it is not free-
standing hardware-independent guidance, and this session is the first time
it's been checked against a same-generation single-GPU baseline at all.

### DP scaling efficiency: real, but not linear — open question

DP=2's peak (877.8 tok/s at concurrency 64) is **1.42×** the single-GPU
sweep's own peak from §5 (618.9 tok/s at `BATCH_MAX_SIZE=64`), not the 2×
a naive doubling would predict. Two replicas, independent GPUs, no shared
state in the hot path — there's no obvious structural reason for a
36-point efficiency loss, and it wasn't diagnosed further this session
(would need `/v1/replicas` polled *during* the sweep to see whether
least-outstanding actually splits load evenly at these concurrencies, or
whether the router itself adds a measurable per-request tax). Flagged, not
explained.

**Also note the same shape as §5's single-GPU curve**: DP throughput peaks
at concurrency 64 and *drops* at 128 (877.8 → 728.5) while p50 latency more
than doubles (5.92s → 14.54s) and p95/p99 blow out to ~24s. Same lesson as
§5: past a hardware-specific ceiling, more concurrency only buys queueing
delay. For this deployment, **64 total concurrency (32/replica) is close
to the actual ceiling**, not 128.

---

## 8. Five follow-ups, closed out

All five candidates from the end of §7's discussion, run this session.
Two ran genuinely in parallel (GPU 0 / GPU 1) since they were independent
single-GPU experiments; the other three needed both GPUs and ran
sequentially.

### 8a. The B=96 hypothesis (§5) — refuted by direct measurement

§5 guessed the throughput dip at `BATCH_MAX_SIZE=96` came from
`get_best_config`'s `min()` tie-break picking an arbitrary config between
the M=64 and M=128 tuned entries. Measured directly instead of guessed:

| Config at M=96 | Latency | Padding | Score |
|---|---:|---:|---:|
| **M=64 config (what's actually selected)** | **0.4630ms** | 37.5% | **0.5498** |
| M=128 config (the other tied candidate) | 0.4677ms | 43.8% | 0.5701 |
| Best of an independent 330-candidate sweep at M=96 | 0.4682ms | 41.7% | 0.5657 |

**The current pick is the best of all three** — better than the other tied
candidate, and better than what an unconstrained sweep specifically for
M=96 found. The tie-break is not the cause of the dip. Retracting the
hypothesis; §5's single-sample dip is more likely ordinary benchmark noise
(each `BATCH_MAX_SIZE` in that sweep was measured once, no repeats) than a
kernel-selection bug. Re-measuring B=96 with repeated trials would confirm,
but the specific mechanism I originally proposed is dead.

### 8b. Fusion A/B on H100 — a real number to replace the A6000-only one

Same methodology as the README's A6000 table, at this hardware's own
measured peak (`BATCH_MAX_SIZE=64`, concurrency 64, fixed prompts):

| | Tok/s | p50 |
|---|---:|---:|
| Fusions off (`LLADA_FUSE_RMSNORM=0 LLADA_FUSE_DECODE=0 LLADA_MOE_FUSED_SILU=0`) | 522.0 | 15.25s |
| **Fusions on (default)** | **698.7** | **11.26s** |

**+33.8% throughput from the fusions on H100**, against the A6000's
measured +19.7% (at its own peak, `BATCH_MAX_SIZE=32`). Larger gain on
faster hardware is the expected direction — the fusions remove fixed
per-call overhead (kernel launches, intermediate HBM round trips) that
matters more as raw compute time shrinks.

### 8c. Decoding-collapse hypothesis (§4d) — confirmed, and resolved

§4d found 5/200 GSM8K questions (indices 96, 122, 139, 168, 200)
collapsing into deterministic degenerate repetition, and hypothesized this
was the project's documented `steps_per_block ≥ block_length/2` stability
boundary being hit exactly at its minimum (ratio 0.5×) rather than
comfortably inside it. Tested by doubling `--steps` (1024 instead of 512,
same `max_tokens=1024 block_length=64`, so ratio 1.0×) on the same 200
questions, same server, same seed:

| `steps_per_block` ratio | Accuracy | Degenerate failures | Time |
|---|---:|---:|---:|
| 0.5× (original — reconfirmed a 3rd time) | 67.5% (135/200) | **5/200**, same indices as always | 768.8s |
| **1.0× (double)** | **71.0% (142/200)** | **0/200** | 816.0s (+6%) |

**Confirmed and fully resolved.** All 5 known failures vanish at ratio
1.0×, accuracy rises 3.5 points, and the time cost is only 6% — not the
2× a naive doubling of steps would suggest, most likely because
confidence-threshold decoding's early-exit absorbs most of the extra step
budget once a block is already resolved. **Actionable finding: the
README's `steps_per_block ≥ block_length/2` guidance describes where
generation stops being *catastrophically* broken (0% accuracy below it),
not where it's actually safe to run.** The knife's edge at exactly 0.5× is
where this project has been running GSM8K the whole time, and it costs
2.5% of real questions. Recommend defaulting to 1.0× rather than 0.5×,
accuracy figures elsewhere in this project's history permitting.

### 8d. Quantized DP — the original hypothesis, refuted

§7 proposed that INT8's 42.4% memory saving (§6) might let more replicas
fit per GPU and raise aggregate throughput. Tested directly: `--gpus 2
--replicas 4` (2 replicas/GPU) at INT8, against the same replica count at
BF16 (isolates oversubscription from quantization) and against the
already-measured BF16 1-replica/GPU baseline from §7:

| Concurrency | BF16, 1/GPU (§7) | BF16, 2/GPU | INT8, 2/GPU |
|---:|---:|---:|---:|
| 32 | 814.0 | 395.3 | 313.4 |
| 64 | **877.8** (peak) | 751.6 | 459.7 |
| 128 | 728.5 | 319.3 | 239.0 |

**Strict ordering at every concurrency: 1-replica BF16 > 2-replica BF16 >
2-replica INT8.** Both effects hurt, and they compound rather than
offset:

- **Oversubscription alone hurts** (BF16 1/GPU vs 2/GPU): co-locating two
  replicas on one GPU doesn't add HBM bandwidth, and `fused_moe_kernel` is
  bandwidth-bound (measured earlier at 81% of theoretical peak) — two
  replicas' kernels now compete for the same bytes/second rather than each
  getting the full card.
- **Quantization makes it worse, not better** (BF16 2/GPU vs INT8 2/GPU):
  counter-intuitive, since the fused W8A16 kernel reads int8 directly and
  streams *fewer* bytes per forward than BF16. Two candidate explanations,
  neither confirmed: the fused kernel's tuned config (§2, tuned under
  *exclusive* GPU access) may not hold up under compute contention from a
  co-resident process on the same 114 SMs; or int8's extra
  scale-multiply work, cheap when bandwidth-bound, becomes the bottleneck
  once bandwidth is shared and the kernel becomes compute-bound instead.
- INT8 also **collapses harder under overload**: −48% from its own peak
  at concurrency 128 (459.7 → 239.0), against BF16 2/GPU's milder −58%→
  actually comparable in relative terms, but INT8's absolute ceiling is
  lower throughout, so the same overload behavior lands it further below
  the safe operating point.

**The original premise was wrong.** Memory headroom was never the binding
constraint on this box (14032 MiB resident against 80 GB of VRAM) — the
bottleneck is HBM bandwidth, and nothing about having spare capacity
changes that. **One BF16 replica per GPU remains the right call on this
hardware.**

### 8e. TP GSM8K at n=50 — accuracy fine, but a real correction to §7's latency claim

| | Accuracy | s/question |
|---|---:|---:|
| TP+EP=2, n=50 | 70.0% (35/50) | **19.4** |
| BF16 single-GPU, n=50 (§4a) | 74.0% (37/50) | 3.4–3.8 |

Accuracy is unremarkable — in the same range as the n=10 sanity check and
close to BF16's 74.0%, nothing an n=50 sample can distinguish from noise
per this project's own established standard. **The latency number is the
finding, and it revises §7.**

§7 measured TP+EP=2 and a single DP replica as statistically tied at
concurrency 1 — but that was at the throughput benchmark's request shape
(`max_tokens=128 steps=128`). GSM8K generates up to 1024 tokens over up to
1024 diffusion steps, and **TP's per-step NCCL all-reduce cost is fixed
per step, so it accumulates linearly with step count** — at 128 steps it's
small enough to be invisible; at up to 1024 steps (8× more), it dominates.
**§7's "TP has no latency edge here either" finding does not generalize
past short generations.** For a realistic GSM8K-length workload, TP is
~5–6× slower than a single GPU, not tied with it. The README's "TP for
single-request latency" framing needed revising for a different reason
than §7 first found: not because TP has no edge on this no-NVLink
hardware, but because whatever edge it might have is workload-length-
dependent and vanishes (worse: inverts) exactly on realistic generation
lengths.

---

## 9. Open items from this session

- [ ] `moe_tune_config.json` still not committed to git, and still not
      device-keyed (unlike `dInfer/configs/*device_name=...json`). Should
      be renamed/keyed and committed so it isn't silently regenerated or
      silently stale on the next machine.
- [ ] Autotuner's `shmem=None occ=None` introspection failure — cosmetic
      so far, but worth root-causing.
- [x] Decoding-collapse hypothesis — **confirmed and resolved in §8c.**
      `steps_per_block` ratio 1.0× eliminates all 5 known failures.
      Remaining: the README's stability guidance (`≥ block_length/2`,
      i.e. ratio ≥ 0.5×) should probably become `≥ block_length`
      (ratio ≥ 1.0×) as the recommended default, not just the floor.
- [x] The B=96 tie-break hypothesis — **refuted in §8a.** The current pick
      is measurably the best of the candidates tried; the dip is more
      likely single-sample noise from §5's sweep (no repeated trials).
      Re-measuring B=96 with repeats would confirm but is low priority
      now that the leading hypothesis for *why* is dead.
- [x] DP across both GPUs, TP=2 correctness pass — done in §7.
- [x] Fusion A/B — **done in §8b.** +33.8% on H100 (698.7 vs 522.0 tok/s),
      vs the A6000's +19.7% — larger gain on faster hardware, as expected.
- [x] TP=2 GSM8K at n≥50 — **done in §8e.** 70.0% (35/50), unremarkable.
      Surfaced a bigger finding: 19.4s/question against BF16's 3.4–3.8s,
      because TP's NCCL cost is fixed *per diffusion step* and GSM8K runs
      far more steps than the throughput benchmark that found TP/DP tied.
- [x] Quantized DP (the memory-headroom-buys-more-replicas idea) —
      **tested and refuted in §8d.** Oversubscription alone hurts
      throughput on this bandwidth-bound kernel; adding quantization on
      top hurts more, not less. One replica per GPU remains correct here.
- [ ] TP+EP=2's concurrency 64/128 rows — still deliberately skipped (§7);
      the trend was already unambiguous at 1/8/32, and §8e's finding
      (TP's cost scales with step count, not request-shape-independent)
      makes the 128-token throughput-benchmark numbers less central to
      the real question anyway.
- [ ] DP's 1.42×-not-2× scaling efficiency (§7) — still undiagnosed. Needs
      `/v1/replicas` polled during a sweep to see whether load actually
      splits evenly, or whether the router itself taxes each request.
- [ ] README's Multi-GPU section needs updating for **two** reasons now,
      not one: §7 found TP+EP=2 has no latency edge over a single GPU on
      this no-NVLink hardware at short generation lengths, and §8e found
      that even where TP might have an edge, it inverts at realistic
      (GSM8K-length) generation because its NCCL cost scales with step
      count. The current "TP for latency" framing should go, not just
      get a caveat.
- [ ] Quantized-DP's counter-intuitive result (§8d: INT8 slower than BF16
      under co-location, despite streaming fewer bytes) — two candidate
      explanations offered, neither confirmed. Would need per-kernel
      profiling under contention (two processes on one GPU) to settle,
      not just end-to-end throughput numbers.
- [ ] FP8 n=200 accuracy re-run (§6) — the 70.0% vs 74.0% n=50 gap needs
      the same noise check §4 already ran for BF16.
- [ ] FP8 fused kernel — currently dequantize-per-access only; the 12.5s vs
      4.2s/question latency gap in §6 is not apples-to-apples until one
      exists.
