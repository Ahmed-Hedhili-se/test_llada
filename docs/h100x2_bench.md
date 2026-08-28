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

## 1. Correctness gates (`tests.test_*`)

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
python dminfr.tuning.autotune_moe.py --model FULL_CFG
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
python -m benchmarks.check_time_inference --weight-dir weights \
    --gen-length 128 --steps 128 --block-length 32 --mode both --num-runs 3
```

| | Time | Tok/s | Speedup |
|---|---:|---:|---:|
| Baseline (`src/`, unfused, no cache) | 32.38 s | 3.95 | 1.00× |
| **Optimized** (`dminfr/engine/`) | **3.60 s** | **35.57** | **9.00×** |

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
python -m benchmarks.correctness.run_math_reasoning_code --task gsm8k \
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
BATCH_MAX_SIZE=$B python -m dminfr.serving.server --backend fast_dense --weight-dir weights
python -m benchmarks.throughput.run_throughput --base-url http://localhost:8000 \
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
per-request scheduling overhead (Python/asyncio in `dminfr/serving/server.py`) making
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
`--quantize fp8` (`dminfr/serving/server.py`), alongside the existing `int8`/`int4`.
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
is `"$PY" -m dminfr.serving.server ...`, and `-m` resolves the `src` package from the
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
This is `dminfr/serving/server.py`'s documented behavior working exactly as designed:
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
**unoptimized `src/` baseline**, not against `dminfr/engine/`'s own
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

## 9. Chasing accuracy without paying throughput

Goal: raise GSM8K accuracy from the §8c baseline (71.0%, n=200) without
giving back the throughput from §5/§7. Two candidate mechanisms were
tested and one methodological finding fell out that matters more than
either.

### 9a. Router precision — a gap the closed investigation never tested

`INVESTIGATION_LOG.md` §2.11 closed the `dminfr.engine`-vs-HF accuracy gap
as **inherent numerical noise**, on the strength of a 2×2 kernel-isolation
matrix (Triton-vs-eager MoE × eager-vs-SDPA attention). That matrix varied
the MoE kernel and the attention kernel. **It never varied the router's
own precision.**

Both implementations do the same thing — reference at
`weights/modeling_lladamoe.py:676-678`, ours previously at `model.py:133`:

```python
router_logits  = self.gate(hidden_states)                 # bf16
routing_weights = F.softmax(router_logits, dtype=float)   # fp32
```

The fp32 upcast is **cosmetic**: `topk` ranks the softmax output, but
softmax is monotonic, so the ordering was already decided when the logits
were rounded to bf16 one line earlier. And §2.11's own measurement says
that ordering is fragile — top-1 routing weight of **1.7–5%**, against
1.56% for a flat 64-way distribution, meaning the whole logit vector sits
in a band narrow enough that the top-8/top-9 boundary is decided by
differences comparable to bf16's own resolution.

**Measured directly** (`_router_logits`, gated by `LLADA_FP32_ROUTER`;
probe on a real 191-token chat-templated GSM8K prompt, all 16 MoE layers,
comparing top-8 *sets* from bf16 vs fp32 logits on identical hidden
states):

| Layer | top-8 set differs | top-1 differs | mean top-1 weight |
|---:|---:|---:|---:|
| 0 | **94.2%** | 34.6% | 2.15% |
| 1 | 58.6% | 9.4% | 2.71% |
| 2 | 43.5% | 4.2% | 2.60% |
| 7 | 10.5% | 1.6% | 4.74% |
| 13 | 11.5% | 1.6% | 8.08% |
| **overall** | **24.1%** | 4.5% | — |

**bf16 rounding in the gate alone flips top-8 membership on 24.1% of
positions**, against the 43–90% total divergence §2.11 measured versus HF.
So a substantial share of what was closed as "unavoidable inter-
implementation noise" is self-inflicted by our own gate precision. The
per-layer correlation is exactly the predicted mechanism: layer 0 has the
most uniform router (2.15% top-1) and the most divergence (94.2%); the
deeper, sharper layers diverge least.

### 9b. Does it help? +3.0 points, but **not** statistically significant

n=200, seed 42, `steps=1024 block_length=64 threshold 0.9/0.4`, both arms
run **simultaneously on separate GPUs** so questions/session/box are
identical:

| Arm | Accuracy |
|---|---:|
| fp32 router | **74.0% (148/200)** |
| bf16 router (control) | 71.0% (142/200) |

The control reproduced §8c's 71.0%/142 **exactly**, confirming the
comparison is clean and the pipeline deterministic.

A +3.0pt marginal at n=200 is inside one standard error, so the marginal
alone proves nothing. These are *paired* runs, so the right test is
McNemar on the discordant pairs:

| | count |
|---|---:|
| both correct | 125 |
| **fp32 only (gained)** | **23** |
| **bf16 only (lost)** | **17** |
| both wrong | 35 |
| discordant | 40 (net **+6** for fp32) |
| **exact McNemar, two-sided** | **p = 0.43** |

**Not significant.** And the more important number is the churn: **40 of
200 questions — 20% of the test set — flip outcome from a router-precision
change alone.** That is the real result. This checkpoint's near-uniform
router makes a fifth of GSM8K a near-coin-flip under *any* numerical
perturbation.

**Methodological consequence for this project:** every accuracy comparison
at n=200 carries roughly ±3pt of intrinsic churn. Marginal comparisons at
that scale cannot distinguish a real 3-point effect from noise — paired
McNemar is required, and even that needs n≈1000 for the power to resolve
an effect this size. This retroactively explains the n=50-vs-n=200
instability in §4 and the historical numbers the project already retired.
It also means **§8c's +3.5pt deserves the same scrutiny** — though ~5 of
its 7 questions were the deterministic collapses, a mechanistic effect
rather than churn.

### 9c. Throughput cost: none

The constraint was to not give back throughput. Measured at production
scale (`BATCH_MAX_SIZE=64`, concurrency 64, fixed prompts, single GPU):

| | Tok/s | p50 |
|---|---:|---:|
| fp32 router ON | 703.2 | 11.51s |
| fp32 router OFF | 705.5 | 11.52s |

**−0.3%, inside noise.** As predicted from the ratio: the gate is
2048×64 = 131k parameters against ~805 MB of expert weights streamed per
layer, so on a kernel that is bandwidth-bound on the experts it cannot
register.

One caveat worth recording, because it looks alarming in isolation: on the
**single-stream** GSM8K path fp32 routing cost **+11.7%** (802.8s vs
718.6s for n=200). That is not a contradiction — at concurrency 1 the gate
sees only ~64 rows, the pipeline is kernel-launch-bound rather than
bandwidth-bound, and the extra cast kernels are visible. The cost vanishes
exactly where throughput is actually made.

### 9d. Extraction failures — ruled out, no free accuracy there

Hypothesis: some wrong answers are correct reasoning that the grader
mis-extracts (the README flags a "last number anywhere" fallback). Tested
against saved transcripts for all 200 questions:

- **14 of 58** wrong answers *do* contain the expected value somewhere in
  the response — but **0 of those have it as the final number.**
- Inspecting them, they are genuine reasoning errors, not extraction
  errors. Q108 is representative: expected 60,000 (hours), the model
  computed 60,000 correctly and then converted to "2,500 days" — a
  question-comprehension failure, graded correctly as wrong.

**The grader is working.** There is no free accuracy hiding in extraction.

Also measured, from the same transcripts: wrong answers are **longer**
than correct ones (median 743 vs 542 chars; max 3,834 vs 1,362),
consistent with §2.9's length-bucketed finding that failures concentrate
in long responses. And **5 responses still exceed 2,000 characters** even
at `steps_per_block` 1.0× — so §8c reduced degeneration rather than
eliminating it entirely.

### 9e. n=1000 settles it: no effect. And the headline number was wrong.

Both arms re-run at **n=1000** — the scale McNemar needs to resolve a
3-point effect — again simultaneously on separate GPUs:

| Arm | Accuracy |
|---|---:|
| fp32 router | 75.7% (757/1000) |
| bf16 router (control) | 75.2% (752/1000) |

| | count |
|---|---:|
| both correct | 671 |
| fp32 only (gained) | 86 |
| bf16 only (lost) | 81 |
| both wrong | 162 |
| discordant | 167 (net **+5**), churn **16.7%** |
| **exact McNemar, two-sided** | **p = 0.757** |

**The fp32 router has no accuracy effect.** The +3.0pt at n=200 collapsed
to +0.5pt at n=1000 — textbook regression to the mean, and exactly what
the n=200 McNemar (p=0.43) was warning about. Had this been adopted on the
n=200 point estimate, the project would have shipped a permanent latency
cost on the single-stream path for nothing.

**The more valuable result is incidental: accuracy is 75.2%, not 71.0%.**
Every accuracy figure in §4/§8c/§9b came from the same 200-question
subset, and that subset is simply harder than the full test set. At n=1000
the same engine, same config, same code scores **75.2%**. Nothing changed
but the sample.

That reframes the whole accuracy picture in this log:

| Measurement | Value | Status |
|---|---:|---|
| n=50 | 74.0% | unrepresentative (§4) |
| n=200 | 71.0% | unrepresentative — pessimistic by ~4pt |
| **n=1000** | **75.2%** | **the number to quote** |

It also lands the engine much closer to the HF reference than the n=200
figure implied. The documented same-harness reference points are HF 74% at
steps=256 and 82% at steps=512 (n=50, pre-`_stable_subset`, so not
directly comparable) — 75.2% sits inside that band rather than well below
it.

### 9f. Status and what this cost

`LLADA_FP32_ROUTER` is committed **default-off**, kept deliberately rather
than reverted. Two reasons: the probe result in §9a (bf16 gate rounding
alone flips top-8 membership on 24.1% of positions) is a real and useful
fact about this checkpoint that future work will want; and this is the
obvious idea to have after reading §2.11's conclusion, so leaving the
switch plus the negative result in place stops someone spending another
two hours rediscovering it.

**Net accuracy gained from this section: zero.** The one mechanism that
looked promising was measured properly and rejected. What the section
actually produced is three corrections:

1. The engine's accuracy is **75.2%**, not 71.0% — the previously-quoted
   figure came from an unrepresentative subset.
2. **~17% of GSM8K flips under any numerical perturbation** on this
   checkpoint, so n=200 comparisons carry ±3pt of intrinsic churn and
   marginal comparison at that scale is not a valid method here.
3. §2.11's "inherent noise" conclusion is **partly wrong** — 24.1% of the
   divergence it attributed to unavoidable inter-implementation
   differences is self-inflicted gate precision. It does not cost accuracy
   (proven above), but the mechanism was misattributed.

---

## 10. Nsight Compute profiling: the "no headroom" claim is wrong on H100

### Tooling

Nsight Compute is not installed on this image and has no apt candidate
until NVIDIA's repo is added (`cuda-keyring_1.1-1_all.deb`, then
`nsight-compute-2025.3.1` — the distro-provided `nsight-compute` resolves
to a 2021 build that predates sm_90). GPU performance counters are
admin-gated, so a normal-user run fails with `ERR_NVGPUCTRPERM`; `sudo ncu`
works and avoids rebooting to flip
`NVreg_RestrictProfilingToAdminUsers`. Binary at
`/opt/nvidia/nsight-compute/2025.3.1/ncu`.

Profiling target is a real forward at the production shape the throughput
sweep peaks at: `BATCH_MAX_SIZE=64`, `block_length=32` → M = 2048, with a
256-token prompt prefix so attention sees a realistic context.

### Kernel breakdown (H100, M=2048)

| Kernel | Self ms | % GPU | Calls |
|---|---:|---:|---:|
| **`fused_moe_kernel`** | 504.47 | **55.29%** | 160 |
| `sm90_xmma_gemm` (cuBLAS, attn/qkv/o_proj) | 110.19 | 12.08% | 325 |
| `elementwise_kernel` (×4 variants, combined) | 113.64 | **12.46%** | 800 |
| `_rmsnorm_kernel` (ours, fused) | 49.47 | 5.42% | 325 |
| `CatArrayBatchedCopy` | 38.40 | 4.21% | 160 |
| `reduce_kernel` | 30.73 | 3.37% | 80 |
| `flash_fwd_kernel` | 27.02 | 2.96% | 80 |

182.48 ms/forward. `fused_moe_kernel` at 55.3% tracks the README's 58.75%
on A6000 — it is still the dominant kernel.

### The headline: it is L2-bound, not DRAM-bound

README currently states `fused_moe_kernel` runs at **"81% of theoretical
weight-streaming bandwidth"** and concludes **"There is no kernel headroom
left there."** Measured on H100 (Speed-of-Light, GEMM2 launch):

| Metric | Value |
|---|---:|
| **DRAM Throughput** | **14.53%** |
| **L2 Cache Throughput** | **79.70%** |
| Memory Throughput (max of hierarchy) | 79.32% |
| Compute (SM) Throughput | 49.59% |
| Tensor pipe utilisation | 49.59% |
| **L1/TEX Hit Rate** | **0.09%** |
| Achieved occupancy | 12.43% |
| Registers/thread | 234 (GEMM2) / 148 (GEMM1) |
| Local memory spilling | **0** |

ncu's own verdict: *"Memory is more heavily utilized than Compute: Look at
the Memory Workload Analysis section to identify the **L2 bottleneck**."*

**DRAM sits at 14.5%, not 81%.** The engine is nowhere near a memory-
streaming wall on this hardware. What saturates is **L2 at 79.7%** — the
expert weights are being served largely out of L2 (`moe_align_block_size`
groups tokens by expert, giving ~256 rows of reuse per expert at M=2048),
so DRAM barely works while L2 does everything.

The README's *reasoning* was right and its *conclusion* was too strong: it
already noted "Nsight's 96% is L2, not DRAM" and correctly inferred
"remove intermediate traffic". But the accompanying "81% of weight-
streaming bandwidth / no headroom left" framing is an A6000 artifact and
does not survive the move to H100. **There is headroom; it is behind L2
traffic, not DRAM bandwidth.**

Occupancy is 12–18%, entirely register-limited (234 regs × 128 threads =
29,952 of the SM's 65,536 → 2 blocks; 8 warps of 64 = 12.5%, matching the
measured 12.43%). Notably **no register spilling**, so the tuner's configs
are not pathological — they trade occupancy for ILP deliberately.

`L1/TEX hit rate of 0.09%` is the other structural fact: Triton's
pipelined loads go global→shared via `cp.async` and bypass L1 entirely, so
L1 contributes nothing and all reuse pressure lands on L2.

### Two optimisations tested and rejected

**Raising the tuner's `BLOCK_SIZE_M ≤ 64` cap.** Hypothesis: each expert's
weight tile is re-read from L2 once per M-block covering it, so at M=2048
(256 rows/expert) `BM=64` re-reads every tile 4× and `BM=128` would halve
the dominant L2 consumer. Swept the full config grid at caps 64/128/256
across M ∈ {512, 1024, 2048, 4096}:

| M | best @ cap64 | best @ cap128 | Δ latency |
|---:|---:|---:|---:|
| 512 | 0.5330 ms | 0.5073 ms | +4.8% |
| 1024 | 0.6249 ms | 0.6459 ms | −3.4% |
| 2048 | 0.9863 ms | 0.9913 ms | −0.5% |
| 4096 | 1.8525 ms | 1.8381 ms | +0.8% |

**Refuted.** The search picks `BM=64` (or 32) even when 128/256 are
allowed, at every M — at M=2048 it returns the identical config. Larger
tiles cost more in occupancy than they save in L2 traffic, and the tuner
was already optimal within its space. The `≤64` cap is not leaving
anything on the floor.

**Upgrading Triton for Hopper codegen.** Built an isolated venv
(`~/venv_new`, torch 2.11.0+cu128 / Triton 3.6.0 vs the pinned
2.5.1+cu124 / 3.1.0) and ran the identical kernel:

| M | Triton 3.1.0 | Triton 3.6.0 | Δ |
|---:|---:|---:|---:|
| 512 | 0.5315 ms | 0.5244 ms | −1.3% |
| 1024 | 0.6396 ms | 0.6120 ms | −4.3% |
| 2048 | 0.9626 ms | 0.9331 ms | −3.1% |
| 4096 | 1.7389 ms | 1.7286 ms | −0.6% |

**1–4%, not the Hopper win hypothesised.** The compiler improved, but
Hopper's real features (TMA descriptors, warp specialisation, persistent
wgmma pipelining) require *explicit opt-in in the kernel source* —
upgrading the compiler does not rewrite the kernel. Not worth a torch
upgrade's correctness risk on its own.

### Two opportunities that are real, with evidence

Op-level attribution (`ProfilerActivity.CPU + CUDA`) over the same forward
identifies where the non-MoE 45% actually goes:

**1. RoPE's `rotate_half` is unfused — ~4% of GPU time.**
`aten::cat` (3.04%, 165 calls) and `aten::neg` (0.90%, 160 calls) are
2-per-layer-per-forward, which is exactly `rotate_half`'s
`torch.cat([-x2, x1], dim=-1)`. This materialises a full rotated copy of Q
and K every layer, every step, purely to express a permutation. A Triton
RoPE kernel that applies the rotation in-register removes the `cat`, the
`neg`, and their L2 traffic. **Low risk, well-understood, ~4% available.**
This is the same class of win as the RMSNorm and decode-tail fusions the
project already landed.

**2. The MoE top-k combine is a separate reduction — ~2.4–3.4%, plus L2
relief.** `aten::sum` (2.43%, 160 calls = 2/layer) and `reduce_kernel`
(3.37% kernel-side) are the `cache2.sum(dim=1)` that combines the `top_k`
expert outputs. GEMM2 writes an `[M, top_k, K]` intermediate — at M=2048
that is **67 MB written then 67 MB read back per layer**, ×16 layers, all
through the L2 that ncu says is the bottleneck at 79.7%. Fusing the
combine into GEMM2's epilogue would write `[M, K]` (8.4 MB) directly and
delete the read entirely. Higher effort than the RoPE fusion — it needs
either atomics or a grid remap that interacts with `moe_align_block_size`'s
expert-major ordering — but it attacks the measured bottleneck head-on,
and it is the same optimisation the SiLU epilogue already did for
`intermediate_cache1`.

Neither was implemented this session. Both are measured, not guessed.

---

## 11. Implemented: fused RoPE (+4.2% throughput, bit-exact)

§10's first identified opportunity, built and shipped.

`rotate_half` was `torch.cat([-x2, x1], dim=-1)` — materialising a full
rotated copy of q and k every layer, every denoising step, purely to
express a permutation of the head dimension. The rotation is an index
permutation, so a kernel reads the partner element directly and never
builds the copy:

```
d <  HD/2 :  out[d] = x[d]*cos[d] + (-x[d + HD/2])*sin[d]
d >= HD/2 :  out[d] = x[d]*cos[d] +   x[d - HD/2] *sin[d]
```

One program per (batch, head, token); the whole head dim (HD=128) fits in
one block, so the partner element is already in the same contiguous load.
`cos`/`sin` arrive as either `[T, HD]` (shared) or `[B, T, HD]` (per-row,
left-padded) — the kernel folds the batch stride to 0 for the shared case
rather than needing a second kernel or a broadcast copy.

### Bit-exact, deliberately

Unlike the RMSNorm and decode-tail fusions, this one reassociates nothing,
so it can be — and therefore must be — bit-exact. `cos`/`sin` are cast to
the model dtype before `apply_rope` (`model.py`), so the eager path is
three bf16 ops, each computed in fp32 by ATen's opmath and rounded back.
The kernel reproduces that order exactly (multiply, round, multiply,
round, add, round) rather than taking the more accurate fully-fp32 route —
the same discipline the SiLU epilogue applies, and for the same reason:
being *more* accurate than the reference is still a behaviour change.

`tests/test_fused_rope.py` asserts `torch.equal`, not a tolerance, across
8 shape/layout combinations including GQA (`KVH != NH`), odd `T`, batch-1
decode and the production `B=64, T=32`:

| | Result |
|---|---|
| Bit-exactness, all 8 cases, q and k | **bit-exact** |
| Kernel speed, B64 T32 shared | 0.1458 → 0.0679 ms (**2.15×**) |
| Kernel speed, B64 T32 per-row | 0.1552 → 0.0679 ms (**2.29×**) |
| Kernel speed, B1 T32 (decode) | 0.1055 → 0.0533 ms (1.98×) |

Because it is bit-exact at the op level, q and k are identical downstream,
so **accuracy is provably unchanged** — no GSM8K re-run was needed to
establish that, which is exactly what bit-exactness is worth.

### A hidden copy, found by reviewing my own kernel

The first version called `.contiguous()` on q/k. `Attention.forward` passes
`transpose(1, 2)` **views** of `[B, T, NH, HD]` — strides
`(T*NH*HD, HD, NH*HD, 1)`, not contiguous — so that call was silently
copying both tensors in full, every layer, every step, reintroducing most
of the traffic the fusion exists to remove.

The kernel already takes explicit strides and only needs the head axis to
be unit-stride, which a transpose of the last-two-of-four axes preserves.
Dropping the copy (and passing input and output strides separately rather
than assuming they match) took the kernel from 2.15× to **2.55×**. The
test now covers non-contiguous inputs explicitly, since that is the layout
production actually uses.

| Case | eager | fused | speedup |
|---|---:|---:|---:|
| B64 T32 shared, transposed | 0.1763 ms | 0.0692 ms | **2.55×** |
| B64 T32 per-row, transposed | 0.1776 ms | 0.0692 ms | **2.57×** |
| B32 T64 per-row, transposed | 0.1784 ms | 0.0692 ms | **2.58×** |
| B1 T32 (decode), transposed | 0.1066 ms | 0.0575 ms | 1.85× |

All 11 bit-exactness cases pass, contiguous and transposed.

### End-to-end: +4.2%, but the harness cannot resolve it

The first A/B gave `on` 730.3 vs `off` 700.9 tok/s — **+4.2%**, matching
§10's profile estimate. The next run of the *same script* gave `on` 705.1
vs `off` 733.7, i.e. **−3.9%**, with the **unchanged `off` arm moving
700.9 → 733.7 (4.7%)**. A single on/off pair proves nothing here.

Re-measured properly: arms alternated across rounds (never all-of-one then
all-of-the-other, which would confound the arm with thermal/clock drift),
3 benchmark reps per server start, 3 rounds → 9 samples per arm.

| Arm | n | mean tok/s | sd | min | max |
|---|---:|---:|---:|---:|---:|
| RoPE ON | 9 | **724.0** | 39.8 | 660.7 | 782.2 |
| RoPE OFF | 9 | 695.2 | 29.8 | 648.1 | 725.9 |

Difference **+28.9 tok/s (+4.15%)**. Pairing by round (the 3 reps inside
one server start are correlated, so the honest n is 3, not 9) gives
per-round deltas of **+44.5, +36.5, +5.6** — all three positive, mean
+28.9, **paired t p ≈ 0.14**.

**So: consistent direction, stable ~+4% point estimate across two
independent experiments, not separable from noise at p<0.05 with 3
rounds.** The claim "+4.2% throughput" as a precise figure is not
supported; "≈+4%, direction consistent, magnitude matches the profile
prediction" is.

What *is* solidly established, and is the reason this ships:

- The kernel is **2.55× faster** on its own (200 timed iterations after
  warmup — low variance, unlike the server harness).
- `aten::cat` and `aten::neg` are **gone from the profile entirely**,
  replaced by `_rope_kernel`, which also absorbs the `mul` and `add` that
  were separate elementwise launches.
- It is **bit-exact**, so it is strictly less work for identical output.

Regression suite 9/9. Shipped default-on (`LLADA_FUSE_ROPE=1`); the switch
is a kill switch, not a numerical hedge. Doing provably less work for
provably identical results is correct regardless of whether a noisy
end-to-end harness can resolve the gain.

### Methodological note: this project's throughput A/Bs are under-powered

The `off` arm moving 4.7% between runs of identical code sets the noise
floor for every single-pair throughput comparison in this log and in the
README. Two consequences:

- **§8b's fusion A/B (+33.8%) and the README's A6000 figure (+19.7%) were
  single pairs.** +33.8% is far outside a ±5% band so the conclusion
  survives, but the precision does not — treat it as "large", not as
  33.8%.
- **§5's batch sweep and §7's concurrency sweep were one sample per
  point.** The B=96 dip that §8a chased as a tuner tie-break artifact is
  comfortably inside this noise band, which is the simpler explanation
  §8a already landed on after the tie-break theory was refuted.

Anything under ~10% needs interleaved repeats to be claimed at all. This
is the throughput analogue of §9's accuracy-churn finding, and it has the
same fix: repeat, interleave, and report the spread.

---

## 12. Current throughput, measured with repeats

Every throughput number earlier in this log was **one sample per point**.
§11 established the noise floor at ~5% (an unchanged arm moved 4.7%), so
those points cannot be quoted as precise. Re-measured with **3 reps per
point**, fused RoPE on (the current default), same request shape
throughout (`max_tokens=128 steps=128 block_length=32`, varied prompts).

### Single GPU

| `BATCH_MAX_SIZE` | tok/s (mean) | sd | spread | p50 |
|---:|---:|---:|---:|---:|
| **32** | **656.9** | **3.5** | 652.1–660.4 | **5.33 s** |
| 64 | 585.0 | 34.7 | 547.0–630.9 | 12.07 s |
| 128 | 607.2 | 54.4 | 530.4–649.7 | 23.04 s |

### DP=2 (both GPUs, one replica each)

| Concurrency | tok/s (mean) | sd | spread | p50 |
|---:|---:|---:|---:|---:|
| 32 | 855.4 | 16.2 | 833.1–871.3 | **4.30 s** |
| **64** | **897.9** | 55.3 | 820.4–946.1 | 5.60 s |
| 128 | 601.4 | 80.2 | 534.8–714.3 | 18.21 s |

### Headline

**~900 tok/s peak on 2×H100** (DP=2, concurrency 64). At concurrency 32
you get 855 tok/s — 5% less throughput for **24% better latency** (4.30 s
vs 5.60 s) and a third of the variance, which is the better operating
point for anything latency-sensitive.

### This corrects §5: the single-GPU optimum is B=32, not B=64

§5's single-sample sweep concluded "throughput peaks at B=64 and is flat
through 256". With repeats that does not hold. **B=32 wins on all three
axes at once** — highest throughput (656.9 vs 585.0/607.2), less than half
the latency (5.33 s vs 12.07/23.04 s), and an order of magnitude less
variance (sd 3.5 vs 34.7/54.4). B=32's *minimum* across reps (652.1)
exceeds B=128's *maximum* (649.7), so the ordering is not a sampling
accident.

B=64 and B=128 are not distinguishable from each other — their ranges
overlap almost entirely (547–631 vs 530–650). §5 reported them as 618.9
and 568.1 and drew a curve through those points; both numbers were single
draws from distributions this wide.

**Practical consequence: `BATCH_MAX_SIZE=32` is the recommended
single-GPU setting**, not 64 and certainly not the 256 the "flat past 64"
reading would have permitted.

### The variance is itself a finding

Measurement noise scales sharply with batch size: sd 3.5 at B=32, 34.7 at
B=64, 54.4 at B=128; and 16.2 → 55.3 → 80.2 across the DP concurrencies.
Whatever the engine is doing at high batch is not just slower on average,
it is *less predictable* — a p50 that swings 21–27 s between identical
runs is a scheduling or memory-pressure symptom, not a throughput curve.
Not diagnosed here; flagged as an open item.

### DP scaling

Best-to-best, DP=2 delivers **897.9 / 656.9 = 1.37×** a single GPU, and at
matched concurrency 32 it is 855.4 / 656.9 = **1.30×**. Both are below
§7's reported 1.42×, and all three are well short of 2×. The sub-linear
scaling noted in §7 stands and remains undiagnosed.

---

## 13. Speedup, current

Re-measured after the RoPE fusion, 5 runs (§3's figures predate it).

### Engine, single request, same GPU

| | Time | Tok/s | Speedup |
|---|---:|---:|---:|
| Baseline (`src/`, unfused, no cache) | 34.92 s | 3.67 | 1.00× |
| **Optimized** (`dminfr/engine/`) | **3.43 s** | **37.29** | **10.17×** |

**Token divergence 0/128** — still character-identical to the baseline,
which the bit-exact RoPE fusion guarantees rather than merely suggests.

Read this carefully against §3's 9.00×: **both sides moved.** The
optimized arm went 3.60 s → 3.43 s (−4.7%, consistent with §11's ~+4%
RoPE result), but the baseline also went 32.38 s → 34.92 s (+7.8%).
`src/` loops 64 experts in Python and is CPU-dispatch-bound, so it drifts
with host load. **The honest statement is "the optimized path got ~5%
faster"; the ratio moving 9.00× → 10.17× overstates that**, because a
slower baseline inflates it for free.

### Total pipeline

Decomposed from separately-measured components, using the in-process
baseline (3.67 tok/s) rather than the README's over-HTTP one (2.7 tok/s on
A6000) — the in-process number is the more conservative comparison because
it does not charge the baseline for HTTP overhead:

| | Tok/s | vs baseline |
|---|---:|---:|
| Baseline, `src/`, single request | 3.67 | 1.00× |
| Optimized, single request (B=1) | 37.29 | **10.2×** |
| Optimized, 1 GPU batched (`BATCH_MAX_SIZE=32`) | 656.9 | **179×** |
| Optimized, 2 GPUs (DP=2, concurrency 64) | 897.9 | **245×** |

The decomposition is self-consistent: 10.16× engine × 17.62× batching =
179.0×, and 10.16 × 24.08 = 244.7× for the DP arm — matching the direct
ratios to within rounding, which is the check that makes the headline
figure trustworthy rather than a stacked guess.

**Caveat on precision.** The baseline drifted 7.8% between two
measurements of identical code, so these ratios carry at least that much
uncertainty. Using §3's baseline instead (3.95 tok/s) gives **166×** and
**227×**. Quote the range — ~165–180× on one GPU, ~225–245× on two — not
a single digit.

**And the DP figure spends twice the hardware.** 245× is a
deployment-to-deployment number against a single-GPU baseline; the
like-for-like engine comparison is the 10.2×, and the single-GPU pipeline
figure is 179×.

For context, the README's A6000 headline is 103× (8.70× engine × 11.67×
batching). The H100 single-GPU figure is higher on both factors: a
slightly better engine ratio, and considerably more batching headroom.

---

## 14. Closing out the backlog

### 14a. Why DP scales sub-linearly — and a measurement of mine that was wrong

§12 measured DP=2 at 1.30–1.37× a single GPU. To separate the candidates —
router overhead, uneven load balancing, or plain GPU/host contention — one
replica was driven **directly on its own port with no router in the path**,
then both together, then through the router.

| | tok/s | |
|---|---:|---|
| Replica 0 alone, direct | 642.6 ± 2.7 | |
| Replica 1 alone, direct | 625.0 ± 0.5 | |
| Both directly, simultaneously | 1190.2 ± 111.7 | **0.93×** of perfect 2× |
| Through the router | 895.8 ± 46.2 | |

Read naively that says the router costs 24.7% while contention costs only
7%. **That reading is wrong, and the error was mine.** The routed arm had a
`curl` poll loop running against `/v1/replicas` every second to check load
balance — ~80 short-lived `curl`+`python3` processes spawned during a 30 s
benchmark, on the same host as both the client and the router. The direct
arm had no such instrumentation. **The measurement designed to test the
router handicapped only the router.**

Re-measured without polling, the routed number is 1087–1149 tok/s, i.e.
**level with the unrouted 1146–1190**. The router is not the bottleneck;
the earlier −24.7% is retracted.

The load-balance question it was meant to answer resolves cleanly anyway:
per-round completed counts were `[97, 96]` and `[91, 102]` after warmup.
Least-outstanding balances fine. (The first round's `[65, 128]` is
warm-up, and the `inflight` samples that looked lopsided were `sort -u`
output — lexicographic, not chronological. Both were misreads on my part.)

### 14b. Two real router inefficiencies, fixed

Found by reading `_forward` rather than by the benchmark:

- **A blocking `print(..., flush=True)` on every request**, executed from
  inside the asyncio event loop on the completion path — stalling the loop
  that is concurrently proxying every other in-flight request. Now behind
  `LLADA_ROUTER_LOG` (default off).
- **A full JSON parse and re-encode of every response body.** `_forward`
  did `await resp.json()` and handed the dict to `JSONResponse`, which
  re-serialised it — reproducing bytes the replica had already produced,
  on the event loop, per request. The router never inspects the body.
  Now passed through as raw bytes with the upstream content-type.

Both are correct independent of measurement: removing a redundant
serialise/deserialise pair and a synchronous write from an async hot path
is right whether or not a noisy harness resolves it.

**Measured effect, cleanest comparison** (§12 vs a fresh single-server,
3-rep, no-polling sweep — same protocol, differing only in router code):

| Concurrency | old router (§12) | fixed router | Δ |
|---:|---:|---:|---:|
| 32 | 855.4 ± 16.2 | **871.6 ± 23.9** | +1.9% |
| **64** | 897.9 ± 55.3 | **954.5 ± 52.8** | **+6.3%** |
| 128 | 601.4 ± 80.2 | 599.3 ± 129.3 | −0.3% |

Directionally positive, largest where the router works hardest, but the
spreads overlap — **not significant**, same as §11's RoPE result. An
isolated A/B of the logging flag alone (6 reps/arm, interleaved) gave
log-off 1149.3 ± 141.3 vs log-on 1087.2 ± 126.7: **+5.7%, also inside
noise**. The honest summary is "principled fixes, ~2–6%, not separable
from a ±15% harness".

**Updated DP peak: 954.5 tok/s at concurrency 64**, giving 954.5 / 656.9 =
**1.45×** a single GPU — up from §12's 1.37×, still short of 2×, and the
residual is now attributable to the 0.93× host/GPU contention measured
above rather than to the router.

### 14c. Autotuner introspection — root-caused and fixed

Every tuner line printed `shmem=NoneB occ=None%` (§2). Two independent
breakages, both swallowed by a bare `except Exception` that discarded the
error and returned all-`None`:

- `profile_config` never passed `SILU_EPILOGUE`, from the moment that
  fusion landed;
- it was still passing `a_scale_ptr`/`b_scale_ptr`, `stride_bse`/
  `stride_bsn` and `use_fp8_w8a8`/`use_int8_w8a16` after `80a7c38`
  removed them from the kernel as dead vLLM inheritance.

Either raises `TypeError`. Fixed both, and the caller now surfaces an
introspection failure **once, loudly**, instead of printing `None` forever.
The occupancy formula was also wrong: it divided the SM register file by
the *thread* count, which is a registers-per-thread budget, not a block
count. It now divides by `n_regs × threads_per_block`.

Now reports `shmem=24576B occ=25%` where it reported nothing.

Enabling it immediately exposed a third bug it had been hiding: a
small-shmem config printed **`occ=400%`**. The block-count cap ignored
the warps-per-SM limit — an SM holds 64 warps, so at 8 warps/block it
can never hold more than 8 blocks regardless of how little shared memory
each uses. Both that limit and the hardware 32-blocks/SM cap are now
applied; the same config reports 100%.

**With a caveat recorded in the code**: Triton 3.1's warmup metadata does
not expose `n_regs`, so only the shared-memory limit applies and the
number is an **upper bound**. Against ncu on the same config it reports
25% where ncu measures 12.4% — because registers (234/thread) bind first
and are invisible here. It is "shared memory does not prevent N blocks",
not achieved occupancy.

### 14d. Device-keyed tuning configs

Tile shapes are hardware-specific — the tuner's own docstring says to run
it per GPU, and the H100 and A6000 winners differ at every M — but the
output was a single unkeyed `moe_tune_config.json`, so a config tuned on
one card loaded silently on another with nothing in the filename to say
so. `dInfer/configs/` already keys by `device_name=`; this now does the
same:

1. `moe_tune_config.device_name=<GPU>.json` — preferred
2. `moe_tune_config.json` — legacy, still honoured, but **warns** that it
   cannot be verified against the running GPU

The H100 config is committed under its keyed name.

### 14e. Final validation from a clean checkout

The server was reset to the committed HEAD (`158f7f5`), the legacy unkeyed
`moe_tune_config.json` moved aside so only the device-keyed file is
present, and everything re-run:

- **Regression suite 9/9** — `test_fusions`, `test_fused_silu_epilogue`,
  `test_variable_length_batch`, `test_num_logits_slice`,
  `test_moe_align_block_size`, `test_select_transfer_indices`,
  `test_threshold_decoding`, `test_router`, `test_oom_retry`.
- **Fused RoPE bit-exact** at all 11 shape/layout cases, kernel 1.81–2.57×.
- Device-keyed config loads without the legacy-file warning.

### 14f. README corrections

Four claims the H100 data contradicts, all now fixed at the source rather
than only in this log:

- **Kernel profile** — "81% of theoretical weight-streaming bandwidth /
  no kernel headroom left" replaced with the per-hardware split (A6000
  near a streaming wall; H100 at DRAM 14.5%, L2 79.7%, compute 49.6%).
- **`steps_per_block`** — reframed from a mandate at `≥ block_length/2` to
  a *floor*, with `≥ block_length` recommended, citing the 5 deterministic
  collapses that ratio 0.5 produces and ratio 1.0 removes.
- **Multi-GPU / TP latency** — the "6.15× vs single-GPU" claim was against
  the *unoptimized* baseline. Replaced with the measured PCIe result: tied
  at concurrency 1 / 128 tokens, and 5–6× *worse* at GSM8K length.
- **Fused QKV** — the A6000 rejection now notes the H100 numbers flip it
  (faster at every size, bit-exact at M≥256), and states why it still is
  not enabled: `lm_head` dominates the cuBLAS time so the end-to-end gain
  is ~0.5%, below the noise floor.

---

## 15. Quantization accuracy, re-measured paired

§6 compared BF16/INT8/FP8 at n=50 and §9 then proved that meaningless
here: ~17% of GSM8K flips under any numerical perturbation, so a
2-question marginal difference is noise. Re-measured **paired against
BF16 on the identical questions**, with McNemar rather than a marginal
comparison. BF16's arm is the §9e n=1000 run (same config: `steps=1024
block_length=64 threshold 0.9/0.4`), so all arms share the seeded
question order.

| Arm | n | BF16 | quantized | Δ | churn | McNemar p |
|---|---:|---:|---:|---:|---:|---:|
| **INT8** (packed, fused W8A16) | 200 | 71.0% | **68.5%** | −2.5 pt | 17.5% | **0.50** |
| **FP8-E4M3** (packed, no fused kernel) | 100 | 73.0% | **70.0%** | −3.0 pt | 15.0% | **0.61** |

**Neither is significant.** INT8's 35 discordant pairs split 15/20 and
FP8's 15 split 6/9 — both indistinguishable from a coin flip. The
measured churn (17.5% and 15.0%) matches §9's 16.7% at n=1000, confirming
that figure is a property of the checkpoint rather than of one experiment.

What this does and does not license:

- **It does not show quantization is free.** Both point estimates are
  negative (−2.5, −3.0) and §6's n=50 run also came out −4.0 for FP8. A
  consistently negative direction across three independent samples is
  weak evidence of a small real cost, not evidence of none. Resolving a
  2–3 point effect needs n≈1000 (§9), which INT8 could reach and FP8
  cannot in reasonable wall-clock without a fused kernel — it runs ~3×
  slower per question, which is why its arm stops at n=100.
- **It does show the n=50 figures in §6 cannot be used to choose between
  these modes.** "INT8 is free, FP8 costs 4 points" was an artifact of
  reading marginals at a sample size this checkpoint's routing noise
  swamps.

The deployment recommendation is unchanged and rests on memory and speed,
not accuracy: INT8 packed+fused saves 42.4% resident memory (§6) at
parity-or-better speed; FP8 saves the same memory but has no fused kernel
yet, so it is 3× slower per question.

---

## 16. Why the A6000's INT8 speed win does not transfer

The README's historical A6000 table reports `INT8 + fused W8A16` at **5.8
s/question on GSM8K against BF16's 7.2 s — 19% faster**, with 42% less
memory. That is a real result and it was the basis for treating INT8 as a
speed optimisation, not only a memory one.

**It reverses on H100.** §6 measured, same config, same harness:

| | A6000 | H100 |
|---|---:|---:|
| BF16, s/question | 7.2 | 3.4 |
| INT8 + fused W8A16, s/question | **5.8** | **4.2** |
| INT8 vs BF16 | **1.24× faster** | **0.81× — 24% slower** |

Same code, opposite sign. §10's profiling explains it without needing a new
experiment:

| | DRAM throughput | Binding constraint | What halving weight bytes does |
|---|---:|---|---|
| A6000 | 66–68% | close to the DRAM wall | relieves it → faster |
| H100 | **14.5%** | **L2 (79.7%)**, not DRAM | nothing, while the W8A16 dequantize still costs → slower |

**INT8 buys speed only when the kernel is bandwidth-starved.** The A6000 is;
the H100 is not. This is the same measurement that overturned the "81% of
weight-streaming bandwidth / no kernel headroom" claim — once DRAM is known
to sit at 14.5%, INT8 losing on H100 is the predicted outcome rather than an
anomaly, and the two findings corroborate each other.

Consequences:

- **The memory saving is hardware-independent** (−42.4%, §6) and holds on
  both cards. The *speed* claim is not portable and should never have been
  stated without the hardware attached.
- This is a third instance of the same pattern in this log — a conclusion
  correct on A6000 that inverts on H100, alongside the kernel-profile claim
  (§10) and fused QKV (§10, README). The common cause is that A6000 is
  bandwidth-bound where H100 is not, so every optimisation that trades
  compute for bytes changes sign between them.
- It also reframes §8d: co-locating INT8 replicas lost to BF16 not because
  quantization is broken under contention, but because on this hardware INT8
  is already the slower arm before contention is added.

---

## 17. Quantization accuracy across every measurement taken

§15 measured each arm once. Pooling every accuracy measurement this project
has of INT8 and FP8 — including the A6000 historical figure — separates the
two formats in a way no single run does.

### INT8: three measurements, scattering around zero

| Measurement | BF16 | INT8 | Δ |
|---|---:|---:|---:|
| A6000, n=50 (README historical) | 74.0% | 76.0% | **+2.0** |
| H100, n=50 (§6) | 74.0% | 74.0% | **0.0** |
| H100, n=200 paired (§15) | 71.0% | 68.5% | **−2.5** (p=0.50) |

Positive, zero, negative — across two different GPUs. **That scatter is what
"no real effect" looks like.** The A6000 run that put INT8 one question ahead
was noise, and so is our n=200 run that puts it 5 questions behind; the
README's original text already called the +2.0 "noise on the accuracy axis",
which was the correct read at the time and survives the extra data.

**Conclusion: INT8 has no measurable accuracy cost.** This is the strongest
statement any arm here supports, and it is stronger than §15 alone could
make, because it rests on three independent samples rather than one.

### FP8: two measurements, both negative

| Measurement | BF16 | FP8 | Δ |
|---|---:|---:|---:|
| H100, n=50 (§6) | 74.0% | 70.0% | **−4.0** |
| H100, n=100 paired (§15) | 73.0% | 70.0% | **−3.0** (p=0.61) |

Neither is significant on its own, and two samples is not much. But **both
point the same way, and the point estimate is stable (−3 to −4)** — a
different pattern from INT8's scatter. That is weak evidence of a small real
cost, not proof of one, and it is the honest reason to prefer INT8 on
accuracy grounds even though no individual p-value clears 0.05.

Settling it needs n≈1000 (§9), which FP8 cannot reach in reasonable
wall-clock without a fused kernel. Until then the defensible statement is
"INT8 is measurably free; FP8 is probably slightly lossy but unconfirmed" —
not "both are indistinguishable", which §15 in isolation implied.

### Why this matters for the recommendation

The two formats are equal on memory (−42.4%, hardware-independent) and INT8
wins decisively on speed (fused W8A16 vs ~3× slower dequantize-per-access).
Accuracy was the one axis where they looked tied. Pooled across every
measurement, INT8 is clean and FP8 is unresolved-but-consistently-negative,
so accuracy stops being a tie and becomes a third, weaker argument in the
same direction.

---

## 18. Open items from this session

- [x] `moe_tune_config.json` — **DONE (§14d).** Device-keyed lookup added
      (`moe_tune_config.device_name=<GPU>.json`), legacy name still honoured
      with a warning, H100 config committed under its keyed name.
- [x] Autotuner `shmem=None occ=None` — **DONE (§14c).** Two breakages
      (missing SILU_EPILOGUE; stale removed kernel params) both hidden by a
      bare except. Fixed; failures now surface once, loudly. Occupancy is an
      upper bound until Triton exposes n_regs at warmup — noted in code.
- [x] **Fuse RoPE's `rotate_half`** — DONE (§11). Kernel 2.55x, ~+4%
      end-to-end (p~0.14, direction consistent). Bit-exact, 9/9. Default-on.
- [~] **Fuse the MoE top-k combine into GEMM2's epilogue** (§10) —
      **analysed and deliberately NOT done.** It is the biggest remaining
      target on paper: the `[M, top_k, K]` intermediate is 67 MB written +
      67 MB read per layer through the L2 ncu identifies as the bottleneck.
      Fusing it requires GEMM2 to accumulate into `[M, K]` with
      `atomic_add`, because `moe_align_block_size` orders rows expert-major
      so one block cannot see all `top_k` experts of a token — and
      reordering token-major would destroy the weight reuse the whole
      kernel is built on. Atomics make the result **non-deterministic
      run-to-run**, and this project's method depends on determinism: it
      was used three times in this session alone to establish that results
      were real (§4c, §8c, §9e). Trading reproducibility for ~3% is the
      wrong trade. Revisit only with a deterministic reduction scheme
      (e.g. a fixed-order two-pass split-K), not with atomics.
- [x] README corrections — **DONE (§14e).** Kernel profile, steps_per_block
      guidance, TP-for-latency framing, and the QKV rejection note all
      corrected at the source (§14f).
- [x] `~/venv_new` — removed from the box.
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
- [ ] High-batch instability (§12) — measurement sd grows from 3.5 tok/s at
      B=32 to 54.4 at B=128, and DP p50 swings 15.8-20.8 s between identical
      runs. Less predictable, not just slower. Undiagnosed.
- [x] DP sub-linear scaling — **DIAGNOSED (§14a/b).** Not the router and
      not load balancing (both measured). Two real router inefficiencies
      fixed anyway (blocking print, redundant JSON re-encode) for +2-6%,
      taking DP to 1.45x. The residual is host/GPU contention: two replicas
      running concurrently WITHOUT any router reach only 0.93x of perfect.
      Also retracts a -24.7% 'router cost' figure that was an artifact of
      my own polling instrumentation.
- [x] README Multi-GPU / TP-for-latency framing — **DONE (§14e).**
- [ ] Quantized-DP's counter-intuitive result (§8d: INT8 slower than BF16
      under co-location, despite streaming fewer bytes) — two candidate
      explanations offered, neither confirmed. Settling it needs ncu
      attached to two co-resident processes on one GPU, which the
      admin-gated counters (§10) make awkward and which would not change
      the deployment recommendation either way: §8d already established
      that one BF16 replica per GPU beats every co-location variant
      tested. Left open as a curiosity, not a blocker.
- [x] fp32 router (§9) — **tested and rejected.** No accuracy effect at
      n=1000 (p=0.757); free at throughput (−0.3%) but ~12% cost on the
      single-stream path. Flag kept default-off with the negative result
      documented so the idea isn't re-investigated from scratch.
- [x] Quantization accuracy re-measured **paired** (§15): INT8 −2.5 pt at
      n=200 (p=0.50), FP8 −3.0 pt at n=100 (p=0.61). Neither significant;
      §6's n=50 marginals cannot be used to choose between modes. Direction
      is consistently negative across three samples, so a small real cost
      is not excluded — resolving it needs n≈1000, which FP8 cannot reach
      without a fused kernel.
- [ ] §8e's TP accuracy figure (n=50) still inherits the §9 sampling
      problem and would need the same paired treatment to mean anything.
- [x] FP8 accuracy re-run — done at n=100 (§15), paired. Capped there
      rather than n≥1000 because without a fused kernel FP8 runs ~3×
      slower per question; n=1000 would be ~3.5 h.
- [ ] FP8 fused kernel — currently dequantize-per-access only, so §6's
      12.5s vs 4.2s/question is not apples-to-apples. This is a real
      engineering project (an FP8xBF16 or native FP8 tensor-core Triton
      kernel), not a session task; it is the same milestone INT8's fused
      W8A16 kernel already represents. Out of scope here, listed so the
      latency comparison is not mistaken for a property of FP8 itself.
