# Historical profiling study

Rerun one fixed workload across five points in the engine's history and compare
the GPU traces. The only variable is the code version.

Every milestone below was chosen by inspecting the actual history and verifying
the optimization is present in the tree at that commit — not by reading commit
messages. The verification is reproduced at the bottom.

---

## The milestones

Each commit adds **exactly one** optimization layer to the previous one. That
monotonicity is what makes the comparison attributable: a difference between
two adjacent traces has one candidate cause, not five.

| # | Label | Commit | Date | Adds | Expected trace difference |
|---|---|---|---|---|---|
| — | `baseline` | *(control arm, see below)* | frozen 2026-07-03 | nothing — the unoptimized reference | 64 sequential expert GEMMs per layer; no KV reuse; full-sequence forward every step |
| 1 | `m1_fused_moe_kv` | `a5f6ebe` | 2026-08-03 | Triton **fused MoE** + **block-wise KV cache**, with the cache-collapse bug fixed | **kernel count collapses** — 64 expert launches per layer become 1 grouped GEMM; prefix K/V stops being recomputed |
| 2 | `m2_host_sync` | `5b2220d` | 2026-08-05 | `moe_align_block_size` and `select_transfer_indices` **vectorized** | **GPU idle gaps shrink** — ~128 host-device syncs per MoE call and a per-row `.item()` per decode step disappear; this is the milestone that shows up as *timeline density*, not kernel duration |
| 3 | `m3_mem_traffic` | `b954121` | 2026-08-16 | **narrowed `lm_head`** + **fused SiLU epilogue** | **memory traffic drops** — the 2·EI-wide MoE intermediate is never materialized (~1.2 GB round-trip per forward), and the widest GEMM stops computing discarded rows |
| 4 | `m4_launch_count` | `c2196ba` | 2026-08-25 | **fused RMSNorm + decode tail** | **launch count drops** — 8 kernels per norm × 65 norms per forward become 1 each; the decode tail's three passes over a 157k-wide tensor become one |
| 5 | `m5_rope_final` | `1b21e25` | 2026-08-28 | **fused RoPE** — final engine state | `aten::cat` + `aten::neg` vanish from the timeline; last of the elementwise fusions |

### The baseline is a control arm, not a checkout

`dminfr/reference/model.py` has **three commits in its entire history**, the last
of which is the rename in `1fca44f`. It has been functionally frozen since
2026-07-03.

So the workload runs `--mode both`, which executes the frozen baseline *and* the
optimized path in the same process, same GPU, same run. That gives a **baseline
arm inside every milestone trace**. If the baseline number drifts between
milestones, the environment changed — not the code. Use it as your control.

### Backup milestones

| Backup | Commit | Date | Replaces | Why it is the fallback, not the pick |
|---|---|---|---|---|
| `b1` | `deb3170` | 2026-08-04 | M2 | Contains only the `moe_align` vectorization, not `select_transfer_indices`. A cleaner *isolation* but a weaker *effect* — use it if M2's two changes need separating. |
| `b2` | `d27ca5c` | 2026-08-16 | M3 | Contains only the `lm_head` narrowing, without the SiLU epilogue. Splits M3's two memory-traffic wins if their contributions must be attributed separately. |
| `b3` | `59a1586` (HEAD) | 2026-08-28 | M5 | Current `main`. Functionally identical to M5 for this workload — the only engine change is device-keyed config loading, which the script normalizes. Use if M5 fails to build. |

### Commits deliberately **not** selected

- Anything before `a5f6ebe` — the KV cache had a **correctness collapse**
  (`prime`/`commit` did not see the full remaining sequence). The engine runs,
  but generations degrade. Profiling it would produce a trace of a broken engine.
- `bff9b41`, `da06154`, `7300a4c` — real work, but each bundles changes that do
  not alter single-stream GPU execution (autotuner redesign, TP/EP plumbing,
  server-side batching is irrelevant at batch 1).
- The July 21–22 run (`fahhh`, `X(`, `tttt`, `siliana`, …) — experimental and
  self-described as unstable.
- `a32e4ce`/`89855d5`, `ac49a43`/`2d093ae`, `c86d451`/`b3525d3`,
  `288d732`/`e941b55` — all four are **added-then-reverted** experiments
  (torch.compile attention, CUDA graphs, compiled activation, remask threshold).
  Profiling a reverted path measures a dead end.
- Documentation-only and restructuring commits.

---

## Requirements

| | |
|---|---|
| GPU | 1× NVIDIA GPU, ≥24 GB, sm_80+ (validated on H100 PCIe 80 GB and RTX A6000) |
| Model | `inclusionAI/LLaDA-MoE-7B-A1B-Instruct`, ~15 GB, in `weights/` |
| Python | 3.10+, `torch` with CUDA, `triton`, `transformers==4.53.2` |
| Profiler | Nsight Systems (`nsys`) — **optional**; without it the script still records timings |
| Disk | ~1–2 GB per milestone for traces |

`nsys` is usually already present with the CUDA toolkit. If not:

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb && sudo apt-get update
sudo apt-get install -y nsight-systems
```

> Nsight **Systems** needs no special permissions. Nsight **Compute** (`ncu`)
> does — GPU performance counters are admin-gated, so `ncu` must run under
> `sudo`. This study uses `nsys` precisely to avoid that.

---

## Running it

**Use `--worktree`.** `git checkout <old-commit>` deletes `profiling/`, because
that directory does not exist in history — the naive workflow removes the script
it is about to run. The `--worktree` flag materializes the commit in a private
worktree and leaves your checkout alone.

```bash
# from the repo root, on any branch, in any order
./profiling/profile_milestone.sh m1_fused_moe_kv  --worktree a5f6ebe
./profiling/profile_milestone.sh m2_host_sync     --worktree 5b2220d
./profiling/profile_milestone.sh m3_mem_traffic   --worktree b954121
./profiling/profile_milestone.sh m4_launch_count  --worktree c2196ba
./profiling/profile_milestone.sh m5_rope_final    --worktree 1b21e25

# optional: current main
./profiling/profile_milestone.sh m6_head
```

Options:

| Flag | Effect |
|---|---|
| `--worktree <commit>` | profile that commit in an isolated worktree |
| `--no-nsys` | skip tracing, record timings only (fast sanity check) |
| `WEIGHT_DIR=...` | weights location (default `<repo>/weights`) |
| `VENV=...` | virtualenv location (default `<repo>/.venv`) |

Clean up worktrees when finished:

```bash
git worktree list
git worktree remove .profiling_worktrees/m1_fused_moe_kv
git worktree prune
```

---

## The workload, and why it is pinned to these values

```
gen_length=128  steps=128  block_length=32
num_runs=3      num_warmup=1  mode=both  batch_size=1 (default)
```

These are **the intersection of flags available at every selected milestone**,
verified against each commit:

- `--mode` exists only from `bff9b41` (2026-07-29)
- `--batch-size` exists only from `b954121` (2026-08-16), so batch size stays at
  its default of **1** — a single-stream profile is also the cleanest signal for
  kernel-level comparison, since server batching does not apply
- everything else (`--weight-dir --gen-length --steps --block-length
  --num-runs --num-warmup`) is present across the whole range

Changing any of these means re-checking availability across all five commits.
Do not edit them casually — the values are load-bearing, not defaults.

### The tuned-config trap

Milestones M1–M4 read `moe_tune_config.json`. HEAD prefers
`moe_tune_config.device_name=<GPU>.json`. **If only the device-keyed file
exists, the older milestones silently fall back to hardcoded kernel tile shapes
while HEAD uses tuned ones** — and the resulting difference would be attributed
to whatever else changed between the commits.

The script mirrors the device-keyed config to the unkeyed name before every run
and records which config was in force under `tuned_config:` in `run_info.txt`.
**Check that line matches across milestones before comparing traces.**

---

## Output

```
profiling/results/<label>/
├── run_info.txt              commit, date, layout, GPU, env, workload, wall time
├── stdout.txt                the benchmark's own timings and speedup table
├── <label>.nsys-rep          the trace (open in the Nsight Systems GUI)
├── <label>_cuda_gpu_kern_sum.csv    per-kernel time, count, mean/min/max
├── <label>_cuda_api_sum.csv         CUDA API calls — launch and sync counts
├── <label>_cuda_gpu_mem_time_sum.csv
└── <label>_cuda_gpu_mem_size_sum.csv
```

`.nsys-rep` files are **not committed** — `profiling/results/` is gitignored,
matching the repo's existing practice for `ncu_deep_report*.txt` and trace JSON.
Keep them locally or attach them to the report.

---

## Metrics to extract

Everything below comes from the CSVs the script already exports, so the report
does not need the GUI:

| Metric | Source |
|---|---|
| Total time, tokens/s, speedup vs baseline | `stdout.txt` (the harness prints its own table) |
| Wall time | `run_info.txt` |
| Kernel count, per-kernel time share | `*_cuda_gpu_kern_sum.csv` |
| Kernel launch frequency, sync count | `*_cuda_api_sum.csv` (`cudaLaunchKernel`, `cudaDeviceSynchronize`, `cudaMemcpy*`) |
| Memory transfer volume and time | `*_cuda_gpu_mem_*.csv` |
| GPU idle gaps, CPU↔GPU serialization | the timeline view in the GUI — this is the one that needs `.nsys-rep` |
| MoE vs attention time split | `*_cuda_gpu_kern_sum.csv`, grouping `fused_moe_kernel` against `flash_fwd`/`sm90_xmma` rows |

---

## What each milestone should demonstrate

| Comparison | The claim it tests | What to look for |
|---|---|---|
| baseline → M1 | Fusing 64 expert GEMMs into one grouped GEMM, plus KV reuse, is the single largest structural win | Kernel count per layer drops by ~64×; `fused_moe_kernel` appears and dominates; prefix recomputation disappears |
| M1 → M2 | Host-device synchronization, not kernel math, was a major cost | Wall time falls with **little change in total kernel time** — the gain is in the gaps. `cudaDeviceSynchronize`/`cudaMemcpy` counts drop sharply |
| M2 → M3 | Intermediate materialization was costing more than the arithmetic | Memory-transfer bytes fall; `fused_moe_kernel` time changes little but the surrounding elementwise/copy kernels shrink |
| M3 → M4 | Launch overhead dominates for small elementwise ops | `cudaLaunchKernel` count drops sharply; many short kernels replaced by few longer ones |
| M4 → M5 | The last elementwise fusion | `aten::cat` / `aten::neg` rows disappear from the kernel summary |
| baseline → M5 | The cumulative result | End-to-end speedup, with the intermediate milestones showing which layer contributed what |

> **On interpreting differences.** The measured throughput noise floor on this
> harness is **~±5%** run-to-run (`docs/h100x2_bench.md` §11). Kernel *counts*
> and *launch counts* are deterministic and can be compared directly; *timings*
> under ~10% should not be treated as a result without repeats. `num_runs=3`
> gives a spread — report it.

---

## How the milestones were verified

Presence of each optimization was checked in the tree at each commit, not
inferred from commit messages:

```
commit    moe  align  lm_head  silu  rmsnorm  rope
a5f6ebe    Y     n       n       n      n       n     <- M1
deb3170    Y     Y       n       n      n       n        (b1)
5b2220d    Y     Y       n       n      n       n     <- M2
d27ca5c    Y     Y       Y       n      n       n        (b2)
b954121    Y     Y       Y       Y      n       n     <- M3
c2196ba    Y     Y       Y       Y      Y       n     <- M4
1b21e25    Y     Y       Y       Y      Y       Y     <- M5
```

Reproduce with:

```bash
for c in a5f6ebe deb3170 5b2220d d27ca5c b954121 c2196ba 1b21e25; do
  printf "%-9s " "$c"
  git show "$c:model_update/fused_moe_triton.py" 2>/dev/null | grep -q SILU_EPILOGUE && printf "silu=Y " || printf "silu=n "
  git show "$c:model_update/fused_ops.py"        2>/dev/null | grep -q _rmsnorm_kernel && printf "rms=Y "  || printf "rms=n "
  git show "$c:model_update/fused_ops.py"        2>/dev/null | grep -q _rope_kernel    && printf "rope=Y " || printf "rope=n "
  echo
done
```

Also verified: `eval/check_time_inference.py` exists at every selected commit;
`vllm` appears only in `fused_moe_triton_raw.py` and `download_vllm.py`, neither
of which the engine or the benchmark imports, so it is **not** an install
blocker for any milestone.

---

## Status

The script has been tested for argument handling, worktree creation and layout
detection across the `1fca44f` restructure boundary — `a5f6ebe` correctly
resolves to `eval.check_time_inference`, HEAD to
`benchmarks.check_time_inference`.

**It has not been run end-to-end**, because that needs a GPU and the weights.
The first real run should be a single `--no-nsys` invocation on HEAD to confirm
the harness executes before committing to the full five-milestone sweep.
