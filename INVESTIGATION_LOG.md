# `model_update` Optimization & Correctness Investigation Log

Full chronological record of what was tried on `model_update` (the
TP+EP, fused-Triton-MoE, KV-cached inference engine), what broke, what we
measured, and why. Two major investigations are covered:

1. **Expert-count reduction** (dynamic top-k, then adaptive/nucleus top-k) —
   tried to cut MoE compute further, measured the accuracy cost, and
   reverted.
2. **`model_update` vs HF correctness gap** — a real accuracy bug found in
   the KV-cache/block-wise generation logic, root-caused and fixed.

`src/` (the unoptimized baseline) is referenced only as a fixed comparison
point throughout — it was never itself under investigation.

---

## Part 1 — Reduced expert activation: dynamic-k, then adaptive/nucleus-k

### 1.1 Motivation

The model's native MoE router activates the **top-8 of 64 experts** per
token on every forward pass. The idea: early denoising steps operate on
mostly-`[MASK]` tokens, where the router's output is less meaningful — so
maybe fewer active experts (e.g. top-5) would cost little accuracy while
cutting MoE compute (and thus wall-clock time) meaningfully.

### 1.2 First attempt: fixed/step-ramped dynamic-k

Implemented as `use_dynamic_experts`/`base_k`/`min_k` in
`model_update/generate.py`: every token in a given denoising step used the
**same** `k`, ramped from `min_k` up to `base_k` (e.g. 5) uniformly across
steps — later steps (more real content) got more experts than earlier steps
(mostly still masked).

**Speed result** (single-GPU, NVIDIA A40-24Q, 32-token generation,
`top-k=5`, `eval/check_time_inference.py`):

| Configuration | top-k | Time (s) | Tok/s | Speedup | Token divergence vs baseline |
|---|:---:|---:|---:|:---:|:---:|
| Baseline (`src/`, unfused, no cache) | 8 | 6.49 | 4.93 | 1.00× | — |
| Optimized (`model_update/`, tuned kernel) | 5 | 1.73 | 18.50 | **3.75×** | 9.38% (3/32 tokens) |

**Important side-finding**: that 9.38% token divergence turned out to be
**constant regardless of top-k** (same 3/32 tokens diverged whether running
top-8, top-5, or top-4). If reduced expert count were the cause, top-8
should have shown ~0% divergence. It didn't — the divergence was actually
coming from **the KV caching itself** changing the attention context
between baseline and optimized (baseline recomputes full attention every
step; optimized only attends over the active block). This was an early,
weaker version of the same class of finding investigated in full in Part 2
below — at this point it was treated as an accepted/expected cost of
caching, not chased further.

**Accuracy result** — MMLU, `fast_dense` backend, dynamic experts + top-5:
**60.0%** vs baseline's **66.0%**. On MMLU-Pro specifically (n=200, later
run): **fixed top-5 scored 38.5% vs static top-8's 40.0%** — a real,
repeatable ~1.5pt cost.

**Root cause of the accuracy cost** (confirmed by re-reading
`MoEBlock.forward`, `model.py:304-309`): `torch.topk` truncation does
**not renormalize** the surviving routing weights — dropping ranks 6-8
deletes their softmax probability mass outright rather than redistributing
it to the kept experts. Whether that's safe depends entirely on whether
*that specific token's* routing distribution is peaked (one expert
dominates — dropping #6-8 costs little real signal) or flat (all 8 carry
meaningful weight — dropping any of them loses real signal). The step-based
ramp had no way to tell peaked from flat; it truncated every token in a
step identically.

### 1.3 Second attempt: adaptive per-token nucleus routing

**Hypothesis**: route per-token, not per-step. Use a nucleus/top-p-style
cumulative-probability threshold so peaked tokens automatically keep fewer
experts and flat tokens keep more/all — instead of a step-based ramp that
can't distinguish them.

```python
vals, ids = torch.topk(routing_weights, cfg.TOPK, dim=-1)   # [T, 8], descending
if nucleus_p >= 1.0:
    keep_mask = torch.ones_like(vals, dtype=torch.bool)      # explicit passthrough
else:
    cum_before = torch.cumsum(vals, dim=-1) - vals            # exclusive cumsum
    keep_mask = cum_before < nucleus_p                         # always keeps slot 0
topk_weights = vals * keep_mask.to(vals.dtype)                 # zero dropped slots, no renorm
```

Staged rollout plan (`.claude/plans/polished-crunching-muffin.md`):

- **Stage 0** — cheap offline validation using an existing routing-mass
  diagnostic (`diagnose_real_activation_pruning.py`), no code changes.
- **Stage 1** — prototype `nucleus_p` on the eager `MoEBlock` path (the
  per-expert loop there is naturally ragged-safe, no kernel work needed to
  get real compute savings). Added a new `fast_dense_eager` server backend,
  `--nucleus-p` in `run_correctness.py`, and nucleus support in
  `diagnose_dynamic_experts.py`'s token-divergence check.
- **Stage 2** — TP/EP determinism check: confirm `keep_mask` is bit-identical
  across TP ranks (a silent per-rank mismatch here would corrupt the MoE
  all-reduce in a much harder-to-debug way than the old scalar-k ramp,
  which was rank-uniform by construction).
- **Stage 3** (never reached) — ragged Triton kernel for real production
  speedup (est. 3-5 days of work).
- **Stage 4** (never reached) — tuning + perf validation.

**Stage 1/2 results — the go/no-go gate that killed the idea:**

Token divergence from a dense (uncached, full-recompute) baseline, at
comparable average active-expert counts:

| Config | Avg divergence | Trial-by-trial pattern |
|---|---|---|
| Fixed top-5 (`--base-k 5 --min-k 5`) | 0.78% | `[2.34, 0.00, 0.78, 0.78, 0.00]` |
| Nucleus p=0.12 | 0.78% | **identical, trial for trial**, to fixed top-5 |
| Nucleus p=0.08 | 0.94% | worse than fixed-5 |
| Nucleus p=0.05 | 1.88% | worse than fixed-5 |
| Nucleus p=0.95, p=1.0 | 0.00% | threshold **unreachable** — top-8's cumulative mass is only ~13-22% of the total distribution, so p=0.95 never triggers any pruning at all; not a real pass, just a no-op |

**Root cause of the negative result**: this checkpoint's 64-way MoE router
is close to **uniform**. Average top-1 routing weight measured at only
~1.7-5%, barely above the 1.56% a purely random router would produce. The
entire premise of nucleus routing — that some tokens are peaked (safe to
prune aggressively) and others flat (need all 8) — requires real per-token
*variance* in peakedness to have anything to exploit. With a near-uniform
router, that variance barely exists, so nucleus thresholding just converges
to picking approximately the same `k` every time — i.e., it degenerates to
plain fixed-k with zero advantage, and gets strictly *worse* than fixed-5 at
any more aggressive threshold.

**Decision**: stopped after Stage 1/2. Stage 3 (the actual kernel work) was
never started — the token-divergence data showed it wouldn't beat
fixed-top-5's existing (already-negative) accuracy/speed tradeoff on this
model, matching exactly the failure mode the original staged plan was
designed to catch cheaply before investing days in kernel work.

### 1.4 Final decision: revert to static top-8

Commit `28ab20b` — *"Decline dynamic/adaptive expert routing; revert to
static top-8"*:

> Both the step-based dynamic-k ramp and the per-token nucleus routing
> experiment were evaluated and found not worth it: at 2x A6000 TP+EP,
> static top-8 alone already gets **4.54x speedup** over baseline, and
> neither alternative improved accuracy over it (nucleus degenerated to
> fixed-k behavior on this checkpoint's near-uniform router; dynamic-5 cost
> 1.5pt on MMLU-Pro for no speed benefit that static top-8 didn't already
> have).

The deciding logic: **Triton Fused MoE + block-wise KV caching alone**
already delivered a 4.54x speedup with the model's native top-8 routing
untouched. Neither expert-reduction scheme improved on that — dynamic-5
cost real accuracy for no additional speedup over what caching+fusion
already provided, and nucleus routing (the more sophisticated, theoretically
better-motivated approach) turned out to offer no advantage at all on this
specific checkpoint's near-uniform router. Removed entirely rather than kept
as a disabled option, to avoid maintaining dead/misleading code paths:
`dynamic_k`/`nucleus_p` removed from `model_update/model.py` and
`generate.py` (model always routes to `cfg.TOPK` now), the
`use_dynamic_experts`/`base_k`/`min_k`/`nucleus_p` request fields and
`fast_dense_eager`/`dyn_experts` backends removed from `src/server.py`, the
corresponding CLI flags removed from `run_correctness.py` and
`check_time_inference.py`, and the three diagnostic scripts that existed
solely to test this feature (`diagnose_dynamic_experts.py`,
`diagnose_nucleus_tp_consistency.py`, `diagnose_real_activation_pruning.py`)
deleted. README updated to match — now documents **two** stacked
optimizations (Triton Fused MoE, Block-wise KV Caching), not three.

This is a clean, documented negative result: the nucleus_p code (Stage 1,
eager-path only) was a legitimate approach that simply didn't pay off on
*this* checkpoint's router — worth revisiting only if a future checkpoint
has a more peaked router.

---

## Part 2 — `model_update` vs HF correctness gap: the KV-cache collapse bug

With expert-count reduction abandoned and `model_update` left running
native static top-8, the next step was validating `model_update`'s output
*quality* against the official HuggingFace reference implementation
end-to-end (explicitly out of scope: `src/`, per direction to focus only on
`model_update` vs HF).

### 2.1 Symptom: large accuracy gap vs HF

Running the correctness harness (`eval/correctness/run_correctness.py`)
with CoT prompting on identical, same-seeded MMLU / MMLU-Pro subsets:

| Backend | MMLU | MMLU-Pro |
|---|---|---|
| `model_update` (`fast_dense`) | 59.0% | 28.0% |
| HF reference | 58.0% | 46.0% |

The MMLU-Pro gap (28% vs 46%) was far too large to be sampling noise, and
survived a deliberate sanity check against a sample-size mismatch (n=100 vs
n=50): both runs shared the same `--seed 42` deterministic shuffle, so they
had an identical first-50-item subset regardless of `--limit`. Matching the
two runs item-for-item on that shared subset confirmed the gap was real, not
an artifact of comparing different-sized samples.

### 2.2 Narrowing it down: raw transcripts

`run_correctness.py` was extended with `--save-transcripts` to dump every
question's raw generated text (not just the parsed answer letter), since
the aggregate accuracy number alone couldn't distinguish "wrong reasoning"
from "reasoning never happened."

Transcript inspection revealed the actual mechanism: a large fraction of
`model_update`'s CoT responses collapsed to a bare

```
Final Answer: G
```

(exactly 15 characters, zero reasoning) despite the system prompt
explicitly asking for step-by-step reasoning first. Counts of responses
under 100 characters:

| Backend | MMLU (n=50) | MMLU-Pro (n=50) |
|---|---|---|
| `model_update` | 36/50 (72%) | 22/50 (44%) |
| HF | 0/50 | 0/50 |

HF never exhibited this pattern, on either task.

### 2.3 Isolating the cause: cache vs. no-cache, identical weights

`eval/diagnose_cache_vs_dense.py` was built to eliminate every confound
except caching: it runs the **same `model_update` model class and
weights** through two generation paths on the identical question —

1. **Cached** — `model_update.generate.generate_cached` (the production
   path).
2. **Dense** — a from-scratch reimplementation that recomputes the *entire*
   sequence on every denoising step (no KV cache at all), otherwise the
   identical algorithm.

Run on 3 known-bad MMLU-Pro questions, single GPU (so TP/EP was ruled out
too):

| item-idx | cached | dense |
|---|---|---|
| 4  | `"Final Answer: G"` (15 chars, wrong) | 924 chars of full reasoning |
| 12 | `"Final Answer: J"` (15 chars) | 1254 chars of full reasoning |
| 17 | `"Final Answer: B"` (15 chars, wrong) | 1018 chars of full reasoning |

3/3: collapse only occurred with caching. This conclusively pinned the bug
to `model_update`'s KV-cache / block-wise generation logic
(`generate.py`'s `generate_cached`/`_generate_block_cached`, and/or
`model.py`'s `KVCacheBuffer`), and ruled out model weights, TP+EP, and any
other codebase difference as the cause.

### 2.4 Finding the exact divergence point

Both paths use `temperature=0` (pure greedy argmax), so any difference
between them is a genuine, deterministic computational discrepancy — not
sampling noise.

`eval/diagnose_step_divergence.py` traced both paths step-by-step through
just block 0 (16 steps), printing the partially-decoded block content after
every single step. The divergence was already visible at **step 0**, before
any tokens had even been revealed:

- **Dense**, step 0: predicts `"The determine"` — real content.
- **Cached**, step 0: predicts `<|endoftext|>` at two positions, with
  enough confidence to be selected for reveal in the very first step.

By step 15, the cached path had filled almost the entire block with
`<|endoftext|>`, decoding down to `"Final Answer: G"`.

### 2.5 Root cause

`generate_cached` primed the KV cache like this
(`model_update/generate.py`):

```python
model(prompt_ids, position_offset=0, cache_buffer=cache_buffer, write_pos=0)
```

This computed and froze the prompt's K/V using **only the prompt**
attending to itself — the model had no way to know a `gen_length`-token
masked continuation was coming. The dense/reference path's very first
forward call, by contrast, always passes the *full* sequence (prompt + the
entire mask-filled generation region), so the prompt's own hidden states —
and therefore what every later position reads back as "prompt K/V" — are
always computed with full knowledge that a masked continuation follows.

The model was trained under exactly that full-context, bidirectional view.
Caching K/V computed from a truncated view (prompt in isolation) is
out-of-distribution for it — it reads "nothing follows me" and collapses to
EOS almost immediately.

The same pattern recurred at every block boundary. Each block's "finalize"
forward pass (which commits that block's K/V once it's fully unmasked) only
processed that block's own tokens:

```python
finalized_ids = x[:, block_start:block_end]   # misses future mask context
```

— again omitting the mask-filled continuation still to come, recurring once
per block (8 times for the CoT config).

### 2.6 Fix

Both commit-time forward passes now run over the full remaining sequence
(everything generated so far + MASK placeholders through the end of
`gen_length`), matching what the dense/reference path always sees. Only the
actually-final portion gets `cache_buffer.commit(...)`-ed; the
provisionally-computed K/V for not-yet-reached future blocks is simply
overwritten again once those blocks are processed later — the same
"redundant overwrite" pattern the code already relied on elsewhere.

Changed in `model_update/generate.py` (commit `a5f6ebe`):

- **Priming**: `model(prompt_ids, ...)` → `model(x, ...)` (full
  `prompt + all-MASK` sequence).
- **Per-block finalize**: `finalized_ids = x[:, block_start:block_end]` →
  `remaining_ids = x[:, block_start:]` (this block + all remaining
  still-masked future blocks).

This does **not** remove or weaken the KV cache — the per-step denoising
loop (the bulk of all forward calls: 15 of every 16 steps per block) is
unchanged and still only recomputes the active block against the cached
prefix. Only the 9 boundary calls (1 prime + 8 finalizes, for an 8-block CoT
run) got more expensive, since they now process more tokens each.

### 2.7 Verification

Re-running `diagnose_cache_vs_dense.py` on the same 3 previously-collapsed
questions after the fix:

| item-idx | cached (post-fix) | dense |
|---|---|---|
| 4  | 1329 chars, full reasoning, ends `Final Answer: A` | 924 chars, full reasoning |
| 12 | 1199 chars, full reasoning | 1254 chars, full reasoning |
| 17 | 1055 chars, full reasoning | 1018 chars, full reasoning |

All 3/3 collapses resolved — cached output length and quality now closely
track dense on every case tested.

Note: `diagnose_step_divergence.py`'s cached trace, if re-run, still
reproduces the old collapse — that script hand-reimplements the priming
call inline for instrumentation purposes rather than calling the real
`generate_cached`, so it was never touched by the fix. It was a one-off
tracer used to *find* the bug; the authoritative post-fix verification is
`diagnose_cache_vs_dense.py`, which does call the real, patched code path.

### 2.8 Full-benchmark verification

Re-ran the full MMLU-Pro correctness comparison
(`eval/correctness/run_correctness.py --task mmlu_pro --seed 42 --limit 50`,
same config used throughout this investigation) against `model_update`
post-fix:

| Backend | MMLU-Pro (n=50) |
|---|---|
| `model_update`, pre-fix | 28.0% |
| `model_update`, post-fix | **40.0%** |
| HF reference | 46.0% |

The aggregate gap closed from 18pt to 6pt (28%→40% vs a 46% target) — most
of the accuracy loss identified in §2.1-2.2 is confirmed fixed at the
whole-benchmark level, not just on the 3 individually-diagnosed questions.
The post-fix 40.0% also matches the static-top-8 MMLU-Pro figure (40.0%,
n=200) measured independently in the Part 1 expert-routing investigation —
a useful cross-check that the fixed cached path now performs in line with
what static top-8 should give.

The remaining 6pt gap to HF has not been root-caused — it may be ordinary
run-to-run variance at n=50, or a smaller residual approximation from the
KV-cache design (see §2.6: the committed prefix is still only refreshed at
block boundaries, not every step, unlike the dense/HF reference). MMLU (not
just MMLU-Pro) has not yet been re-run against this fix.

### 2.9 Residual gap corroborated on a second task (GSM8K), and localized to long generations

Much later (separate session, see `README.md`'s "Adaptive Decoding"
section for the surrounding context), the same `model_update`-vs-HF
comparison was re-run on GSM8K (`gen_length=1024, block_length=64` — the
paper's own long-generation config, 16 blocks per response instead of
MMLU-Pro's 8) using the identical chat-templated harness for both
backends:

| Backend | GSM8K (n=50, seed=42) |
|---|---:|
| `model_update` (`fast_dense`, chat-template, no threshold) | 68.0% (34/50) |
| HF reference (same harness) | 74.0% (37/50) |

Same ~6pt gap, different task — already suggestive this is systematic
rather than MMLU-Pro-specific noise. `analyze_length_vs_gap.py` (built
during the earlier investigation specifically to test the block-commit-
staleness hypothesis but never actually run until now) was applied to
matched `--save-transcripts` output from both backends on this GSM8K run,
bucketing accuracy by raw response length:

| Response length (chars) | n | `model_update` acc | HF acc | gap |
|---|---:|---:|---:|---:|
| 100-300 | 6 | 83.3% | 83.3% | +0.0pt |
| 300-600 | 26 | 80.8% | 73.1% | -7.7pt |
| 600-1200 | 12 | 66.7% | 83.3% | +16.7pt |
| 1200+ | 6 | **0.0%** | 50.0% | **+50.0pt** |

The gap grows sharply and consistently across the three buckets that have
enough response length to matter (300-600 → 600-1200 → 1200+: -7.7pt →
+16.7pt → +50.0pt) — `model_update` is not uniformly worse than HF, it is
specifically much worse on the longest responses, the ones that cross the
most block-boundary commits. Most strikingly: on the 6 longest responses
(1200+ chars), `model_update` got **zero correct** while HF got half
right. This is strong corroborating evidence for the block-commit-
staleness hypothesis from §2.8 — not a controlled per-step logit trace
(the extreme buckets are only n=6, so treat this as strong support, not
proof), but a second, independent, much more direct signal than the
original single MMLU-Pro accuracy delta.

**Decision: documented as the working explanation, not pursued further.**
Closing this gap would mean refreshing the committed KV prefix more often
than block boundaries — which cuts directly against the reason block-wise
caching exists at all (see "Block-wise KV Caching" in `README.md`'s
Optimizations section). The tradeoff was judged not worth chasing without
a specific accuracy target requiring it: the gap is now well-understood
and localized (long generations specifically), which is enough to inform
future decisions (e.g. being cautious about `model_update` accuracy claims
on very long generations) without giving back the caching speedup to close
it.

---

## Summary timeline

| Stage | What | Outcome |
|---|---|---|
| Dynamic-k (fixed/step-ramped top-5) | Reduce active experts per token, ramped by denoising step | 3.75x single-GPU speedup, but cost 1.5-6pt accuracy (MMLU 66%→60%, MMLU-Pro 40.0%→38.5%); found the "9.38% token divergence" is caused by caching, not top-k |
| Adaptive/nucleus-k | Per-token cumulative-probability threshold instead of per-step fixed count | No advantage over fixed-5 — router is near-uniform on this checkpoint, so nucleus has no peakedness variance to exploit; worse than fixed-5 at aggressive thresholds |
| Revert to static top-8 | Remove both schemes entirely | Static top-8 + Triton fused MoE + KV caching alone already gets 4.54x speedup at 2x A6000 TP+EP — neither alternative beat that with acceptable accuracy |
| `model_update` vs HF correctness testing | Compare CoT MMLU/MMLU-Pro accuracy | Found real, large gap (28% vs 46% MMLU-Pro) caused by response collapse to bare `"Final Answer: X"` |
| Root-cause + fix | KV-cache priming/finalize calls missing mask-placeholder context | Fixed in commit `a5f6ebe`; confirmed on 3/3 known-bad questions and at whole-benchmark level: MMLU-Pro 28.0%→40.0% (HF: 46.0%); 6pt residual gap unexplained, MMLU re-run still pending |
| Residual gap corroboration (§2.9) | Re-ran `model_update` vs HF on GSM8K, bucketed accuracy by response length | Same ~6pt gap on a second task (68.0% vs 74.0%); gap grows sharply with response length (-7.7pt → +16.7pt → +50.0pt across 300-600/600-1200/1200+ char buckets), with `model_update` scoring 0/6 vs HF's 3/6 on the longest responses — strong corroboration of the block-commit-staleness hypothesis. Documented as the working explanation; not pursued into a fix, since closing it would mean sacrificing much of the block-wise caching speedup |
