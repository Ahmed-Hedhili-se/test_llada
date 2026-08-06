"""
Block-wise KV-cached masked diffusion generation for LLaDA-MoE.

Same algorithm as generate.py (add_gumbel_noise, get_num_transfer_tokens,
low-confidence remasking, block restriction), but:
  - prompt + finalized blocks are cached once (K/V), never recomputed
  - each denoising step only runs the model over the ACTIVE block
  - each block gets one extra "finalize" forward pass after full unmask,
    purely to compute correct K/V to push into the cache
"""

from typing import Optional

import torch
import torch.nn.functional as F

from .model import KVCacheBuffer

MASK_ID = 156895


def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    """Vectorized: the original `num_transfer[i, :remainder[i]] += 1` loop
    used a per-row tensor as a slice bound, which implicitly calls
    `__index__` (a hidden host sync, once per batch row per block) -- the
    same class of cost eval/test_moe_align_block_size.py's vectorization
    eliminated for MoE routing, just smaller blast radius here since this
    runs once per block rather than once per step. `positions < remainder`
    reproduces the same "first `remainder[i]` columns get one extra" pattern
    with a single broadcast comparison instead."""
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    positions = torch.arange(steps, device=mask_index.device).unsqueeze(0)
    return base + (positions < remainder).to(base.dtype)


def select_transfer_indices(confidence: torch.Tensor, num_transfer_step: torch.Tensor) -> torch.Tensor:
    """
    For each batch row, select the top-k highest-confidence positions to
    reveal this step, where k = num_transfer_step[row] (can differ per row).

    Vectorized: the original looped per batch row, calling `.item()` on
    that row's transfer count and running an independent torch.topk() per
    row -- a host sync PER ROW PER STEP, the exact bottleneck a trace-
    profiling investigation found (thousands of ~8us idle gaps between
    kernel launches in a batched generation run). torch.topk has no
    per-row-k mode, so instead: one topk call at the batch's max k, then
    mask each row down to its own (smaller-or-equal) k. topk's output is
    sorted descending, so "keep the first k_j of max_k" is exactly the
    top-k_j for row j -- the same result as calling topk(k=k_j) directly
    for that row, not an approximation (see eval/test_select_transfer_indices.py).

    confidence: [B, T] float, -inf at positions that must not be selected.
    num_transfer_step: [B] int, how many positions each row should reveal.
    Returns: [B, T] bool mask.
    """
    transfer_index = torch.zeros_like(confidence, dtype=torch.bool)
    max_k = int(num_transfer_step.max().item())  # one sync, not one per row
    if max_k > 0:
        _, topk_idx = torch.topk(confidence, k=max_k, dim=-1)
        positions = torch.arange(max_k, device=confidence.device).unsqueeze(0)
        keep_mask = positions < num_transfer_step.unsqueeze(1)
        transfer_index.scatter_(1, topk_idx, keep_mask)
    return transfer_index


def select_transfer_indices_hierarchy(
    confidence: torch.Tensor,
    threshold: Optional[float] = None,
    low_threshold: Optional[float] = 0.4,
) -> torch.Tensor:
    """
    Ported from dInfer's HierarchyDecoder (inclusionAI/dInfer,
    python/dinfer/decoding/parallel_strategy.py -- the same project's
    gsm8k-llada-moe.yaml eval config this project's GSM8K/BBH/CRUX-O harness
    already aligned to elsewhere). "Force separate decisions" instead of a
    plain per-position threshold: an earlier, simpler threshold-only
    implementation here (select_transfer_indices_threshold, reverted -- see
    git history) revealed every position independently clearing a threshold
    in one step, which caused a real ~33% relative MMLU-Pro accuracy drop
    (36.0%->24.0%, n=50) -- locally-confident-but-jointly-uncoordinated
    reveals, with no chance for the model to reconsider before the next
    step, corrupted CoT reasoning chains. This is dInfer's own fix for
    exactly that failure mode.

    Three-part selection, unioned together:
      1. At most ONE position per contiguous run of currently-selectable
         positions (a "segment") -- specifically, that segment's highest-
         confidence position. Instead of every qualifying position across
         the whole block, at most one reveal per locally-correlated span
         per step, forcing separate, sequential decisions about spans that
         are far enough apart to be genuinely independent, rather than a
         free-for-all across the whole block.
      2. Segment picks are floored by low_threshold -- a segment's pick
         doesn't count if even its best position isn't reasonably
         confident (default 0.4, matching dInfer's HierarchyDecoder default).
      3. Positions independently clearing `threshold` are unioned in
         regardless of the segment/low_threshold gating (still lets a
         genuinely highly-confident position skip ahead of its segment
         peers), and the single globally highest-confidence position is
         always included unconditionally -- guarantees at least one token
         transfers per row per step whenever that row has any selectable
         position left, the same progress guarantee
         select_transfer_indices_threshold's fallback provided, just
         applied unconditionally here (matching dInfer's own reference)
         rather than only when nothing else qualified.

    Fully vectorized: segment IDs via a cumsum over segment-start markers,
    per-segment max via scatter_reduce(reduce="amax") -- no Python loop, no
    host sync. dInfer's own reference implementation of this exact
    algorithm asserts batch size 1; the segment-id/scatter_reduce operations
    here are already independent along dim=1 (verified by inspection: every
    op either broadcasts per-row or scatters within dim=1 using a per-row
    index tensor), so this generalizes to any batch size without changing
    the per-row result.

    confidence: [B, T] float, -inf at positions that must not be selected
    (already-revealed, outside the active block, or -- for a row with no
    selectable position left at all, e.g. already fully done in a batched
    call where other rows aren't -- every position in that row).
    Returns: [B, T] bool mask.
    """
    B, L = confidence.shape
    device = confidence.device
    neg_inf_val = torch.finfo(confidence.dtype).min

    selectable = torch.isfinite(confidence)

    prev = torch.cat([selectable.new_zeros((B, 1)), selectable[:, :-1]], dim=1)
    starts = selectable & ~prev
    seg_id = torch.cumsum(starts.to(torch.int64), dim=-1) - 1
    seg_id = torch.where(selectable, seg_id, torch.zeros_like(seg_id))

    seg_max = torch.full((B, L), neg_inf_val, device=device, dtype=confidence.dtype)
    seg_max = torch.scatter_reduce(seg_max, dim=1, index=seg_id, src=confidence, reduce="amax", include_self=True)
    seg_max_at_pos = seg_max.gather(dim=1, index=seg_id)
    transfer_index = selectable & (confidence == seg_max_at_pos)

    if low_threshold is not None:
        transfer_index = transfer_index & (confidence > low_threshold)

    if threshold is not None:
        transfer_index = transfer_index | (confidence > threshold)

    # Always include the single globally highest-confidence position per
    # row (unconditionally, matching dInfer's HierarchyDecoder) -- but only
    # for rows that actually have a selectable position left, since a row
    # already fully done (every position -inf) must select nothing.
    has_candidate = selectable.any(dim=-1)
    if has_candidate.any():
        top1_idx = confidence.argmax(dim=-1)
        row_idx = torch.nonzero(has_candidate, as_tuple=True)[0]
        transfer_index[row_idx, top1_idx[row_idx]] = True

    return transfer_index


def _generate_block_cached(
    model,
    x: torch.Tensor,
    block_start: int,
    block_end: int,
    steps_per_block: int,
    cache_buffer: KVCacheBuffer,
    temperature: float,
    remasking: str,
    confidence_threshold: Optional[float] = None,
    low_confidence_threshold: Optional[float] = 0.4,
    remask_threshold: Optional[float] = None,
):
    """
    confidence_threshold: opt-in hierarchical threshold-based token
    selection (select_transfer_indices_hierarchy, ported from dInfer's
    HierarchyDecoder) instead of the default fixed per-step reveal count
    (select_transfer_indices). None (default) preserves the exact original
    behavior byte-for-byte -- the early-exit check below never triggers for
    the fixed schedule (it spreads reveals evenly across every allotted
    step by construction, so mask_index can only become all-False on the
    very last iteration, which is where the loop would end anyway), so
    existing callers are unaffected either way.

    low_confidence_threshold: floor applied to hierarchy decoding's
    per-segment picks (see select_transfer_indices_hierarchy). Ignored
    when confidence_threshold is None.

    remask_threshold: opt-in, ported from dInfer's
    get_transfer_index_hierarchy_remask -- lets an already-revealed
    position get reverted back to MASK and reconsidered if its confidence
    (re-evaluated against the now-more-complete context) drops below this
    value, instead of every reveal being permanent. Requires
    confidence_threshold to also be set; ignored otherwise.

    Plain hierarchical decoding (confidence_threshold set, remask_threshold
    None) has no way to correct an early mistake once revealed -- validated
    on MMLU-Pro (real accuracy improvement over the fixed schedule) but
    found on GSM8K's much longer generations (1024 tokens vs MMLU-Pro's
    256) to sometimes lock into degenerate repetition loops: once a
    confidently-wrong token starts a repeat, repeating it is itself often a
    high-confidence prediction, and nothing before this could ever revisit
    that choice. remask_threshold adds that correction path back: at every
    step (except the last, where it's disabled -- see below), the model's
    fresh confidence is re-checked against EVERY position in the block, not
    just currently-masked ones; already-revealed positions whose confidence
    has now dropped below remask_threshold are added to the selectable set
    for this step (unioned with -- not replacing -- select_transfer_indices_
    hierarchy's normal segment/threshold logic operating on that same
    expanded set). A shaky position that gets re-selected this step is
    REVISED to the model's current top pick at that position -- which may
    differ from what was already there, not merely re-confirmed as-is --
    since confidence/selection is computed from the raw, unmodified argmax
    at every position (matching dInfer's reference exactly, not a
    mask_index-preserved version). A shaky position that does NOT get
    re-selected is reverted to MASK_ID instead and marked as this step's
    write. A position can therefore be revealed, revised, reverted to MASK,
    and revealed again with a different value multiple times across a
    block's steps.

    Differs from dInfer's reference implementation in one respect: dInfer's
    version guarantees "at least (count being remasked this step + 1)"
    selections via its own gap-filling top-up; this reuses
    select_transfer_indices_hierarchy unmodified, whose own guarantee is
    simpler (the single globally highest-confidence position is always
    included). Chosen deliberately for lower implementation risk (no new
    selection logic to separately validate) -- correctness/termination
    doesn't depend on either progress-rate guarantee anyway, since the
    last-step force-reveal below is an unconditional, independent backstop.

    Early exit (breaking out of the step loop once nothing is masked) is
    unconditionally DISABLED whenever remasking is active (confidence_
    threshold and remask_threshold both set): a row with no currently-
    masked positions can still have shaky already-revealed content worth
    reconsidering, so "nothing is masked" no longer means "nothing left to
    do." Remasking-enabled generation therefore always uses the full
    steps_per_block budget for a block. Plain hierarchy decoding (no
    remasking) keeps its existing early-exit behavior unchanged.

    When a threshold is given, a block never exceeds steps_per_block
    forward passes regardless of confidence/remasking behavior on earlier
    steps: the last allowed iteration always force-reveals any remaining
    masked positions outright and never remasks anything, guaranteeing the
    block is never left with leftover MASK tokens and the step-count
    ceiling from `steps` is never exceeded.
    """
    block_length = block_end - block_start
    device = x.device
    remask_active = confidence_threshold is not None and remask_threshold is not None

    block_mask_index = (x[:, block_start:block_end] == MASK_ID)
    num_transfer = None
    if confidence_threshold is None:
        num_transfer = get_num_transfer_tokens(block_mask_index, steps_per_block)

    for step in range(steps_per_block):
        active_ids = x[:, block_start:block_end]
        mask_index = (active_ids == MASK_ID)
        if not mask_index.any() and not remask_active:
            break  # every row's block already fully unmasked -- no forward call needed

        suffix_ids = x[:, block_start:]
        suffix_logits, _ = model(
            suffix_ids,
            position_offset=block_start,
            cache_buffer=cache_buffer,
            write_pos=block_start,
        )
        logits = suffix_logits[:, :block_length]

        logits_with_noise = add_gumbel_noise(logits, temperature)
        x0_raw = logits_with_noise.argmax(dim=-1)  # the model's own top pick at EVERY position, unmodified

        if remasking == "low_confidence":
            p = F.softmax(logits.float(), dim=-1)
            x0_p = p.gather(-1, x0_raw.unsqueeze(-1)).squeeze(-1)  # confidence of x0_raw at EVERY position
        elif remasking == "random":
            x0_p = torch.rand(x0_raw.shape, device=device)
        else:
            raise ValueError(f"Unknown remasking: {remasking}")

        is_last_step = (step == steps_per_block - 1)

        if confidence_threshold is not None and remask_active and not is_last_step:
            # Remasking path: deliberately uses x0_raw directly, NOT a
            # mask_index-preserved version -- an already-revealed position
            # that gets reconfirmed (selected but not reverted) may
            # legitimately be REVISED to the model's fresh top pick here,
            # not just kept at its exact old value or fully erased. This
            # matches dInfer's reference exactly: its x0 is also the raw,
            # per-position argmax throughout, with revert positions
            # overwritten to mask_id as the only modification before the
            # caller applies transfer_index.
            low_conf_now = x0_p < remask_threshold
            remask_candidates = low_conf_now & ~mask_index  # already-revealed, now shaky
            selectable_now = mask_index | remask_candidates
            remask_confidence = torch.where(selectable_now, x0_p, torch.full_like(x0_p, -torch.inf))
            transfer_index = select_transfer_indices_hierarchy(
                remask_confidence, threshold=confidence_threshold, low_threshold=low_confidence_threshold,
            )
            revert = remask_candidates & ~transfer_index  # shaky positions not re-confirmed this step
            x0 = torch.where(revert, torch.full_like(x0_raw, MASK_ID), x0_raw)
            transfer_index = transfer_index | revert
        else:
            # Every other path (fixed schedule, plain hierarchy decoding,
            # and the always-force-reveal last step even when remasking is
            # otherwise active) preserves existing values at non-candidate
            # positions exactly as before -- x0 only matters at positions
            # transfer_index actually selects, and those are always a
            # subset of mask_index here, so this is unchanged from the
            # validated pre-remasking behavior.
            x0 = torch.where(mask_index, x0_raw, active_ids)
            confidence = torch.where(mask_index, x0_p, torch.full_like(x0_p, -torch.inf))
            if confidence_threshold is not None:
                if is_last_step:
                    transfer_index = mask_index.clone()  # last chance -- force full completion, no remasking
                else:
                    transfer_index = select_transfer_indices_hierarchy(
                        confidence, threshold=confidence_threshold, low_threshold=low_confidence_threshold,
                    )
            else:
                transfer_index = select_transfer_indices(confidence, num_transfer[:, step])

        active_ids = active_ids.clone()
        active_ids[transfer_index] = x0[transfer_index]
        x[:, block_start:block_end] = active_ids

    # Recompute K/V for this block AND everything after it (still MASK at this
    # point), not just the block in isolation -- the model was trained to always
    # see the full sequence length with mask placeholders for ungenerated
    # content, so committing K/V computed from a truncated view (missing the
    # "there's more sequence after me" context) is out-of-distribution and
    # causes premature EOS collapse in later blocks. Only [block_start:block_end)
    # actually gets committed; the freshly-computed K/V for future blocks is
    # provisional and gets overwritten again once those blocks are processed.
    remaining_ids = x[:, block_start:]
    model(
        remaining_ids,
        position_offset=block_start,
        cache_buffer=cache_buffer,
        write_pos=block_start,
    )
    cache_buffer.commit(block_end)

    return x


@torch.no_grad()
def generate_cached(
    model,
    prompt_ids: torch.Tensor,
    gen_length: int = 128,
    steps: int = 128,
    block_length: int = 128,
    temperature: float = 0.0,
    remasking: str = "low_confidence",
    confidence_threshold: Optional[float] = None,
    low_confidence_threshold: Optional[float] = 0.4,
    remask_threshold: Optional[float] = None,
) -> torch.Tensor:
    """
    Same signature/semantics as generate.generate(), minus cfg_scale
    (CFG doubles the batch and complicates cache bookkeeping; add back
    once single-sequence caching is verified correct).

    confidence_threshold: opt-in hierarchical threshold-based decoding
    (select_transfer_indices_hierarchy, ported from dInfer's
    HierarchyDecoder -- see its docstring for why this replaced an earlier,
    simpler, reverted threshold implementation) instead of the default
    fixed-per-step reveal schedule. None (default) preserves the exact
    original behavior. `steps` is still an exact ceiling either way: this
    can only make a block finish in FEWER forward passes, never more,
    UNLESS remask_threshold is also set (see below), in which case the
    full step budget is always used.
    low_confidence_threshold: floor applied to hierarchy decoding's
    per-segment picks (ignored when confidence_threshold is None).
    remask_threshold: opt-in, ported from dInfer's
    get_transfer_index_hierarchy_remask -- lets an already-revealed
    position be reverted to MASK and reconsidered if its confidence drops
    on a later step, instead of every reveal being permanent (see
    _generate_block_cached's docstring for the full mechanism and why
    plain hierarchy decoding needed this: it was found to sometimes lock
    into degenerate repetition loops on long generations, e.g. GSM8K's
    1024-token budget, with no way to correct once revealed). Requires
    confidence_threshold to also be set; ignored otherwise.
    """
    assert gen_length % block_length == 0, "gen_length must be divisible by block_length"
    num_blocks = gen_length // block_length
    steps_per_block = steps // num_blocks

    device = prompt_ids.device
    B, P = prompt_ids.shape
    total_len = P + gen_length

    x = torch.full((B, total_len), MASK_ID, dtype=torch.long, device=device)
    x[:, :P] = prompt_ids

    cache_buffer = KVCacheBuffer(
        num_layers=model.cfg.NL,
        batch_size=B,
        kvh_local=model.layers[0].self_attn.KVH_local,
        max_len=total_len,
        head_dim=model.cfg.HD,
        dtype=next(model.parameters()).dtype,
        device=device,
    )

    # Prime the cache by running the FULL sequence (prompt + mask-filled
    # generation region), not just prompt_ids in isolation -- the model was
    # trained to always see the whole sequence length with mask placeholders
    # for ungenerated content, so caching K/V computed from the prompt alone
    # (as if nothing followed it) is out-of-distribution and causes the
    # cached path to collapse to premature EOS from the very first step. Only
    # [0:P) actually gets committed; the freshly-computed K/V for the
    # (all-MASK) generation region is provisional and gets overwritten once
    # block 0 is processed.
    model(x, position_offset=0, cache_buffer=cache_buffer, write_pos=0)
    cache_buffer.commit(P)

    for block_idx in range(num_blocks):
        block_start = P + block_idx * block_length
        block_end = P + (block_idx + 1) * block_length

        x = _generate_block_cached(
            model=model,
            x=x,
            block_start=block_start,
            block_end=block_end,
            steps_per_block=steps_per_block,
            cache_buffer=cache_buffer,
            temperature=temperature,
            remasking=remasking,
            confidence_threshold=confidence_threshold,
            low_confidence_threshold=low_confidence_threshold,
            remask_threshold=remask_threshold,
        )

    return x[:, P:]