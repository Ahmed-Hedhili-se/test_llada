"""
Block-wise KV-cached masked diffusion generation for LLaDA-MoE.

Same algorithm as generate.py (add_gumbel_noise, get_num_transfer_tokens,
low-confidence remasking, block restriction), but:
  - prompt + finalized blocks are cached once (K/V), never recomputed
  - each denoising step only runs the model over the ACTIVE block
  - each block gets one extra "finalize" forward pass after full unmask,
    purely to compute correct K/V to push into the cache
"""

import warnings
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
    position_ids: Optional[torch.Tensor] = None,
    pad_mask: Optional[torch.Tensor] = None,
):
    """
    position_ids/pad_mask: set only when the batch mixes prompt lengths and was
    left-padded to a common width (see generate_cached). Both None keeps the
    original single-length path untouched.

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

    A remask_threshold mechanism (ported from dInfer's
    get_transfer_index_hierarchy_remask -- letting an already-revealed
    position be reverted to MASK and reconsidered) was tried here and
    REVERTED: it was meant to fix degenerate repetition loops that plain
    hierarchy decoding can hit on long generations, but real GSM8K
    evaluation showed it made things categorically worse (0% accuracy,
    every generation collapsing into repeated-token garbage from the very
    first question, versus 68% without it at the same config). Likely
    mechanism: the "always force the single highest-confidence selectable
    position" progress guarantee in select_transfer_indices_hierarchy,
    applied to a selectable pool that includes already-decided remask
    candidates (not just genuinely-masked positions), unconditionally
    forces low-confidence revisions of already-settled content instead of
    leaving it masked-and-pending -- destroying the "reveal once, stay
    fixed" stability that made the fixed schedule and plain hierarchy
    decoding work in the first place. See README.md's "Adaptive Decoding"
    section for the full investigation. The actual fix for long-generation
    repetition turned out to be unrelated to remasking: see
    generate_cached's docstring for the steps_per_block/block_length ratio
    requirement.

    When a threshold is given, a block never exceeds steps_per_block
    forward passes: the last allowed iteration always force-reveals any
    remaining masked positions outright, guaranteeing the block is never
    left with leftover MASK tokens and the step-count ceiling from `steps`
    is never exceeded.
    """
    block_length = block_end - block_start
    device = x.device

    block_mask_index = (x[:, block_start:block_end] == MASK_ID)
    num_transfer = None
    if confidence_threshold is None:
        num_transfer = get_num_transfer_tokens(block_mask_index, steps_per_block)

    for step in range(steps_per_block):
        active_ids = x[:, block_start:block_end]
        mask_index = (active_ids == MASK_ID)
        if not mask_index.any():
            break  # every row's block already fully unmasked -- no forward call needed

        # The layers run over the whole suffix (future MASK context is load-
        # bearing, see below), but only the active block's logits are ever
        # used -- num_logits keeps lm_head from projecting the rest to a
        # 157k-wide vocabulary just to have it sliced away here.
        suffix_ids = x[:, block_start:]
        logits, _ = model(
            suffix_ids,
            position_offset=block_start,
            cache_buffer=cache_buffer,
            write_pos=block_start,
            num_logits=block_length,
            position_ids=None if position_ids is None else position_ids[:, block_start:],
            # The key axis spans the whole cache up to block_end, not just the
            # suffix, since attention reads [prefix ; active] from the buffer.
            attn_mask=None if pad_mask is None else pad_mask[..., :x.shape[1]],
        )

        logits_with_noise = add_gumbel_noise(logits, temperature)
        x0_raw = logits_with_noise.argmax(dim=-1)  # the model's own top pick at EVERY position, unmodified

        if remasking == "low_confidence":
            p = F.softmax(logits.float(), dim=-1)
            x0_p = p.gather(-1, x0_raw.unsqueeze(-1)).squeeze(-1)  # confidence of x0_raw at EVERY position
        elif remasking == "random":
            x0_p = torch.rand(x0_raw.shape, device=device)
        else:
            raise ValueError(f"Unknown remasking: {remasking}")

        x0 = torch.where(mask_index, x0_raw, active_ids)
        confidence = torch.where(mask_index, x0_p, torch.full_like(x0_p, -torch.inf))
        if confidence_threshold is not None:
            is_last_step = (step == steps_per_block - 1)
            if is_last_step:
                transfer_index = mask_index.clone()  # last chance -- force full completion
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
    # num_logits=0: this pass exists purely for its K/V side effect on
    # cache_buffer -- the logits were computed and discarded outright.
    remaining_ids = x[:, block_start:]
    model(
        remaining_ids,
        position_offset=block_start,
        cache_buffer=cache_buffer,
        write_pos=block_start,
        num_logits=0,
        position_ids=None if position_ids is None else position_ids[:, block_start:],
        attn_mask=None if pad_mask is None else pad_mask[..., :x.shape[1]],
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
    prompt_lens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Same signature/semantics as generate.generate(), minus cfg_scale
    (CFG doubles the batch and complicates cache bookkeeping; add back
    once single-sequence caching is verified correct).

    prompt_lens: [B] int, the true (unpadded) length of each row of
    `prompt_ids`, which must be LEFT-padded to a common width. This is what
    lets a batch mix prompt lengths -- without it, `src/server.py` can only
    group requests whose prompts happen to tokenize to exactly the same
    length, which in real traffic collapses batches to one or two requests.

    Left-padding rather than right-padding because generation has to start at
    a single column for the whole batch: the block loop below walks
    `block_start = P + block_idx * block_length` once, not per row. Padding on
    the left aligns every prompt's END at column P.

    Two things follow from the padding and are handled here:
      * RoPE positions are computed per row, so a padded prompt is encoded
        identically to the same prompt unpadded (pad slots take position 0);
      * an additive attention mask makes the pad columns unattendable, which
        also covers the KV cache -- the buffer holds uninitialised values at
        those slots and must never be read.

    None (the default) is the original equal-length path, untouched.

    confidence_threshold: opt-in hierarchical threshold-based decoding
    (select_transfer_indices_hierarchy, ported from dInfer's
    HierarchyDecoder -- see its docstring for why this replaced an earlier,
    simpler, reverted threshold implementation) instead of the default
    fixed-per-step reveal schedule. None (default) preserves the exact
    original behavior. `steps` is still an exact ceiling either way: this
    can only make a block finish in FEWER forward passes, never more.
    low_confidence_threshold: floor applied to hierarchy decoding's
    per-segment picks (ignored when confidence_threshold is None).

    IMPORTANT when confidence_threshold is set: steps_per_block (= steps //
    (gen_length // block_length)) needs enough headroom relative to
    block_length, or the last-step force-reveal in _generate_block_cached
    ends up dumping most of the block uncoordinated in one shot -- the same
    jointly-uncoordinated-reveal failure hierarchy decoding exists to
    prevent, reintroduced via a different path. Confirmed on real hardware
    (GSM8K, n=50, seed=42): block_length=64/steps_per_block=16 (ratio 0.25)
    collapsed into degenerate repetition (0% acc); block_length=64/
    steps_per_block=32 (ratio 0.5) fixed it (68.0% acc, still below the
    non-threshold baseline's expected ~82% paper reference but a large
    recovery from 0%). MMLU-Pro at ratio 0.5 (block_length=32,
    steps_per_block=16) separately validated at 44.0% acc. Guideline:
    steps_per_block >= block_length / 2 -- enforced with a runtime warning
    below. A remask_threshold correction mechanism was also tried for the
    remaining gap and found to make things categorically worse (0% acc,
    reverted -- see README.md's "Adaptive Decoding" section for the full
    investigation and _generate_block_cached's docstring for why).
    """
    assert gen_length % block_length == 0, "gen_length must be divisible by block_length"
    num_blocks = gen_length // block_length
    steps_per_block = steps // num_blocks

    if confidence_threshold is not None and steps_per_block < block_length / 2:
        warnings.warn(
            f"confidence_threshold is set with steps_per_block={steps_per_block} and "
            f"block_length={block_length} (ratio {steps_per_block / block_length:.2f} < 0.5 "
            "guideline). This ratio has been observed to cause degenerate repetition on long "
            "generations (see README.md's 'Adaptive Decoding' section) -- consider raising "
            "steps or lowering block_length.",
            stacklevel=2,
        )

    device = prompt_ids.device
    B, P = prompt_ids.shape
    total_len = P + gen_length

    x = torch.full((B, total_len), MASK_ID, dtype=torch.long, device=device)
    x[:, :P] = prompt_ids

    position_ids = None
    pad_mask = None
    if prompt_lens is not None:
        prompt_lens = prompt_lens.to(device=device, dtype=torch.long)
        if int(prompt_lens.max()) > P:
            raise ValueError(
                f"prompt_lens max {int(prompt_lens.max())} exceeds the padded "
                f"prompt width {P}; prompt_ids must already be left-padded."
            )
        pad_len = P - prompt_lens                                   # [B]
        cols = torch.arange(total_len, device=device).unsqueeze(0)  # [1, total_len]

        # Real content starts at column pad_len[b]. Subtracting it gives each
        # row positions 0..(its own length + gen_length), so a padded prompt
        # encodes exactly like the unpadded one. Pad slots clamp to 0 and are
        # masked out, so their value is irrelevant.
        position_ids = (cols - pad_len.unsqueeze(1)).clamp_(min=0)

        # Additive -inf on the pad columns of the KEY axis. Additive rather
        # than boolean because SDPA adds it pre-softmax, which composes with
        # any future mask; broadcast over heads and queries.
        is_pad = cols < pad_len.unsqueeze(1)                         # [B, total_len]
        if is_pad.any():
            dtype = next(model.parameters()).dtype
            pad_mask = torch.zeros(B, 1, 1, total_len, device=device, dtype=dtype)
            pad_mask.masked_fill_(is_pad.view(B, 1, 1, total_len), torch.finfo(dtype).min)
        else:
            # Equal lengths after all: fall back to the untouched fast path
            # rather than carrying an all-zero mask through every layer.
            position_ids = None

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
    # num_logits=0: like the finalize pass, this one is run only for its K/V
    # side effect -- its logits were computed over the entire (prompt +
    # all-MASK) sequence and thrown away.
    model(x, position_offset=0, cache_buffer=cache_buffer, write_pos=0, num_logits=0,
          position_ids=position_ids, attn_mask=pad_mask)
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
            position_ids=position_ids,
            pad_mask=pad_mask,
        )

    # Generation always occupies [P, P+gen_length) for every row -- that is the
    # point of left-padding -- so the slice is uniform even with mixed prompts.
    return x[:, P:]


@torch.no_grad()
def generate_dense(
    model,
    prompt_ids: torch.Tensor,
    gen_length: int = 128,
    steps: int = 128,
    block_length: int = 128,
    temperature: float = 0.0,
    remasking: str = "low_confidence",
) -> torch.Tensor:
    """Same fixed-schedule algorithm as generate_cached, but with NO KV
    cache at all -- every step recomputes the FULL sequence from scratch
    via a stateless forward pass (cache_buffer=None), exactly like
    src.generate.generate()'s and HF's reference algorithm, just through
    this project's own model class/weights. Exists to isolate block-wise
    KV caching's own contribution to speed and accuracy, separate from
    Triton fused MoE (both generate_cached and generate_dense run the
    same model_update model class/weights, so caching is the only
    variable) -- used by eval/check_time_inference.py's --no-cache flag
    for timing, and eval/diagnose_cache_vs_dense.py for the correctness
    investigation that originally found the KV-cache priming bug (see
    INVESTIGATION_LOG.md Part 2).

    No confidence_threshold support -- that's specific to the cached
    path's block-wise force-reveal safety net (see generate_cached's
    docstring); this function only implements the fixed schedule.
    """
    assert gen_length % block_length == 0, "gen_length must be divisible by block_length"
    num_blocks = gen_length // block_length
    steps_per_block = steps // num_blocks
    device = prompt_ids.device
    B, P = prompt_ids.shape
    total_len = P + gen_length

    x = torch.full((B, total_len), MASK_ID, dtype=torch.long, device=device)
    x[:, :P] = prompt_ids

    for block_idx in range(num_blocks):
        block_start = P + block_idx * block_length
        block_end = P + (block_idx + 1) * block_length
        block_mask_index = (x[:, block_start:block_end] == MASK_ID)
        num_transfer = get_num_transfer_tokens(block_mask_index, steps_per_block)

        for step in range(steps_per_block):
            active_ids = x[:, block_start:block_end]
            mask_index = (active_ids == MASK_ID)
            if not mask_index.any():
                break

            full_logits, _ = model(x, position_offset=0)
            logits = full_logits[:, block_start:block_end]

            logits_with_noise = add_gumbel_noise(logits, temperature)
            x0_raw = logits_with_noise.argmax(dim=-1)

            if remasking == "low_confidence":
                p = F.softmax(logits.float(), dim=-1)
                x0_p = p.gather(-1, x0_raw.unsqueeze(-1)).squeeze(-1)
            elif remasking == "random":
                x0_p = torch.rand(x0_raw.shape, device=device)
            else:
                raise ValueError(f"Unknown remasking: {remasking}")

            x0 = torch.where(mask_index, x0_raw, active_ids)
            confidence = torch.where(mask_index, x0_p, torch.full_like(x0_p, -torch.inf))
            transfer_index = select_transfer_indices(confidence, num_transfer[:, step])

            active_ids = active_ids.clone()
            active_ids[transfer_index] = x0[transfer_index]
            x[:, block_start:block_end] = active_ids

    return x[:, P:]