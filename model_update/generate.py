"""
Block-wise KV-cached masked diffusion generation for LLaDA-MoE.

Same algorithm as generate.py (add_gumbel_noise, get_num_transfer_tokens,
low-confidence remasking, block restriction), but:
  - prompt + finalized blocks are cached once (K/V), never recomputed
  - each denoising step only runs the model over the ACTIVE block
  - each block gets one extra "finalize" forward pass after full unmask,
    purely to compute correct K/V to push into the cache
"""

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


def _generate_block_cached(
    model,
    x: torch.Tensor,
    block_start: int,
    block_end: int,
    steps_per_block: int,
    cache_buffer: KVCacheBuffer,
    temperature: float,
    remasking: str,
    graph_runner=None,
):
    block_length = block_end - block_start
    device = x.device

    block_mask_index = (x[:, block_start:block_end] == MASK_ID)
    num_transfer = get_num_transfer_tokens(block_mask_index, steps_per_block)

    for step in range(steps_per_block):
        suffix_ids = x[:, block_start:]
        active_ids = x[:, block_start:block_end]
        mask_index = (active_ids == MASK_ID)

        # graph_runner (if given -- opt-in, see generate_cached's
        # use_cuda_graph) replaces only this repeated per-step call, not
        # the once-per-block finalize call below or the once-per-generation
        # prime call in generate_cached: those each run once, so capturing
        # them wouldn't amortize the capture cost the way this one does
        # (called steps_per_block times per block with an identical shape).
        forward = graph_runner if graph_runner is not None else model
        suffix_logits, _ = forward(
            suffix_ids,
            position_offset=block_start,
            cache_buffer=cache_buffer,
            write_pos=block_start,
        )
        logits = suffix_logits[:, :block_length]

        logits_with_noise = add_gumbel_noise(logits, temperature)
        x0 = logits_with_noise.argmax(dim=-1)

        if remasking == "low_confidence":
            p = F.softmax(logits.float(), dim=-1)
            x0_p = p.gather(-1, x0.unsqueeze(-1)).squeeze(-1)
        elif remasking == "random":
            x0_p = torch.rand(x0.shape, device=device)
        else:
            raise ValueError(f"Unknown remasking: {remasking}")

        x0 = torch.where(mask_index, x0, active_ids)
        confidence = torch.where(mask_index, x0_p, torch.full_like(x0_p, -torch.inf))

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
    use_cuda_graph: bool = False,
) -> torch.Tensor:
    """
    Same signature/semantics as generate.generate(), minus cfg_scale
    (CFG doubles the batch and complicates cache bookkeeping; add back
    once single-sequence caching is verified correct).

    use_cuda_graph: capture+replay each denoising step's model forward as
    a CUDA graph instead of dispatching it normally every time. OFF by
    default -- opt-in, and NOT validated end-to-end on real hardware as
    written (see CUDAGraphRunner's docstring in model_update/model.py for
    the full list of correctness-relevant invariants it relies on). Only
    affects the per-step calls inside a block's denoising loop -- the
    prime call (once per generation) and each block's finalize call (once
    per block) stay ordinary eager calls, since capturing something that
    runs once doesn't amortize the capture cost.
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

    # Fresh runner per generate_cached() call, matching cache_buffer's own
    # per-call lifetime -- see CUDAGraphRunner's docstring (invariant 4) for
    # why graphs must never be reused across different cache_buffers.
    graph_runner = None
    if use_cuda_graph:
        from .model import CUDAGraphRunner
        graph_runner = CUDAGraphRunner(model)

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
            graph_runner=graph_runner,
        )

    return x[:, P:]