"""
Masked diffusion generation for LLaDA-MoE.

The model is NOT autoregressive. Generation works by:
  1. Filling the response slots with MASK_ID tokens
  2. Running multiple denoising steps, each time unmasking the
     highest-confidence tokens in the current block
  3. Iterating block-by-block (block_length tokens at a time)

This mirrors the reference implementation from the HF model card exactly.
"""

import math
import numpy as np
import torch
import torch.nn.functional as F

from kv_cache import LayerKVCache, SparsePattern, _candidate_mass

MASK_ID = 156895


def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    """How many tokens to unmask at each step, distributed as evenly as possible."""
    mask_num = mask_index.sum(dim=1, keepdim=True)           # [B, 1]
    base      = mask_num // steps
    remainder = mask_num % steps
    num_transfer = torch.zeros(
        mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64
    ) + base
    for i in range(mask_num.size(0)):
        num_transfer[i, : remainder[i]] += 1
    return num_transfer                                        # [B, steps]


@torch.no_grad()
def generate(
    model,
    prompt_ids: torch.Tensor,
    gen_length: int = 128,
    steps: int = 128,
    block_length: int = 128,
    temperature: float = 0.0,
    cfg_scale: float = 0.0,
    remasking: str = "low_confidence",
) -> torch.Tensor:
    """
    Args:
        model        : LLaDAMoE instance (already on device, eval mode)
        prompt_ids   : [1, P] long tensor of prompt token ids
        gen_length   : number of tokens to generate
        steps        : total denoising steps (split across blocks)
        block_length : tokens per block; gen_length must be divisible by block_length
        temperature  : Gumbel noise temperature (0 = greedy)
        cfg_scale    : classifier-free guidance scale (0 = disabled)
        remasking    : "low_confidence" or "random"

    Returns:
        generated token ids: [1, gen_length]
    """
    assert gen_length % block_length == 0, "gen_length must be divisible by block_length"
    num_blocks   = gen_length // block_length
    steps_per_block = steps // num_blocks

    device = prompt_ids.device
    P = prompt_ids.shape[1]

    # Full sequence: [prompt | MASK...MASK]
    x = torch.full((1, P + gen_length), MASK_ID, dtype=torch.long, device=device)
    x[:, :P] = prompt_ids
    prompt_index = (x != MASK_ID)   # boolean mask: True where prompt tokens are

    for block_idx in range(num_blocks):
        block_start = P + block_idx * block_length
        block_end   = P + (block_idx + 1) * block_length

        block_mask_index = (x[:, block_start:block_end] == MASK_ID)   # [1, block_length]
        num_transfer = get_num_transfer_tokens(block_mask_index, steps_per_block)  # [1, steps]

        for step in range(steps_per_block):
            mask_index = (x == MASK_ID)   # [1, L]

            if cfg_scale > 0.0:
                # Classifier-free guidance: run once with prompt, once fully masked
                un_x = x.clone()
                un_x[prompt_index] = MASK_ID
                x_cat = torch.cat([x, un_x], dim=0)          # [2, L]
                logits = model(x_cat)                          # [2, L, V]
                logits, un_logits = logits.chunk(2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x)                              # [1, L, V]

            logits_with_noise = add_gumbel_noise(logits, temperature)
            x0 = logits_with_noise.argmax(dim=-1)             # [1, L]

            if remasking == "low_confidence":
                p = F.softmax(logits.float(), dim=-1)
                x0_p = p.gather(-1, x0.unsqueeze(-1)).squeeze(-1)  # [1, L]
            elif remasking == "random":
                x0_p = torch.rand(x0.shape, device=device)
            else:
                raise ValueError(f"Unknown remasking: {remasking}")

            # Don't consider tokens beyond the current block for transfer
            x0_p[:, block_end:] = -torch.inf

            # Only update currently masked positions
            x0    = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, torch.full_like(x0_p, -torch.inf))

            # Pick the top-k highest-confidence masked tokens to unmask this step
            transfer_index = torch.zeros_like(x0, dtype=torch.bool)
            for j in range(confidence.shape[0]):
                k = num_transfer[j, step].item()
                if k > 0:
                    _, sel = torch.topk(confidence[j], k=int(k))
                    transfer_index[j, sel] = True

            x[transfer_index] = x0[transfer_index]

    return x[:, P:]   # return only the generated tokens


# ═══════════════════════════════════════════════════════════════════════════
# Sparse-dLLM (arXiv 2508.02558) + SparseD (arXiv 2509.24014) integration.
#
# generate() above is untouched and remains the fully-dense, no-cache
# correctness baseline. Everything below is additive.
#
# Scoping note (carried over from the integration request): both papers'
# published wins are largest at long context (tens of thousands of tokens,
# hundreds+ of steps). GSM8K-CoT generations are short. `estimate_compute_savings`
# below lets you sanity-check, from arithmetic alone, whether sparsity/eviction
# are worth it at your *actual* generation lengths before trusting a full run.
# ═══════════════════════════════════════════════════════════════════════════


@torch.no_grad()
def calibrate_sparse_pattern(
    model,
    calibration_prompt_ids: list,
    candidate_windows=(16, 32, 64, 128),
    candidate_strides=(0, 16, 32, 64),
    mass_threshold: float = 0.9,
) -> SparsePattern:
    """
    SparseD-style one-time offline calibration.

    For each (layer, head), sweep a small grid of (local window, global
    stride) candidates and measure, on `calibration_prompt_ids` (a handful of
    GSM8K-style prompts, run through the model with full dense attention),
    what fraction of that head's attention mass each candidate captures.
    Pick the cheapest candidate that clears `mass_threshold`; if none does,
    fall back to whichever candidate captured the most mass. The resulting
    SparsePattern is fixed and reused, unchanged, across every denoising step
    (past `sparse_step_threshold`) and every future call to
    generate_sparse_cached() — it is NOT recomputed per request.

    calibration_prompt_ids: list of [1, T] long tensors (can be different T's;
        calibration is done independently per prompt and averaged per-mass,
        so mixed lengths are fine).
    """
    assert len(calibration_prompt_ids) > 0, "need at least one calibration prompt"

    stats_sum = None
    n_prompts = 0

    for ids in calibration_prompt_ids:
        _, all_attn = model.forward_with_attn(ids)   # list[NL] of [1, NH, T, T]
        NL = len(all_attn)
        NH = all_attn[0].shape[1]
        if stats_sum is None:
            stats_sum = torch.zeros(NL, NH, len(candidate_windows), len(candidate_strides))

        for li, aw in enumerate(all_attn):
            heads = aw[0]   # [NH, T, T]
            for h in range(NH):
                for wi, window in enumerate(candidate_windows):
                    for si, stride in enumerate(candidate_strides):
                        stats_sum[li, h, wi, si] += _candidate_mass(heads[h], window, stride)
        n_prompts += 1

    stats = stats_sum / n_prompts
    NL, NH = stats.shape[0], stats.shape[1]
    window_out = torch.zeros(NL, NH, dtype=torch.long)
    stride_out = torch.zeros(NL, NH, dtype=torch.long)
    max_stride_cost = max(candidate_windows)  # rough per-token "coverage cost" proxy for stride

    for li in range(NL):
        for h in range(NH):
            best = None
            for wi, window in enumerate(candidate_windows):
                for si, stride in enumerate(candidate_strides):
                    mass = stats[li, h, wi, si].item()
                    if mass >= mass_threshold:
                        # cheaper (smaller window, sparser stride) is better among
                        # candidates that already meet the quality bar
                        cost = window + (max_stride_cost // stride if stride > 0 else 0)
                        if best is None or cost < best[0]:
                            best = (cost, window, stride)
            if best is None:
                flat = stats[li, h].flatten()
                idx = int(flat.argmax().item())
                wi, si = divmod(idx, len(candidate_strides))
                window, stride = candidate_windows[wi], candidate_strides[si]
            else:
                _, window, stride = best
            window_out[li, h] = window
            stride_out[li, h] = stride

    return SparsePattern(NL, NH, window_out, stride_out)


@torch.no_grad()
def generate_sparse_cached(
    model,
    prompt_ids: torch.Tensor,
    gen_length: int = 128,
    steps: int = 128,
    block_length: int = 128,
    sparse_pattern: SparsePattern = None,
    sparse_step_threshold: int = 4,
    cache_budget: int = None,
    temperature: float = 0.0,
    remasking: str = "low_confidence",
) -> torch.Tensor:
    """
    Sparse-dLLM (evictable saliency KV cache) + SparseD (calibrated per-head
    sparse attention) version of generate(). Mirrors generate()'s block-wise
    denoising algorithm exactly (add_gumbel_noise, get_num_transfer_tokens,
    low-confidence remasking, block restriction) — only the attention
    computation and caching differ.

    Args:
        sparse_pattern: SparsePattern from calibrate_sparse_pattern(), or None
            to run with Sparse-dLLM eviction only (no SparseD sparsity).
        sparse_step_threshold: the first `sparse_step_threshold` denoising
            steps of *every* block use full dense attention (SparseD found
            this necessary for quality); the calibrated sparse_pattern is
            only applied from that step onward.
        cache_budget: max cached (prompt + finalized-block) tokens per layer
            before Sparse-dLLM eviction kicks in. None disables eviction
            (plain, unbounded block-wise cache).

    Approximation note (same as any block-wise dLLM cache): once a block is
    finalized and its K/V committed to the cache, they are frozen — not
    recomputed when later blocks change — even though exact bidirectional
    attention would in principle let that happen. Do not expect a bit-exact
    match with generate(); validate on the real GSM8K accuracy gate instead.

    Note on cost: this path always computes attention with an explicit
    softmax (not fused SDPA) so it can read off attention weights for
    Sparse-dLLM's saliency updates and apply SparseD's mask. That is fine for
    this correctness/accuracy prototype; a production version would need a
    custom kernel to actually realize the FLOP savings (Phase 4, out of scope
    here — see the integration notes).
    """
    assert gen_length % block_length == 0, "gen_length must be divisible by block_length"
    num_blocks = gen_length // block_length
    steps_per_block = steps // num_blocks

    device = prompt_ids.device
    P = prompt_ids.shape[1]
    NL = len(model.layers)
    layer_caches = [LayerKVCache(budget=cache_budget) for _ in range(NL)]

    # ---- Prefill: cache the prompt (one-off, always dense) ----
    _, new_kv0, positions0, _ = model.forward_active(
        prompt_ids, prefix_len=0, layer_caches=layer_caches,
        sparse_pattern=None, step=0, sparse_step_threshold=10 ** 9, need_weights=False,
    )
    for li, cache in enumerate(layer_caches):
        k_new, v_new = new_kv0[li]
        cache.append(k_new, v_new, positions0, protected=False)

    x_active = torch.full((1, gen_length), MASK_ID, dtype=torch.long, device=device)

    for block_idx in range(num_blocks):
        block_start = block_idx * block_length
        block_end = (block_idx + 1) * block_length
        prefix_len = P + block_start   # already-cached prompt + finalized blocks

        block_mask_index = (x_active[:, block_start:block_end] == MASK_ID)
        num_transfer = get_num_transfer_tokens(block_mask_index, steps_per_block)

        for step in range(steps_per_block):
            active_ids = x_active[:, block_start:]   # current block onward, not yet committed
            mask_index = (active_ids == MASK_ID)

            logits, new_kv, q_positions, all_attn = model.forward_active(
                active_ids, prefix_len=prefix_len, layer_caches=layer_caches,
                sparse_pattern=sparse_pattern, step=step,
                sparse_step_threshold=sparse_step_threshold, need_weights=True,
            )

            # Sparse-dLLM: update saliency of the cached prefix, then evict.
            for li, cache in enumerate(layer_caches):
                cached = cache.get()
                if cached is None or all_attn[li] is None:
                    continue
                prefix_n = cached[2].shape[0]
                aw_prefix = all_attn[li][:, :, :, :prefix_n]
                cache.update_saliency(aw_prefix, cached[2])
                cache.evict()

            logits_with_noise = add_gumbel_noise(logits, temperature)
            x0 = logits_with_noise.argmax(dim=-1)

            if remasking == "low_confidence":
                p = F.softmax(logits.float(), dim=-1)
                x0_p = p.gather(-1, x0.unsqueeze(-1)).squeeze(-1)
            elif remasking == "random":
                x0_p = torch.rand(x0.shape, device=device)
            else:
                raise ValueError(f"Unknown remasking: {remasking}")

            local_block_end = block_end - block_start
            x0_p[:, local_block_end:] = -torch.inf   # only the current block may transfer this step

            x0 = torch.where(mask_index, x0, active_ids)
            confidence = torch.where(mask_index, x0_p, torch.full_like(x0_p, -torch.inf))

            transfer_index = torch.zeros_like(x0, dtype=torch.bool)
            for j in range(confidence.shape[0]):
                k = num_transfer[j, step].item()
                if k > 0:
                    _, sel = torch.topk(confidence[j], k=int(k))
                    transfer_index[j, sel] = True

            active_ids = torch.where(transfer_index, x0, active_ids)
            x_active[:, block_start:] = active_ids

        # Block finished: commit its K/V into the cache (dense, using only the
        # context available at commit time), then it drops out of "active".
        finished_ids = x_active[:, block_start:block_end]
        _, new_kv_final, positions_final, _ = model.forward_active(
            finished_ids, prefix_len=prefix_len, layer_caches=layer_caches,
            sparse_pattern=None, step=steps_per_block, sparse_step_threshold=10 ** 9,
            need_weights=False,
        )
        for li, cache in enumerate(layer_caches):
            k_new, v_new = new_kv_final[li]
            cache.append(k_new, v_new, positions_final, protected=False)

    return x_active[:, :]   # full x_active IS the generated tokens (no prompt prefix in it)


def estimate_compute_savings(
    prompt_len: int,
    gen_length: int,
    block_length: int,
    steps: int,
    sparse_step_threshold: int,
    avg_window: float,
    avg_stride: float,
    cache_budget: int = None,
) -> dict:
    """
    Back-of-envelope attention compute-volume comparison (# of query-key pairs
    scored, summed over all denoising steps) for generate() vs
    generate_sparse_cached(), at a given (prompt_len, gen_length, block_length,
    steps) configuration. No model forward passes — pure arithmetic — so you
    can quickly check, per the integration notes, whether the papers' wins
    (reported at 64k context / 1024 steps) still show up at your actual
    GSM8K-CoT lengths before trusting a full run.

    avg_window / avg_stride: rough averages of the calibrated SparsePattern's
        per-head window/stride (e.g. pattern.window.float().mean().item()).
    """
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    steps_per_block = steps // num_blocks

    dense_pairs = 0
    sparse_pairs = 0
    cached_pairs = 0   # generate_sparse_cached with caching but no SparseD sparsity

    for b in range(num_blocks):
        block_start = b * block_length
        seq_len_dense = prompt_len + gen_length         # generate(): always full sequence
        active_len = gen_length - block_start            # generate_sparse_cached(): active suffix only
        prefix_len = prompt_len + block_start

        for step in range(steps_per_block):
            # Dense baseline: full [seq_len, seq_len] attention every step.
            dense_pairs += seq_len_dense * seq_len_dense

            # Cached, still-dense-attention version: active queries attend to
            # (possibly evicted) cached prefix + active keys.
            cached_prefix_len = min(prefix_len, cache_budget) if cache_budget else prefix_len
            keys_len = cached_prefix_len + active_len
            cached_pairs += active_len * keys_len

            # Cached + SparseD sparsity, once the step threshold is passed.
            if step >= sparse_step_threshold:
                stride_term = (keys_len / avg_stride) if avg_stride > 0 else 0.0
                per_query_keys = min(keys_len, 2 * avg_window + stride_term)
                sparse_pairs += active_len * per_query_keys
            else:
                sparse_pairs += active_len * keys_len

    return {
        "dense_pairs": dense_pairs,
        "cached_only_pairs": cached_pairs,
        "cached_plus_sparse_pairs": sparse_pairs,
        "cached_speedup_vs_dense": dense_pairs / max(cached_pairs, 1),
        "cached_plus_sparse_speedup_vs_dense": dense_pairs / max(sparse_pairs, 1),
    }
