"""
Aggressively optimized masked diffusion generation for LLaDA-MoE.

Key optimizations (in order of impact):
  1. LAYER SKIPPING: Skip every other layer in early denoising steps
  2. DYNAMIC EXPERT PRUNING: Ramp from min_k to base_k within each block
  3. FAST PATH: Minimal-overhead caching when sparse attention is disabled
  4. TOKEN FREEZING: Don't recompute high-confidence finalized tokens
  5. Sparse-dLLM + SparseD (kept for long-context scenarios)
"""

import math
import numpy as np
import torch
import torch.nn.functional as F

from model_update.kv_cache import LayerKVCache, SparsePattern, _candidate_mass

MASK_ID = 156895


def add_gumbel_noise(logits, temperature):
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer = torch.zeros(
        mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64
    ) + base
    for i in range(mask_num.size(0)):
        num_transfer[i, : remainder[i]] += 1
    return num_transfer


def get_dynamic_k(step, steps_per_block, base_k=8, min_k=2):
    """Ramp from min_k to base_k. More aggressive default: min_k=2."""
    progress = step / max(steps_per_block - 1, 1)
    k = min_k + int((base_k - min_k) * progress)
    return max(k, 1)


def get_expert_threshold(step, steps_per_block, expert_threshold=0.0, max_threshold=0.08):
    """Higher threshold for more aggressive pruning."""
    if expert_threshold == 0.0:
        return 0.0
    progress = step / max(steps_per_block - 1, 1)
    threshold = max_threshold * (1.0 - progress)
    return max(expert_threshold, threshold)


def get_active_layers(step, steps_per_block, num_layers=16, layer_skip_threshold=0.5):
    """
    Skip every other layer in early steps.
    Returns a list of layer indices to actually run.
    """
    progress = step / max(steps_per_block - 1, 1)
    if progress < layer_skip_threshold:
        # Early steps: run only even layers (0, 2, 4, 6, 8, 10, 12, 14)
        return list(range(0, num_layers, 2))
    else:
        # Late steps: run all layers
        return list(range(num_layers))


# ═══════════════════════════════════════════════════════════════════════════
# Baseline generate() - UNCHANGED
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def generate(
    model,
    prompt_ids,
    gen_length=128,
    steps=128,
    block_length=128,
    temperature=0.0,
    cfg_scale=0.0,
    remasking="low_confidence",
):
    assert gen_length % block_length == 0, "gen_length must be divisible by block_length"
    num_blocks = gen_length // block_length
    steps_per_block = steps // num_blocks

    device = prompt_ids.device
    P = prompt_ids.shape[1]

    x = torch.full((1, P + gen_length), MASK_ID, dtype=torch.long, device=device)
    x[:, :P] = prompt_ids
    prompt_index = (x != MASK_ID)

    for block_idx in range(num_blocks):
        block_start = P + block_idx * block_length
        block_end = P + (block_idx + 1) * block_length

        block_mask_index = (x[:, block_start:block_end] == MASK_ID)
        num_transfer = get_num_transfer_tokens(block_mask_index, steps_per_block)

        for step in range(steps_per_block):
            mask_index = (x == MASK_ID)

            if cfg_scale > 0.0:
                un_x = x.clone()
                un_x[prompt_index] = MASK_ID
                x_cat = torch.cat([x, un_x], dim=0)
                logits = model(x_cat)
                logits, un_logits = logits.chunk(2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x)

            logits_with_noise = add_gumbel_noise(logits, temperature)
            x0 = logits_with_noise.argmax(dim=-1)

            if remasking == "low_confidence":
                p = F.softmax(logits.float(), dim=-1)
                x0_p = p.gather(-1, x0.unsqueeze(-1)).squeeze(-1)
            elif remasking == "random":
                x0_p = torch.rand(x0.shape, device=device)
            else:
                raise ValueError(f"Unknown remasking: {remasking}")

            x0_p[:, block_end:] = -torch.inf
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, torch.full_like(x0_p, -torch.inf))

            transfer_index = torch.zeros_like(x0, dtype=torch.bool)
            for j in range(confidence.shape[0]):
                k = num_transfer[j, step].item()
                if k > 0:
                    _, sel = torch.topk(confidence[j], k=int(k))
                    transfer_index[j, sel] = True

            x[transfer_index] = x0[transfer_index]

    return x[:, P:]


# ═══════════════════════════════════════════════════════════════════════════
# FAST generate_dense_cached() - caching WITHOUT sparse path overhead
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def generate_dense_cached(
    model,
    prompt_ids,
    gen_length=128,
    steps=128,
    block_length=128,
    cache_budget=None,
    temperature=0.0,
    remasking="low_confidence",
    use_dynamic_experts=True,
    base_k=8,
    min_k=2,
    expert_threshold=0.05,
    use_layer_skipping=True,
    layer_skip_threshold=0.5,
):
    """
    Dense attention + KV caching + dynamic expert pruning + layer skipping.
    NO sparse attention overhead. NO saliency tracking. NO mask building.
    This is the FAST path for short sequences where sparse attention
    overhead exceeds its savings.
    """
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    steps_per_block = steps // num_blocks

    device = prompt_ids.device
    P = prompt_ids.shape[1]
    NL = len(model.layers)
    layer_caches = [LayerKVCache(budget=cache_budget) for _ in range(NL)]

    # Prefill
    _, new_kv0, positions0, _ = model.forward_active(
        prompt_ids, prefix_len=0, layer_caches=layer_caches,
        sparse_pattern=None, step=0, sparse_step_threshold=10**9,
        need_weights=False, dynamic_k=None, expert_threshold=0.0,
        active_layers=list(range(NL)),
    )
    for li, cache in enumerate(layer_caches):
        cache.append(new_kv0[li][0], new_kv0[li][1], positions0, protected=False)

    x_active = torch.full((1, gen_length), MASK_ID, dtype=torch.long, device=device)

    for block_idx in range(num_blocks):
        block_start = block_idx * block_length
        block_end = (block_idx + 1) * block_length
        prefix_len = P + block_start

        block_mask_index = (x_active[:, block_start:block_end] == MASK_ID)
        num_transfer = get_num_transfer_tokens(block_mask_index, steps_per_block)

        for step in range(steps_per_block):
            active_ids = x_active[:, block_start:]
            mask_index = (active_ids == MASK_ID)

            # Dynamic expert pruning
            if use_dynamic_experts:
                dynamic_k = get_dynamic_k(step, steps_per_block, base_k=base_k, min_k=min_k)
                thresh = get_expert_threshold(step, steps_per_block, expert_threshold=expert_threshold)
            else:
                dynamic_k = None
                thresh = 0.0

            # Layer skipping
            if use_layer_skipping:
                active_layers = get_active_layers(step, steps_per_block, num_layers=NL, layer_skip_threshold=layer_skip_threshold)
            else:
                active_layers = list(range(NL))

            logits, new_kv, q_positions, _ = model.forward_active(
                active_ids, prefix_len=prefix_len, layer_caches=layer_caches,
                sparse_pattern=None, step=step, sparse_step_threshold=10**9,
                need_weights=False, dynamic_k=dynamic_k, expert_threshold=thresh,
                active_layers=active_layers,
            )

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
            x0_p[:, local_block_end:] = -torch.inf

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

        # Commit block
        finished_ids = x_active[:, block_start:block_end]
        _, new_kv_final, positions_final, _ = model.forward_active(
            finished_ids, prefix_len=prefix_len, layer_caches=layer_caches,
            sparse_pattern=None, step=steps_per_block, sparse_step_threshold=10**9,
            need_weights=False, dynamic_k=None, expert_threshold=0.0,
            active_layers=list(range(NL)),
        )
        for li, cache in enumerate(layer_caches):
            cache.append(new_kv_final[li][0], new_kv_final[li][1], positions_final, protected=False)

    return x_active[:, :]


# ═══════════════════════════════════════════════════════════════════════════
# Original generate_sparse_cached() - kept for long-context scenarios
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def generate_sparse_cached(
    model,
    prompt_ids,
    gen_length=128,
    steps=128,
    block_length=128,
    sparse_pattern=None,
    sparse_step_threshold=4,
    cache_budget=None,
    saliency_update_interval=8,
    temperature=0.0,
    remasking="low_confidence",
    use_dynamic_experts=True,
    base_k=8,
    min_k=2,
    expert_threshold=0.05,
):
    """Original sparse path with dynamic expert pruning."""
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    steps_per_block = steps // num_blocks

    device = prompt_ids.device
    P = prompt_ids.shape[1]
    NL = len(model.layers)
    layer_caches = [LayerKVCache(budget=cache_budget) for _ in range(NL)]

    _, new_kv0, positions0, _ = model.forward_active(
        prompt_ids, prefix_len=0, layer_caches=layer_caches,
        sparse_pattern=None, step=0, sparse_step_threshold=10**9,
        need_weights=False, dynamic_k=None, expert_threshold=0.0,
        active_layers=list(range(NL)),
    )
    for li, cache in enumerate(layer_caches):
        cache.append(new_kv0[li][0], new_kv0[li][1], positions0, protected=False)

    x_active = torch.full((1, gen_length), MASK_ID, dtype=torch.long, device=device)

    for block_idx in range(num_blocks):
        block_start = block_idx * block_length
        block_end = (block_idx + 1) * block_length
        prefix_len = P + block_start

        block_mask_index = (x_active[:, block_start:block_end] == MASK_ID)
        num_transfer = get_num_transfer_tokens(block_mask_index, steps_per_block)

        for step in range(steps_per_block):
            active_ids = x_active[:, block_start:]
            mask_index = (active_ids == MASK_ID)
            track_saliency = (step % saliency_update_interval == 0)

            if use_dynamic_experts:
                dynamic_k = get_dynamic_k(step, steps_per_block, base_k=base_k, min_k=min_k)
                thresh = get_expert_threshold(step, steps_per_block, expert_threshold=expert_threshold)
            else:
                dynamic_k = None
                thresh = 0.0

            logits, new_kv, q_positions, all_attn = model.forward_active(
                active_ids, prefix_len=prefix_len, layer_caches=layer_caches,
                sparse_pattern=sparse_pattern, step=step,
                sparse_step_threshold=sparse_step_threshold, need_weights=track_saliency,
                dynamic_k=dynamic_k, expert_threshold=thresh,
                active_layers=list(range(NL)),
            )

            if track_saliency:
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
            x0_p[:, local_block_end:] = -torch.inf

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

        finished_ids = x_active[:, block_start:block_end]
        _, new_kv_final, positions_final, _ = model.forward_active(
            finished_ids, prefix_len=prefix_len, layer_caches=layer_caches,
            sparse_pattern=None, step=steps_per_block, sparse_step_threshold=10**9,
            need_weights=False, dynamic_k=None, expert_threshold=0.0,
            active_layers=list(range(NL)),
        )
        for li, cache in enumerate(layer_caches):
            cache.append(new_kv_final[li][0], new_kv_final[li][1], positions_final, protected=False)

    return x_active[:, :]


# ═══════════════════════════════════════════════════════════════════════════
# Calibration (unchanged)
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def calibrate_sparse_pattern(
    model,
    calibration_prompt_ids,
    candidate_windows=(16, 32, 64, 128),
    candidate_strides=(0, 16, 32, 64),
    mass_threshold=0.9,
):
    assert len(calibration_prompt_ids) > 0
    stats_sum = None
    n_prompts = 0
    for ids in calibration_prompt_ids:
        _, all_attn = model.forward_with_attn(ids)
        NL = len(all_attn)
        NH = all_attn[0].shape[1]
        if stats_sum is None:
            stats_sum = torch.zeros(NL, NH, len(candidate_windows), len(candidate_strides))
        for li, aw in enumerate(all_attn):
            heads = aw[0]
            for h in range(NH):
                for wi, window in enumerate(candidate_windows):
                    for si, stride in enumerate(candidate_strides):
                        stats_sum[li, h, wi, si] += _candidate_mass(heads[h], window, stride)
        n_prompts += 1

    stats = stats_sum / n_prompts
    NL, NH = stats.shape[0], stats.shape[1]
    window_out = torch.zeros(NL, NH, dtype=torch.long)
    stride_out = torch.zeros(NL, NH, dtype=torch.long)
    max_stride_cost = max(candidate_windows)

    for li in range(NL):
        for h in range(NH):
            best = None
            for wi, window in enumerate(candidate_windows):
                for si, stride in enumerate(candidate_strides):
                    mass = stats[li, h, wi, si].item()
                    if mass >= mass_threshold:
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