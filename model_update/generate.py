"""
Optimized masked diffusion generation for LLaDA-MoE with dynamic expert pruning.

Integrates:
  1. Dynamic expert pruning (reduce active experts in early/noisy steps)
  2. Threshold-based expert filtering (drop negligible-weight experts)
  3. Sparse-dLLM evictable KV cache (unchanged)
  4. SparseD calibrated sparse attention patterns (unchanged)

The baseline generate() remains untouched as the correctness reference.
"""

import math
import numpy as np
import torch
import torch.nn.functional as F

from model_update.kv_cache import LayerKVCache, SparsePattern, _candidate_mass

MASK_ID = 156895


def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer = torch.zeros(
        mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64
    ) + base
    for i in range(mask_num.size(0)):
        num_transfer[i, : remainder[i]] += 1
    return num_transfer


def get_dynamic_k(step: int, steps_per_block: int, base_k: int = 8, min_k: int = None) -> int:
    """
    Reduce active experts in early steps when tokens are noisy.
    Ramps from min_k to base_k as denoising progresses within a block.
    """
    if min_k is None:
        min_k = max(2, base_k // 2)
    progress = step / max(steps_per_block - 1, 1)
    k = min_k + int((base_k - min_k) * progress)
    return k


def get_expert_threshold(step: int, steps_per_block: int, expert_threshold: float = 0.0, max_threshold: float = 0.05) -> float:
    """
    Apply threshold-based expert pruning in early steps.
    Drops experts with softmax weight below threshold after top-k selection.
    """
    if expert_threshold == 0.0:
        return 0.0
    progress = step / max(steps_per_block - 1, 1)
    threshold = max_threshold * (1.0 - progress)
    return max(expert_threshold, threshold)


# ═══════════════════════════════════════════════════════════════════════════
# Baseline generate() - UNCHANGED (dense, no-cache correctness baseline)
# ═══════════════════════════════════════════════════════════════════════════

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
# Optimized generate_sparse_cached() with dynamic expert pruning
# ═══════════════════════════════════════════════════════════════════════════

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
    saliency_update_interval: int = 8,
    temperature: float = 0.0,
    remasking: str = "low_confidence",
    # ── NEW: Dynamic expert pruning params ──
    use_dynamic_experts: bool = True,
    base_k: int = 8,
    min_k: int = 4,
    expert_threshold: float = 0.03,
) -> torch.Tensor:
    """
    Optimized generation with:
      - Sparse-dLLM evictable KV cache
      - SparseD calibrated sparse attention
      - Dynamic expert pruning (reduce active experts in early steps)
      - Threshold-based expert filtering
    """
    assert gen_length % block_length == 0, "gen_length must be divisible by block_length"
    num_blocks = gen_length // block_length
    steps_per_block = steps // num_blocks

    device = prompt_ids.device
    P = prompt_ids.shape[1]
    NL = len(model.layers)
    layer_caches = [LayerKVCache(budget=cache_budget) for _ in range(NL)]

    # ── Prefill: cache the prompt (always dense, full experts) ──
    _, new_kv0, positions0, _ = model.forward_active(
        prompt_ids, prefix_len=0, layer_caches=layer_caches,
        sparse_pattern=None, step=0, sparse_step_threshold=10 ** 9,
        need_weights=False, dynamic_k=None, expert_threshold=0.0,
    )
    for li, cache in enumerate(layer_caches):
        k_new, v_new = new_kv0[li]
        cache.append(k_new, v_new, positions0, protected=False)

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

            # ── NEW: Dynamic expert pruning ──
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
            )

            # Sparse-dLLM: update saliency and evict
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

        # Block finished: commit K/V to cache (dense, full experts)
        finished_ids = x_active[:, block_start:block_end]
        _, new_kv_final, positions_final, _ = model.forward_active(
            finished_ids, prefix_len=prefix_len, layer_caches=layer_caches,
            sparse_pattern=None, step=steps_per_block, sparse_step_threshold=10 ** 9,
            need_weights=False, dynamic_k=None, expert_threshold=0.0,
        )
        for li, cache in enumerate(layer_caches):
            k_new, v_new = new_kv_final[li]
            cache.append(k_new, v_new, positions_final, protected=False)

        print(f"  [Block {block_idx+1}/{num_blocks}] Cache size (layer 0): {len(layer_caches[0])} tokens")

    return x_active[:, :]


# ═══════════════════════════════════════════════════════════════════════════
# Calibration (unchanged from original)
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def calibrate_sparse_pattern(
    model,
    calibration_prompt_ids: list,
    candidate_windows=(16, 32, 64, 128),
    candidate_strides=(0, 16, 32, 64),
    mass_threshold: float = 0.9,
) -> SparsePattern:
    assert len(calibration_prompt_ids) > 0, "need at least one calibration prompt"

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


# ═══════════════════════════════════════════════════════════════════════════
# Compute savings estimator (updated with dynamic expert pruning)
# ═══════════════════════════════════════════════════════════════════════════

def estimate_compute_savings(
    prompt_len: int,
    gen_length: int,
    block_length: int,
    steps: int,
    sparse_step_threshold: int = 4,
    avg_window: float = 32.0,
    avg_stride: float = 16.0,
    cache_budget: int = None,
    use_dynamic_experts: bool = True,
    base_k: int = 8,
    min_k: int = 4,
) -> dict:
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    steps_per_block = steps // num_blocks

    dense_pairs = 0
    sparse_pairs = 0
    cached_pairs = 0
    dense_expert_flops = 0
    sparse_expert_flops = 0

    for b in range(num_blocks):
        block_start = b * block_length
        seq_len_dense = prompt_len + gen_length
        active_len = gen_length - block_start
        prefix_len = prompt_len + block_start

        for step in range(steps_per_block):
            dense_pairs += seq_len_dense * seq_len_dense

            cached_prefix_len = min(prefix_len, cache_budget) if cache_budget else prefix_len
            keys_len = cached_prefix_len + active_len
            cached_pairs += active_len * keys_len

            if step >= sparse_step_threshold:
                stride_term = (keys_len / avg_stride) if avg_stride > 0 else 0.0
                per_query_keys = min(keys_len, 2 * avg_window + stride_term)
                sparse_pairs += active_len * per_query_keys
            else:
                sparse_pairs += active_len * keys_len

            dense_expert_flops += seq_len_dense * base_k

            if use_dynamic_experts:
                progress = step / max(steps_per_block - 1, 1)
                k_eff = min_k + (base_k - min_k) * progress
            else:
                k_eff = base_k
            sparse_expert_flops += active_len * k_eff

    return {
        "dense_pairs": dense_pairs,
        "cached_only_pairs": cached_pairs,
        "cached_plus_sparse_pairs": sparse_pairs,
        "dense_expert_flops": dense_expert_flops,
        "sparse_expert_flops": sparse_expert_flops,
        "cached_speedup_vs_dense": dense_pairs / max(cached_pairs, 1),
        "cached_plus_sparse_speedup_vs_dense": dense_pairs / max(sparse_pairs, 1),
        "expert_speedup_from_dynamic": dense_expert_flops / max(sparse_expert_flops, 1),
    }