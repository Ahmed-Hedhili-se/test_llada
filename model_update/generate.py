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

from .model import concat_kv

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

def get_dynamic_k(step, steps_per_block, base_k=8, min_k=4):
    """Conservative: ramp from min_k to base_k experts."""
    progress = step / max(steps_per_block - 1, 1)
    k = min_k + (base_k - min_k) * progress
    return int(round(k))

def get_expert_threshold(step, steps_per_block, expert_threshold=0.03, max_threshold=0.05):
    """Start with high threshold, drop to baseline (e.g. 0.03) at the end."""
    progress = step / max(steps_per_block - 1, 1)
    thresh = max_threshold - (max_threshold - expert_threshold) * progress
    return thresh

def _generate_block_cached(
    model,
    x: torch.Tensor,
    block_start: int,
    block_end: int,
    steps_per_block: int,
    cache,
    temperature: float,
    remasking: str,
    use_dynamic_experts: bool = False,
    base_k: int = 8,
    min_k: int = 4,
    expert_threshold: float = 0.03,
    max_threshold: float = 0.05,
):
    block_length = block_end - block_start
    device = x.device
    
    block_mask_index = (x[:, block_start:block_end] == MASK_ID)
    num_transfer = get_num_transfer_tokens(block_mask_index, steps_per_block)

    for step in range(steps_per_block):
        suffix_ids = x[:, block_start:]
        active_ids = x[:, block_start:block_end]
        mask_index = (active_ids == MASK_ID)

        if use_dynamic_experts:
            dynamic_k = get_dynamic_k(step, steps_per_block, base_k, min_k)
            thresh = get_expert_threshold(step, steps_per_block, expert_threshold, max_threshold)
        else:
            dynamic_k = None
            thresh = 0.0

        suffix_logits, _ = model(
            suffix_ids,
            position_offset=block_start,
            past_kv=cache,
            dynamic_k=dynamic_k,
            expert_threshold=thresh
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

        transfer_index = torch.zeros_like(x0, dtype=torch.bool)
        for j in range(confidence.shape[0]):
            k = num_transfer[j, step].item()
            if k > 0:
                _, sel = torch.topk(confidence[j], k=int(k))
                transfer_index[j, sel] = True

        active_ids = active_ids.clone()
        active_ids[transfer_index] = x0[transfer_index]
        x[:, block_start:block_end] = active_ids

    finalized_ids = x[:, block_start:block_end]
    _, new_kv = model(
        finalized_ids,
        position_offset=block_start,
        past_kv=cache,
        dynamic_k=None,
        expert_threshold=0.0
    )
    cache = concat_kv(cache, new_kv)
    
    return x, cache


@torch.no_grad()
def generate_cached(
    model,
    prompt_ids: torch.Tensor,
    gen_length: int = 128,
    steps: int = 128,
    block_length: int = 128,
    temperature: float = 0.0,
    remasking: str = "low_confidence",
    use_dynamic_experts: bool = False,
    base_k: int = 8,
    min_k: int = 4,
    expert_threshold: float = 0.03,
    max_threshold: float = 0.05,
) -> torch.Tensor:
    """
    Same signature/semantics as generate.generate(), minus cfg_scale
    (CFG doubles the batch and complicates cache bookkeeping; add back
    once single-sequence caching is verified correct).
    """
    assert gen_length % block_length == 0, "gen_length must be divisible by block_length"
    num_blocks = gen_length // block_length
    steps_per_block = steps // num_blocks

    device = prompt_ids.device
    P = prompt_ids.shape[1]

    x = torch.full((1, P + gen_length), MASK_ID, dtype=torch.long, device=device)
    x[:, :P] = prompt_ids

    # Prime the cache with the prompt (prefix) once.
    _, cache = model(prompt_ids, position_offset=0, past_kv=None)

    for block_idx in range(num_blocks):
        block_start = P + block_idx * block_length
        block_end = P + (block_idx + 1) * block_length

        x, cache = _generate_block_cached(
            model=model,
            x=x,
            block_start=block_start,
            block_end=block_end,
            steps_per_block=steps_per_block,
            cache=cache,
            temperature=temperature,
            remasking=remasking,
            use_dynamic_experts=use_dynamic_experts,
            base_k=base_k,
            min_k=min_k,
            expert_threshold=expert_threshold,
            max_threshold=max_threshold
        )

    return x[:, P:]