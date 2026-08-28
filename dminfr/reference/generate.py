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
