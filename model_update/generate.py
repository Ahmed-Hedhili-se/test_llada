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


@torch.no_grad()
def generate_cached(
    model,
    prompt_ids: torch.Tensor,
    gen_length: int = 128,
    steps: int = 128,
    block_length: int = 128,
    temperature: float = 0.0,
    remasking: str = "low_confidence",
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
    prompt_index = (x != MASK_ID)

    # Prime the cache with the prompt (prefix) once.
    _, cache = model(prompt_ids, position_offset=0, past_kv=None)

def _generate_block_cached(
    model,
    x: torch.Tensor,
    block_start: int,
    block_end: int,
    steps_per_block: int,
    cache,
    temperature: float,
    remasking: str,
    topk_override: int | None = None,
):
    block_length = block_end - block_start
    device = x.device
    
    block_mask_index = (x[:, block_start:block_end] == MASK_ID)
    num_transfer = get_num_transfer_tokens(block_mask_index, steps_per_block)

    for step in range(steps_per_block):
        suffix_ids = x[:, block_start:]
        active_ids = x[:, block_start:block_end]
        mask_index = (active_ids == MASK_ID)

        suffix_logits, _ = model(
            suffix_ids,
            position_offset=block_start,
            past_kv=cache,
            topk_override=topk_override
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
        topk_override=topk_override
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
            topk_override=None
        )

    return x[:, P:]


class Verifier:
    def score(self, prompt_ids, partial_response_ids) -> float:
        """Higher = more promising partial solution. Must be implemented by a concrete verifier."""
        raise NotImplementedError

class LogProbVerifier(Verifier):
    def __init__(self, model):
        self.model = model

    def score(self, prompt_ids, partial_response_ids) -> float:
        """
        Trivial placeholder for testing the search mechanics without a real PRM.
        NOT competitive with a trained PRM. Real accuracy gains depend on a real verifier.
        Scores by the model's own mean token confidence over the partial response.
        """
        with torch.no_grad():
            logits, _ = self.model(partial_response_ids, position_offset=0, past_kv=None)
            logprobs = F.log_softmax(logits.float(), dim=-1)
            max_logprobs = logprobs.max(dim=-1).values
            return max_logprobs.mean().item()


@torch.no_grad()
def generate_des(
    model,
    prompt_ids: torch.Tensor,
    verifier: Verifier,
    gen_length: int = 128,
    steps: int = 128,
    block_length: int = 128,
    N: int = 32,
    M: int | None = None,
    k_candidates: tuple = (4, 5, 6, 7, 8, 9, 10, 11),
    max_des_steps: int | None = None,
    temperature: float = 0.8,
):
    if M is None:
        M = max(1, N // 4)
    if max_des_steps is None:
        max_des_steps = gen_length // block_length
        
    assert gen_length % block_length == 0
    steps_per_block = steps // (gen_length // block_length)
    device = prompt_ids.device
    P = prompt_ids.shape[1]

    # Initial prompt caching
    _, prompt_cache = model(prompt_ids, position_offset=0, past_kv=None)
    
    # Create N candidates using k_candidates uniformly
    C = []
    num_k = len(k_candidates)
    
    for i in range(N):
        k = k_candidates[i % num_k]
        state = torch.full((1, P + gen_length), MASK_ID, dtype=torch.long, device=device)
        state[:, :P] = prompt_ids
        C.append({"state": state, "k": k, "cache": prompt_cache, "score": 0.0})

    for step_idx in range(max_des_steps):
        block_start = P + step_idx * block_length
        block_end = block_start + block_length
        
        new_C = []
        for cand in C:
            new_state, new_cache = _generate_block_cached(
                model=model,
                x=cand["state"].clone(),
                block_start=block_start,
                block_end=block_end,
                steps_per_block=steps_per_block,
                cache=cand["cache"],
                temperature=temperature,
                remasking="low_confidence",
                topk_override=cand["k"]
            )
            
            partial_response = new_state[:, :block_end]
            score = verifier.score(prompt_ids, partial_response)
            
            new_C.append({"state": new_state, "k": cand["k"], "cache": new_cache, "score": score})
            
        # Sort and prune to top-M
        new_C.sort(key=lambda c: c["score"], reverse=True)
        C = new_C[:M]
        
        k_counts = {k: sum(1 for c in C if c["k"] == k) for k in k_candidates}
        print(f"[DES Step {step_idx+1}] Retained {len(C)} candidates. Expert counts (k): {k_counts}")
        
    # Return highest scoring candidate
    return C[0]["state"][:, P:P+max_des_steps*block_length]