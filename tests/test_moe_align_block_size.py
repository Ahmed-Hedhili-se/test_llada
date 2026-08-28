"""
Regression test for the vectorized dminfr/engine/fused_moe_triton.py::moe_align_block_size.

The original implementation looped over `range(num_experts)` in Python and
called `.item()` twice per expert (128 host-device syncs per call for
num_experts=64) purely to build Python-side padding bookkeeping. The
vectorized replacement computes the same sort/pad/scatter entirely with
GPU tensor ops, syncing only once at the end to size the output tensor.

This file:
  1. Keeps a frozen copy of the original loop-based implementation as a
     reference oracle.
  2. Asserts the vectorized version produces bit-identical output
     (dtype, shape, and values) across randomized configs and the specific
     edge cases that are easy to get wrong in a rewrite: zero-count
     ("vanishing") experts, exact block-size multiples (no padding),
     heavy padding, single-expert concentration, and empty input.
  3. Runs a CUDA-only end-to-end sanity check of fused_moe() itself
     (which calls moe_align_block_size internally) against a naive
     per-expert PyTorch loop, to confirm the full pipeline -- not just the
     alignment step in isolation -- still produces correct output.

moe_align_block_size is pure tensor math (no Triton/CUDA kernel), so part
1/2 run on CPU too. Part 3 needs a CUDA GPU (Triton requirement) and is
skipped with a message if unavailable.
"""

import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dminfr.engine.fused_moe_triton import moe_align_block_size


def _reference_moe_align_block_size(topk_ids: torch.Tensor, block_size: int, num_experts: int):
    """Frozen copy of the pre-vectorization implementation. Ground truth for this test."""
    num_tokens, top_k = topk_ids.shape
    flatten_ids = topk_ids.flatten()
    sorted_indices = torch.argsort(flatten_ids, stable=True)
    sorted_expert_ids = flatten_ids[sorted_indices]

    expert_counts = torch.bincount(sorted_expert_ids, minlength=num_experts)
    padded_expert_counts = ((expert_counts + block_size - 1) // block_size) * block_size

    padded_tokens = []
    padded_experts = []
    offset = 0
    for e in range(num_experts):
        count = expert_counts[e].item()
        padded_count = padded_expert_counts[e].item()

        if count > 0:
            padded_tokens.append(sorted_indices[offset: offset + count])
        if padded_count > count:
            padding = torch.full((padded_count - count,), num_tokens * top_k,
                                  dtype=sorted_indices.dtype, device=sorted_indices.device)
            padded_tokens.append(padding)

        if padded_count > 0:
            padded_experts.extend([e] * (padded_count // block_size))
        offset += count

    sorted_token_ids = torch.cat(padded_tokens) if padded_tokens else torch.empty(
        0, dtype=sorted_indices.dtype, device=sorted_indices.device)
    expert_ids = torch.tensor(padded_experts, dtype=torch.int32, device=sorted_indices.device)
    num_tokens_post_padded = torch.tensor([sorted_token_ids.size(0)], dtype=torch.int32,
                                           device=sorted_indices.device)

    return sorted_token_ids, expert_ids, num_tokens_post_padded


def _assert_equivalent(topk_ids, block_size, num_experts, label):
    ref_ids, ref_experts, ref_padded = _reference_moe_align_block_size(topk_ids, block_size, num_experts)
    new_ids, new_experts, new_padded = moe_align_block_size(topk_ids, block_size, num_experts)

    assert ref_ids.dtype == new_ids.dtype, f"[{label}] sorted_token_ids dtype: {ref_ids.dtype} vs {new_ids.dtype}"
    assert ref_ids.shape == new_ids.shape, f"[{label}] sorted_token_ids shape: {ref_ids.shape} vs {new_ids.shape}"
    assert torch.equal(ref_ids, new_ids), f"[{label}] sorted_token_ids values differ"

    assert ref_experts.dtype == new_experts.dtype, f"[{label}] expert_ids dtype mismatch"
    assert ref_experts.shape == new_experts.shape, f"[{label}] expert_ids shape: {ref_experts.shape} vs {new_experts.shape}"
    assert torch.equal(ref_experts, new_experts), f"[{label}] expert_ids values differ"

    assert torch.equal(ref_padded, new_padded), f"[{label}] num_tokens_post_padded: {ref_padded} vs {new_padded}"


def test_moe_align_block_size_matches_reference_randomized():
    device = "cpu"
    configs = [
        # (num_tokens, top_k, num_experts, block_size)
        (1, 1, 1, 16),
        (5, 2, 8, 16),
        (17, 4, 16, 16),
        (64, 8, 64, 16),
        (64, 8, 64, 32),
        (128, 8, 64, 64),
        (257, 8, 64, 16),
    ]
    seeds = range(5)
    cases_run = 0
    for (num_tokens, top_k, num_experts, block_size) in configs:
        for seed in seeds:
            g = torch.Generator().manual_seed(seed)
            # Each token's top_k experts are distinct, matching real torch.topk output.
            topk_ids = torch.stack([
                torch.randperm(num_experts, generator=g)[:top_k] for _ in range(num_tokens)
            ]).to(device)
            _assert_equivalent(
                topk_ids, block_size, num_experts,
                label=f"random T={num_tokens} k={top_k} E={num_experts} bs={block_size} seed={seed}",
            )
            cases_run += 1
    print(f"test_moe_align_block_size_matches_reference_randomized: {cases_run} cases passed")


def test_moe_align_block_size_edge_cases():
    device = "cpu"

    # Exact block-size multiple: no padding needed anywhere.
    topk_ids = torch.tensor([[0, 1], [1, 2], [0, 2]], device=device)
    _assert_equivalent(topk_ids, block_size=2, num_experts=3, label="exact-multiple, no padding")

    # Single-expert concentration: every token routes to expert 0 only.
    # Exercises heavy padding plus many zero-count ("vanishing") experts.
    topk_ids = torch.zeros((6, 1), dtype=torch.long, device=device)
    _assert_equivalent(topk_ids, block_size=4, num_experts=8, label="single-expert concentration")

    # More experts than tokens*top_k: most experts get zero tokens.
    topk_ids = torch.tensor([[3]], device=device)
    _assert_equivalent(topk_ids, block_size=16, num_experts=64, label="mostly-empty experts")

    # block_size larger than any single expert's count -> every expert pads up to one full block.
    topk_ids = torch.tensor([[0], [1], [2], [3]], device=device)
    _assert_equivalent(topk_ids, block_size=64, num_experts=4, label="block_size >> per-expert count")

    # Empty input.
    topk_ids = torch.empty((0, 8), dtype=torch.long, device=device)
    _assert_equivalent(topk_ids, block_size=16, num_experts=64, label="empty input")

    print("test_moe_align_block_size_edge_cases: all edge cases passed")


def test_fused_moe_end_to_end():
    if not torch.cuda.is_available():
        print("test_fused_moe_end_to_end: SKIPPED (no CUDA GPU available; Triton requires CUDA)")
        return

    from dminfr.engine.fused_moe_triton import fused_moe

    torch.manual_seed(0)
    device = "cuda"
    M, K, E, EI, top_k = 37, 256, 16, 128, 4  # deliberately non-round M to stress padding

    hidden_states = torch.randn(M, K, device=device, dtype=torch.bfloat16)
    w1 = torch.randn(E, 2 * EI, K, device=device, dtype=torch.bfloat16) * 0.02
    w2 = torch.randn(E, K, EI, device=device, dtype=torch.bfloat16) * 0.02

    router_logits = torch.randn(M, E, device=device, dtype=torch.float32)
    routing_weights = torch.softmax(router_logits, dim=-1)
    topk_weights, topk_ids = torch.topk(routing_weights, top_k, dim=-1)

    out = fused_moe(
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        gating_output=topk_weights.to(hidden_states.dtype),
        topk_ids=topk_ids.to(torch.int32),
    )

    assert out.shape == (M, K), f"expected shape {(M, K)}, got {tuple(out.shape)}"
    assert torch.isfinite(out).all(), "fused_moe output contains NaN/Inf"

    # Naive per-expert reference in plain PyTorch (no Triton), same math.
    ref = torch.zeros_like(hidden_states, dtype=torch.float32)
    for e in range(E):
        mask = (topk_ids == e)  # [M, top_k]
        if not mask.any():
            continue
        token_idx, slot_idx = torch.where(mask)
        x = hidden_states[token_idx].float()
        gate, up = (x @ w1[e].float().T).chunk(2, dim=-1)
        h = (torch.nn.functional.silu(gate) * up) @ w2[e].float().T
        weight = topk_weights[token_idx, slot_idx].float()
        ref.index_add_(0, token_idx, h * weight.unsqueeze(-1))

    cos_sim = torch.nn.functional.cosine_similarity(
        out.float().flatten(), ref.flatten(), dim=0
    ).item()
    assert cos_sim > 0.999, f"fused_moe vs naive reference cosine similarity too low: {cos_sim:.6f}"

    print(f"test_fused_moe_end_to_end: PASSED (shape={tuple(out.shape)}, cosine_sim={cos_sim:.6f})")


if __name__ == "__main__":
    test_moe_align_block_size_matches_reference_randomized()
    test_moe_align_block_size_edge_cases()
    test_fused_moe_end_to_end()
    print("\nAll tests passed.")
