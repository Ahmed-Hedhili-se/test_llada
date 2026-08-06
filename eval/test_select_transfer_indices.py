"""
Regression test for model_update/generate.py's vectorized
get_num_transfer_tokens() and select_transfer_indices().

Both originally used a per-batch-row Python loop with an implicit or
explicit `.item()` call -- a host-device sync per row, per step for
select_transfer_indices (called once per denoising step) and per row, per
block for get_num_transfer_tokens (called once per block). A Chrome-trace
profiling investigation of a batched (B=57) generation run found ~19,700
idle GPU gaps averaging ~8us each -- the signature of exactly this kind of
per-row host-sync overhead, not a few large stalls. Both were rewritten to
compute the same result with a single batched operation instead of a loop.

This file keeps frozen copies of the original loop-based implementations as
reference oracles and asserts the vectorized versions produce identical
output across randomized configs and edge cases: B=1 (the TP+EP path
always uses this -- must match exactly, not just "close"), all-zero
transfer counts, ties in confidence values (the trickiest case for
topk-consistency between a single batched call and independent per-row
calls), and mismatched per-row counts within one batch (what batched
concurrent-request serving actually produces, since different requests can
have different numbers of masked positions remaining).

Pure tensor math, no model/CUDA dependency -- runs on CPU.
"""

import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_update.generate import get_num_transfer_tokens, select_transfer_indices


def _reference_get_num_transfer_tokens(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    """Frozen copy of the pre-vectorization implementation."""
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer = torch.zeros(
        mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64
    ) + base
    for i in range(mask_num.size(0)):
        num_transfer[i, : remainder[i]] += 1
    return num_transfer


def _reference_select_transfer_indices(confidence: torch.Tensor, num_transfer_step: torch.Tensor) -> torch.Tensor:
    """Frozen copy of the pre-vectorization implementation (the per-row
    loop body that used to live inline in _generate_block_cached)."""
    transfer_index = torch.zeros(confidence.shape, dtype=torch.bool)
    for j in range(confidence.shape[0]):
        k = num_transfer_step[j].item()
        if k > 0:
            _, sel = torch.topk(confidence[j], k=int(k))
            transfer_index[j, sel] = True
    return transfer_index


def test_get_num_transfer_tokens_matches_reference():
    torch.manual_seed(0)
    configs = [
        # (B, T, steps)
        (1, 16, 16), (1, 32, 8), (1, 7, 3),
        (2, 16, 16), (5, 32, 8), (8, 64, 32), (17, 16, 4),
    ]
    cases = 0
    for B, T, steps in configs:
        for _ in range(5):
            mask_index = torch.rand(B, T) < torch.rand(1).item()  # varying mask density
            ref = _reference_get_num_transfer_tokens(mask_index, steps)
            new = get_num_transfer_tokens(mask_index, steps)
            assert ref.dtype == new.dtype, f"dtype mismatch: {ref.dtype} vs {new.dtype}"
            assert ref.shape == new.shape, f"shape mismatch: {ref.shape} vs {new.shape}"
            assert torch.equal(ref, new), f"values differ for B={B},T={T},steps={steps}\nref={ref}\nnew={new}"
            cases += 1

    # Edge cases: no masked tokens at all, and every token masked.
    for mask_index in [torch.zeros(4, 16, dtype=torch.bool), torch.ones(4, 16, dtype=torch.bool)]:
        ref = _reference_get_num_transfer_tokens(mask_index, 8)
        new = get_num_transfer_tokens(mask_index, 8)
        assert torch.equal(ref, new)
        cases += 1

    print(f"test_get_num_transfer_tokens_matches_reference: {cases} cases passed")


def test_select_transfer_indices_matches_reference():
    torch.manual_seed(0)
    cases = 0

    # Randomized: varying batch size, active-block length, and per-row k
    # (including rows with k=0 while others in the same batch have k>0 --
    # exactly what concurrent batched requests with different remaining
    # mask counts produce).
    for B, T in [(1, 8), (1, 32), (2, 16), (5, 32), (8, 64), (16, 16)]:
        for _ in range(8):
            confidence = torch.rand(B, T)
            # Randomly mask out some positions with -inf, like real usage
            # (already-revealed positions aren't selectable).
            mask = torch.rand(B, T) < 0.3
            confidence = torch.where(mask, torch.full_like(confidence, -torch.inf), confidence)
            max_valid = (~mask).sum(dim=1)
            num_transfer_step = torch.stack([
                torch.randint(0, int(m.item()) + 1, (1,)).squeeze(0) if m > 0 else torch.tensor(0)
                for m in max_valid
            ])

            ref = _reference_select_transfer_indices(confidence, num_transfer_step)
            new = select_transfer_indices(confidence, num_transfer_step)
            assert ref.shape == new.shape
            assert torch.equal(ref, new), (
                f"mismatch for B={B} T={T}\n"
                f"num_transfer_step={num_transfer_step}\nconfidence={confidence}\n"
                f"ref={ref}\nnew={new}"
            )
            cases += 1

    print(f"test_select_transfer_indices_matches_reference: {cases} randomized cases passed")


def test_select_transfer_indices_edge_cases():
    # All rows k=0: nothing selected anywhere.
    confidence = torch.rand(4, 10)
    num_transfer_step = torch.zeros(4, dtype=torch.int64)
    ref = _reference_select_transfer_indices(confidence, num_transfer_step)
    new = select_transfer_indices(confidence, num_transfer_step)
    assert torch.equal(ref, new)
    assert not new.any()

    # Single row (B=1) -- the shape the TP+EP path always uses. Must match
    # exactly, not approximately, since this path has no batching to hide
    # behind.
    confidence = torch.tensor([[0.9, 0.1, 0.5, -torch.inf, 0.7]])
    num_transfer_step = torch.tensor([2])
    ref = _reference_select_transfer_indices(confidence, num_transfer_step)
    new = select_transfer_indices(confidence, num_transfer_step)
    assert torch.equal(ref, new)
    # Top-2 of [0.9, 0.1, 0.5, -inf, 0.7] are positions 0 (0.9) and 4 (0.7).
    assert new.tolist() == [[True, False, False, False, True]]

    # Exact ties in confidence -- the trickiest case for "does a single
    # batched topk(k=max_k) agree with independent per-row topk(k=k_j)
    # calls on tie-breaking." A batched torch.topk(dim=-1) processes each
    # row independently regardless of batch context, so this should hold,
    # but it's cheap to verify directly rather than assume it.
    confidence = torch.tensor([
        [0.5, 0.5, 0.5, 0.1],  # three-way tie for the top spot(s)
        [0.9, 0.9, 0.2, 0.2],  # two ties
    ])
    for k0 in range(5):
        for k1 in range(5):
            num_transfer_step = torch.tensor([k0, k1])
            ref = _reference_select_transfer_indices(confidence, num_transfer_step)
            new = select_transfer_indices(confidence, num_transfer_step)
            assert torch.equal(ref, new), f"tie-break mismatch at k=({k0},{k1})\nref={ref}\nnew={new}"
            # Sanity: number of selected positions per row equals k (capped at row width).
            assert ref[0].sum().item() == min(k0, 4)
            assert ref[1].sum().item() == min(k1, 4)

    # Mismatched per-row k within one batch: one row needs many, another
    # needs none -- exactly what concurrent requests at different
    # denoising progress produce when batched together.
    confidence = torch.rand(3, 20)
    num_transfer_step = torch.tensor([0, 5, 20])
    ref = _reference_select_transfer_indices(confidence, num_transfer_step)
    new = select_transfer_indices(confidence, num_transfer_step)
    assert torch.equal(ref, new)
    assert ref[0].sum().item() == 0
    assert ref[1].sum().item() == 5
    assert ref[2].sum().item() == 20

    print("test_select_transfer_indices_edge_cases: all edge cases passed")


if __name__ == "__main__":
    test_get_num_transfer_tokens_matches_reference()
    test_select_transfer_indices_matches_reference()
    test_select_transfer_indices_edge_cases()
    print("\nAll tests passed.")
