"""
Integration test for dminfr/engine/generate.py's opt-in confidence_threshold
decoding path (_generate_block_cached / generate_cached's
confidence_threshold + low_confidence_threshold parameters, using
select_transfer_indices_hierarchy -- ported from dInfer's HierarchyDecoder,
see that function's docstring for why), on top of the CPU-only
selection-logic tests in tests/test_select_transfer_indices.py.

Two things checked, each catching a different failure mode:

  1. confidence_threshold=None (the default) never actually triggers the
     early-exit check added to the block loop. That check is unconditional
     (added regardless of whether a threshold is given), so it's the one
     thing that could silently change default behavior -- the fixed
     schedule (get_num_transfer_tokens) spreads reveals evenly across every
     allotted step by construction, so mask_index can only become all-False
     on the loop's own last iteration, never earlier, but that's an
     argument, not a test. Checked directly here by counting real model()
     forward calls: for confidence_threshold=None, the count must be
     EXACTLY steps_per_block per block, never fewer.

  2. confidence_threshold=<value> (a) never calls the model more than
     steps_per_block times for a single block -- the step-count ceiling
     from `steps` must never be exceeded regardless of how the threshold
     behaves -- and (b) never leaves a MASK_ID token in the output -- the
     last-step force-reveal safety net must actually fire. Checked across
     three thresholds: 0.0 (every position trivially qualifies immediately,
     so real early-exit savings should appear), 1.1 (impossible to ever
     reach since softmax probabilities are <= 1.0, so with only a single
     contiguous masked segment per block at the start, hierarchy decoding
     reduces to "one reveal per step" here too -- the same worst case the
     step budget's force-reveal safety net has to cover), and 0.9 (dInfer's
     own default value for this exact model).

Uses eager (non-Triton) MoE -- device-agnostic, unlike the fused Triton
path tested in test_cuda_graph_forward.py -- so this runs on CPU or CUDA,
whichever is available. NOT a substitute for a real accuracy check
(benchmarks/correctness/run_correctness.py) with confidence_threshold enabled on
the actual 7B model and real weights -- this only proves the mechanism is
safe (bounded step count, no leftover MASK tokens), not that the resulting
generations are still good text.
"""

import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dminfr.engine.model import LLaDAMoEKV, SMALL_CFG, KVCacheBuffer
from dminfr.engine.generate import generate_cached, MASK_ID


def _build_model(device):
    torch.manual_seed(0)
    model = LLaDAMoEKV(SMALL_CFG, use_fused_moe=False).to(torch.float32).to(device).eval()
    return model


def _count_forward_calls(model, fn):
    """Wraps model.forward to count real invocations for the duration of fn(),
    then restores it -- avoids needing any instrumentation inside
    generate.py itself."""
    call_count = [0]
    orig_forward = model.forward

    def counting_forward(*args, **kwargs):
        call_count[0] += 1
        return orig_forward(*args, **kwargs)

    model.forward = counting_forward
    try:
        result = fn()
    finally:
        model.forward = orig_forward
    return result, call_count[0]


def test_default_path_never_early_exits():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _build_model(device)
    cfg = SMALL_CFG

    B, P = 2, 8
    gen_length, steps, block_length = 16, 8, 8
    num_blocks = gen_length // block_length
    steps_per_block = steps // num_blocks

    prompt_ids = torch.randint(0, cfg.VS, (B, P), device=device)

    with torch.no_grad():
        _, call_count = _count_forward_calls(
            model,
            lambda: generate_cached(
                model, prompt_ids, gen_length=gen_length, steps=steps,
                block_length=block_length, confidence_threshold=None,
            ),
        )

    # 1 prime call (once per generate_cached call) + per block: steps_per_block
    # per-step calls + 1 finalize call.
    expected = 1 + num_blocks * (steps_per_block + 1)
    assert call_count == expected, (
        f"confidence_threshold=None made {call_count} forward calls, expected exactly "
        f"{expected} -- the early-exit check must never trigger for the default path "
        f"(if it did, this is fewer than expected, meaning default behavior silently changed)."
    )
    print(f"test_default_path_never_early_exits: PASSED ({call_count} calls, matches "
          f"1 + {num_blocks}*({steps_per_block}+1) exactly)")


def test_threshold_path_bounded_and_complete():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _build_model(device)
    cfg = SMALL_CFG

    B, P = 2, 8
    gen_length, steps, block_length = 16, 8, 8
    num_blocks = gen_length // block_length
    steps_per_block = steps // num_blocks
    max_calls = 1 + num_blocks * (steps_per_block + 1)  # same ceiling as the default path

    prompt_ids = torch.randint(0, cfg.VS, (B, P), device=device)

    for threshold in [0.0, 1.1, 0.9]:
        with torch.no_grad():
            out, call_count = _count_forward_calls(
                model,
                lambda: generate_cached(
                    model, prompt_ids, gen_length=gen_length, steps=steps,
                    block_length=block_length, confidence_threshold=threshold,
                    low_confidence_threshold=0.4,
                ),
            )

        assert call_count <= max_calls, (
            f"[threshold={threshold}] made {call_count} forward calls, exceeding the "
            f"ceiling of {max_calls} -- the step-count budget from `steps` was violated."
        )
        assert not (out == MASK_ID).any(), (
            f"[threshold={threshold}] output still contains MASK_ID tokens -- the "
            f"last-step force-reveal safety net did not fire correctly."
        )
        print(f"test_threshold_path_bounded_and_complete: threshold={threshold} PASSED "
              f"({call_count}/{max_calls} calls, no leftover MASK tokens)")

    # threshold=0.0 should show REAL savings (every position trivially
    # qualifies every step via the threshold union, so each block should
    # finish in far fewer than steps_per_block calls) -- not just "within
    # budget" but actually exercising the early-exit path.
    with torch.no_grad():
        _, trivial_call_count = _count_forward_calls(
            model,
            lambda: generate_cached(
                model, prompt_ids, gen_length=gen_length, steps=steps,
                block_length=block_length, confidence_threshold=0.0,
                low_confidence_threshold=0.4,
            ),
        )
    assert trivial_call_count < max_calls, (
        f"threshold=0.0 (trivially-always-qualifies) made {trivial_call_count} calls, "
        f"the same as the full budget of {max_calls} -- expected real early-exit savings."
    )
    print(f"test_threshold_path_bounded_and_complete: confirmed threshold=0.0 exercises "
          f"real early-exit savings ({trivial_call_count} < {max_calls})")


if __name__ == "__main__":
    test_default_path_never_early_exits()
    test_threshold_path_bounded_and_complete()
    print("\nAll tests passed.")
