"""
Integration test for model_update/generate.py's opt-in confidence_threshold
decoding path (_generate_block_cached / generate_cached's
confidence_threshold + low_confidence_threshold parameters, using
select_transfer_indices_hierarchy -- ported from dInfer's HierarchyDecoder,
see that function's docstring for why), on top of the CPU-only
selection-logic tests in eval/test_select_transfer_indices.py.

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
(eval/correctness/run_correctness.py) with confidence_threshold enabled on
the actual 7B model and real weights -- this only proves the mechanism is
safe (bounded step count, no leftover MASK tokens), not that the resulting
generations are still good text.
"""

import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_update.model import LLaDAMoEKV, SMALL_CFG, KVCacheBuffer
from model_update.generate import generate_cached, MASK_ID


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


def test_remask_path_always_uses_full_budget_and_completes():
    """remask_threshold's safety properties on the real (small) model: never
    exceeds the step budget, never leaves MASK tokens, and -- unlike plain
    hierarchy decoding -- ALWAYS uses the full step budget, since early
    exit is unconditionally disabled once remasking is active (a row with
    nothing currently masked can still have shaky already-revealed content
    worth reconsidering)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _build_model(device)
    cfg = SMALL_CFG

    B, P = 2, 8
    gen_length, steps, block_length = 16, 8, 8
    num_blocks = gen_length // block_length
    steps_per_block = steps // num_blocks
    max_calls = 1 + num_blocks * (steps_per_block + 1)

    prompt_ids = torch.randint(0, cfg.VS, (B, P), device=device)

    with torch.no_grad():
        out, call_count = _count_forward_calls(
            model,
            lambda: generate_cached(
                model, prompt_ids, gen_length=gen_length, steps=steps,
                block_length=block_length, confidence_threshold=0.9,
                low_confidence_threshold=0.4, remask_threshold=0.4,
            ),
        )

    assert call_count == max_calls, (
        f"remasking must always use the FULL step budget (early exit disabled), "
        f"got {call_count} calls, expected exactly {max_calls}."
    )
    assert not (out == MASK_ID).any(), (
        "leftover MASK tokens with remasking enabled -- the last-step force-reveal "
        "(which disables remasking) did not fire correctly."
    )
    print(f"test_remask_path_always_uses_full_budget_and_completes: PASSED "
          f"({call_count}/{max_calls} calls, no leftover MASK tokens)")


def _mock_forward_sequence(logits_list):
    """Replaces model.forward with a function that pops one pre-built [B,T,VS]
    logits tensor per call, ignoring the real input_ids/cache_buffer/KV state
    entirely. Used to deterministically control exactly what confidence/argmax
    the remasking logic sees at each step, independent of what a real (even a
    tiny, randomly-initialized) model would actually predict -- needed because
    proving "position X gets reverted because position Y outcompetes it for
    the top1 slot" requires precise, engineered confidence values, not
    whatever an untrained model happens to produce."""
    call_idx = [0]

    def forward(input_ids, position_offset=0, past_kv=None, cache_buffer=None, write_pos=None):
        logits = logits_list[call_idx[0]]
        call_idx[0] += 1
        return logits, None

    return forward


def _controlled_logits(T, VS, device, dtype, picks):
    """Builds a [1, T, VS] logits tensor such that argmax(dim=-1) and the
    softmax confidence of that argmax exactly match `picks` at each
    specified position (B=1 only, for these tests).

    picks: {position: (token_id, confidence)}. Positions not listed get a
    flat/uniform distribution (argmax = token 0, confidence = 1/VS) --
    irrelevant filler for positions a given test doesn't care about.

    Construction: a "two-level" logit setup -- logits[token] = log(conf),
    logits[every other token] = log((1-conf)/(VS-1)) -- gives EXACTLY
    `conf` at the target token after softmax and every other token equal
    probability, for any 1/VS < conf < 1.
    """
    import math
    logits = torch.zeros(1, T, VS, dtype=dtype, device=device)
    for pos, (token, conf) in picks.items():
        other = (1.0 - conf) / (VS - 1)
        logits[0, pos, :] = math.log(other)
        logits[0, pos, token] = math.log(conf)
    return logits


def test_remask_reverts_a_position_that_loses_the_selection_competition():
    """The critical, easy-to-get-wrong case: a shaky already-revealed
    position must actually get reverted to MASK when it loses out to a
    higher-confidence competing position in the same step -- not merely
    "sometimes not re-picked," which the unconditional top1-include
    guarantee could otherwise mask (a LONE remask candidate always wins by
    default, since there's nothing to lose to -- this only shows up with a
    genuine competitor)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _build_model(device)
    cfg = SMALL_CFG
    dtype = next(model.parameters()).dtype

    B, P = 1, 4
    gen_length = block_length = 3  # single block, 3 positions
    steps = steps_per_block = 3
    confidence_threshold, low_confidence_threshold, remask_threshold = 0.9, 0.4, 0.4

    prompt_ids = torch.randint(0, cfg.VS, (B, P), device=device)

    TOK_0_INITIAL, TOK_1, TOK_2_REVEAL, TOK_0_FINAL = 5, 8, 20, 99

    # Step 0 (prime call is separate/unmocked-shape-only -- see below):
    # positions 0 and 1 clear confidence_threshold and get revealed;
    # position 2 (0.3, well under 0.9) stays masked.
    step0 = _controlled_logits(block_length, cfg.VS, device, dtype, {
        0: (TOK_0_INITIAL, 0.96), 1: (TOK_1, 0.95), 2: (999, 0.3),
    })
    # Step 1 (NOT the last step): position 0's confidence in its OWN
    # existing token collapses to 0.1 (well under remask_threshold=0.4) --
    # flagged shaky. Position 1 stays stable at 0.95 (not shaky). Position
    # 2 (still masked from step 0) gets revealed at 0.85. Position 2's
    # higher confidence (0.85 vs 0.1) wins the top1 slot -- position 0 has
    # no other way to get selected (0.1 clears neither low_threshold=0.4
    # nor confidence_threshold=0.9), so it must be reverted.
    step1 = _controlled_logits(block_length, cfg.VS, device, dtype, {
        0: (TOK_0_INITIAL, 0.1), 1: (TOK_1, 0.95), 2: (TOK_2_REVEAL, 0.85),
    })
    # Step 2 (the last step -- force-reveal, remasking disabled): whatever
    # is still masked (position 0, reverted in step 1) gets a fresh
    # prediction, force-selected regardless of confidence.
    step2 = _controlled_logits(block_length, cfg.VS, device, dtype, {
        0: (TOK_0_FINAL, 0.5), 1: (TOK_1, 0.95), 2: (TOK_2_REVEAL, 0.95),
    })

    # generate_cached's prime call runs over the FULL prompt+gen region
    # (shape [B, P+gen_length]), a different width than the 3 per-step
    # calls above ([B, block_length]) -- needs its own matching-shape entry.
    prime = _controlled_logits(P + gen_length, cfg.VS, device, dtype, {})
    # The block's finalize call (once, after the step loop) also runs over
    # the full remaining suffix width.
    finalize = _controlled_logits(gen_length, cfg.VS, device, dtype, {})

    model.forward = _mock_forward_sequence([prime, step0, step1, step2, finalize])

    with torch.no_grad():
        out = generate_cached(
            model, prompt_ids, gen_length=gen_length, steps=steps, block_length=block_length,
            confidence_threshold=confidence_threshold, low_confidence_threshold=low_confidence_threshold,
            remask_threshold=remask_threshold,
        )

    got = out[0].tolist()
    assert got == [TOK_0_FINAL, TOK_1, TOK_2_REVEAL], (
        f"expected [{TOK_0_FINAL}, {TOK_1}, {TOK_2_REVEAL}] (position 0 reverted in step 1 "
        f"then force-revealed differently in step 2, position 1 stable throughout, position "
        f"2 revealed once in step 1), got {got}"
    )
    print(f"test_remask_reverts_a_position_that_loses_the_selection_competition: PASSED "
          f"(position 0 went {TOK_0_INITIAL} -> MASK -> {TOK_0_FINAL} as expected)")


def test_remask_revises_a_reselected_position_to_a_new_value():
    """A shaky position that DOES get reselected (not reverted) is REVISED
    to the model's current top pick, not merely re-confirmed at its old
    value -- matching dInfer's reference (raw, unmodified argmax throughout,
    not a mask_index-preserved version). Uses a single-position block, so
    the shaky candidate is always the sole selectable position and wins via
    the unconditional top1-include guarantee -- deliberately set
    low_confidence_threshold high enough that this position would FAIL the
    ordinary low_threshold filter, isolating that the top1 override is what
    saves (and revises) it, not low_threshold clearing on its own."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _build_model(device)
    cfg = SMALL_CFG
    dtype = next(model.parameters()).dtype

    B, P = 1, 4
    gen_length = block_length = 1  # single block, single position
    steps = steps_per_block = 3
    confidence_threshold, low_confidence_threshold, remask_threshold = 0.9, 0.7, 0.6

    prompt_ids = torch.randint(0, cfg.VS, (B, P), device=device)

    TOK_INITIAL, TOK_REVISED = 5, 7

    step0 = _controlled_logits(block_length, cfg.VS, device, dtype, {0: (TOK_INITIAL, 0.95)})
    # Step 1 (not last): confidence 0.5 is < remask_threshold=0.6 (flagged
    # shaky) AND < low_confidence_threshold=0.7 (would fail low_threshold
    # alone) AND < confidence_threshold=0.9 (doesn't clear via threshold
    # union either) -- the ONLY way this gets reselected is the
    # unconditional top1-include step, since it's the sole candidate.
    step1 = _controlled_logits(block_length, cfg.VS, device, dtype, {0: (TOK_REVISED, 0.5)})
    step2 = _controlled_logits(block_length, cfg.VS, device, dtype, {0: (TOK_REVISED, 0.95)})
    prime = _controlled_logits(P + gen_length, cfg.VS, device, dtype, {})
    finalize = _controlled_logits(gen_length, cfg.VS, device, dtype, {})

    model.forward = _mock_forward_sequence([prime, step0, step1, step2, finalize])

    with torch.no_grad():
        out = generate_cached(
            model, prompt_ids, gen_length=gen_length, steps=steps, block_length=block_length,
            confidence_threshold=confidence_threshold, low_confidence_threshold=low_confidence_threshold,
            remask_threshold=remask_threshold,
        )

    got = out[0].tolist()
    assert got == [TOK_REVISED], (
        f"expected [{TOK_REVISED}] (revised directly from {TOK_INITIAL} in step 1, without "
        f"ever passing through an explicit MASK state), got {got}"
    )
    print(f"test_remask_revises_a_reselected_position_to_a_new_value: PASSED "
          f"(position 0 went {TOK_INITIAL} -> {TOK_REVISED} directly, no revert in between)")


if __name__ == "__main__":
    test_default_path_never_early_exits()
    test_threshold_path_bounded_and_complete()
    test_remask_path_always_uses_full_budget_and_completes()
    test_remask_reverts_a_position_that_loses_the_selection_competition()
    test_remask_revises_a_reselected_position_to_a_new_value()
    print("\nAll tests passed.")
