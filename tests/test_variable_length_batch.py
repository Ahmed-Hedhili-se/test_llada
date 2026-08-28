"""Batching prompts of different lengths.

Until now `dminfr/serving/server.py` could only group requests whose prompts tokenized to
exactly the same length, because `Attention.forward` called
`scaled_dot_product_attention(..., attn_mask=None)` and therefore could not
tolerate padding. Under real traffic that fragments the queue into batches of
one or two -- which is why every throughput figure in README.md was measured
with `--fixed-prompt`.

`generate_cached(..., prompt_lens=...)` now accepts LEFT-padded prompts. Left,
not right, because the block loop walks `block_start = P + block_idx *
block_length` once for the whole batch, so every row's prompt has to END at the
same column.

The load-bearing test is :func:`test_padded_row_matches_unpadded_run`. Padding
must be *inert*: a prompt batched next to a longer one has to produce exactly
what it produces alone. Two separate mechanisms have to hold for that, and
either failing silently degrades output rather than raising:

  * per-row RoPE positions -- otherwise a padded prompt is encoded as though it
    started partway into the sequence, which is a different input;
  * the additive attention mask -- otherwise queries attend to pad columns,
    which in the cached path are uninitialised `torch.empty` KV slots.

The remaining tests pin the pieces that would let a regression pass the first
one for the wrong reason.

Needs a CUDA GPU (the fused MoE path is Triton-only); the eager path runs on
CPU and is used where possible so the checks stay cheap.
"""

import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dminfr.engine.generate import MASK_ID, generate_cached
from dminfr.engine.model import (
    Cfg,
    LLaDAMoEKV,
    build_rope_freqs,
    build_rope_freqs_from_positions,
)

TEST_CFG = Cfg(H=64, NH=4, KVH=4, NL=2, NE=8, TOPK=2, EI=32, VS=157184)


def _build(device, dtype=torch.float32):
    torch.manual_seed(0)
    return LLaDAMoEKV(TEST_CFG, use_fused_moe=False).to(dtype).to(device).eval()


def _pad_left(ids, width, pad_id=0):
    pad = width - ids.shape[1]
    if pad <= 0:
        return ids
    return torch.cat(
        [torch.full((ids.shape[0], pad), pad_id, dtype=ids.dtype, device=ids.device), ids],
        dim=1,
    )


# ---------------------------------------------------------------------------

def test_rope_positions_agree_with_the_unpadded_range():
    """The per-row builder must reproduce the shared one when there is no pad.

    If these two ever disagree, every equal-length batch silently changes
    behaviour -- so this guards the fast path as much as the new one.
    """
    hd, theta = TEST_CFG.HD, TEST_CFG.THETA
    cos_a, sin_a = build_rope_freqs(0, 12, hd, theta, torch.device("cpu"))
    positions = torch.arange(12).unsqueeze(0)
    cos_b, sin_b = build_rope_freqs_from_positions(positions, hd, theta)

    assert torch.equal(cos_a, cos_b[0]), "cos differs between the two RoPE builders"
    assert torch.equal(sin_a, sin_b[0]), "sin differs between the two RoPE builders"


@torch.no_grad()
def test_padded_row_matches_unpadded_run():
    """A short prompt batched beside a long one == that prompt run alone.

    This is the whole point of the feature. Greedy decoding, so the comparison
    is exact token equality rather than a tolerance.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _build(device)

    torch.manual_seed(7)
    short = torch.randint(0, 1000, (1, 5), device=device)
    long_ = torch.randint(0, 1000, (1, 11), device=device)
    gen_kwargs = dict(gen_length=8, steps=8, block_length=4, temperature=0.0)

    alone_short = generate_cached(model, short.clone(), **gen_kwargs)
    alone_long = generate_cached(model, long_.clone(), **gen_kwargs)

    width = max(short.shape[1], long_.shape[1])
    batched_ids = torch.cat([_pad_left(short, width), _pad_left(long_, width)], dim=0)
    prompt_lens = torch.tensor([short.shape[1], long_.shape[1]], device=device)
    batched = generate_cached(model, batched_ids, prompt_lens=prompt_lens, **gen_kwargs)

    assert batched.shape == (2, 8), batched.shape
    assert torch.equal(batched[0], alone_short[0]), (
        f"padded row diverged from its solo run\n"
        f"  batched: {batched[0].tolist()}\n  alone:   {alone_short[0].tolist()}"
    )
    assert torch.equal(batched[1], alone_long[0]), "unpadded row diverged"


@torch.no_grad()
def test_equal_lengths_are_untouched_by_the_new_path():
    """prompt_lens on an already-equal batch must change nothing.

    generate_cached drops back to the original code path when no row actually
    needs padding; this asserts that shortcut is real.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _build(device)

    torch.manual_seed(11)
    ids = torch.randint(0, 1000, (3, 7), device=device)
    gen_kwargs = dict(gen_length=8, steps=8, block_length=8, temperature=0.0)

    without = generate_cached(model, ids.clone(), **gen_kwargs)
    with_lens = generate_cached(
        model, ids.clone(), prompt_lens=torch.full((3,), 7, device=device), **gen_kwargs
    )
    assert torch.equal(without, with_lens), "equal-length batch changed when prompt_lens was passed"


@torch.no_grad()
def test_pad_columns_are_not_attended():
    """Garbage in the pad slots must not reach the output.

    Uses a pad token id far outside anything the real prompts contain: if the
    mask leaked, the embedding of that id would perturb attention and the row
    would stop matching its solo run.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _build(device)

    torch.manual_seed(3)
    short = torch.randint(0, 1000, (1, 4), device=device)
    long_ = torch.randint(0, 1000, (1, 10), device=device)
    gen_kwargs = dict(gen_length=8, steps=8, block_length=4, temperature=0.0)
    alone = generate_cached(model, short.clone(), **gen_kwargs)

    prompt_lens = torch.tensor([4, 10], device=device)
    for pad_id in (0, 999, MASK_ID):
        ids = torch.cat([_pad_left(short, 10, pad_id=pad_id), long_], dim=0)
        out = generate_cached(model, ids, prompt_lens=prompt_lens, **gen_kwargs)
        assert torch.equal(out[0], alone[0]), (
            f"pad_id={pad_id} leaked into the output -- the attention mask is "
            "not covering the pad columns"
        )


def test_rejects_prompt_lens_longer_than_the_padded_width():
    """A caller that passes unpadded ids should be told, not silently mis-run."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _build(device)
    ids = torch.randint(0, 1000, (2, 6), device=device)

    try:
        generate_cached(model, ids, prompt_lens=torch.tensor([6, 9], device=device),
                        gen_length=4, steps=4, block_length=4, temperature=0.0)
    except ValueError as exc:
        assert "left-padded" in str(exc)
    else:
        raise AssertionError("expected ValueError for prompt_lens beyond the padded width")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    for fn in (
        test_rope_positions_agree_with_the_unpadded_range,
        test_padded_row_matches_unpadded_run,
        test_equal_lengths_are_untouched_by_the_new_path,
        test_pad_columns_are_not_attended,
        test_rejects_prompt_lens_longer_than_the_padded_width,
    ):
        fn()
        print(f"  [ok] {fn.__name__}")
    print("\nPASS - padding is inert: a batched row matches its solo run exactly.")


if __name__ == "__main__":
    main()
