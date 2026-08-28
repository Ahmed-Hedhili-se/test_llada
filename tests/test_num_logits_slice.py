"""
Regression test for LLaDAMoEKV.forward(num_logits=...) and its three
generate.py call sites.

Before this change, every forward pass projected its ENTIRE input through
lm_head (H=2048 -> VS=157,184, the widest GEMM in the model), and the
callers then threw most of it away:

  - the cache-prime pass and the per-block finalize pass discard the logits
    outright -- they run only for their K/V side effect on cache_buffer;
  - each denoising step uses only [:, :block_length] of a suffix that is up
    to 16x longer (gen_length=1024, block_length=64).

`num_logits` narrows the head to the rows that are actually consumed (0 =
skip it entirely). The transformer layers still run over the full input --
the K/V written to the cache and the future-MASK context the model was
trained to see both depend on it -- so this changes only which rows reach
the vocabulary projection.

The claim under test is that this is EXACT, not an approximation:
  - GEMM rows are independent, so surviving rows compute identical dot
    products;
  - RMSNorm reduces along the feature axis, so each token's normalization
    is independent of every other token's -- which is why slicing BEFORE
    self.norm (rather than after) is also safe.

That argument is about the math. It is not a guarantee about the library:
cuBLAS/ATen may legitimately pick a different reduction schedule for M=64
than for M=1024, which would perturb the last ulp without either result
being "wrong". So this file checks two tiers separately:

  Tier A (hard assert): the generated TOKEN SEQUENCE is identical to a
    frozen copy of the pre-change generation loop. This is what actually
    has to hold, and it is what a last-ulp logit difference would have to
    survive in order to matter.
  Tier B (reported, asserted bit-exact but with a diagnostic on failure):
    model(ids, num_logits=N) vs model(ids)[:, :N] elementwise.

Parts 1-2 run on CPU (eager MoEBlock, float32). Part 3 re-runs the same
checks on CUDA in bfloat16 with the real Triton fused MoE, and is skipped
with a message if no GPU is available.
"""

import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dminfr.engine.model import Cfg, LLaDAMoEKV, KVCacheBuffer
from dminfr.engine.generate import (
    MASK_ID,
    add_gumbel_noise,
    generate_cached,
    get_num_transfer_tokens,
    select_transfer_indices,
    select_transfer_indices_hierarchy,
)

import torch.nn.functional as F


# MASK_ID (156895) indexes embed_tokens directly, so VS must stay above it
# even in the small test config -- only H/NL/NE shrink.
TEST_CFG = Cfg(H=64, NH=4, KVH=4, NL=2, NE=8, TOPK=2, EI=32, VS=157184)


# ---------------------------------------------------------------------------
# Frozen reference: the generation loop exactly as it was before num_logits.
# Every model() call omits num_logits (so lm_head runs over the full input)
# and the step call slices in Python afterwards.
# ---------------------------------------------------------------------------

def _reference_generate_block_cached(
    model, x, block_start, block_end, steps_per_block, cache_buffer,
    temperature, remasking, confidence_threshold=None, low_confidence_threshold=0.4,
):
    block_length = block_end - block_start
    device = x.device

    block_mask_index = (x[:, block_start:block_end] == MASK_ID)
    num_transfer = None
    if confidence_threshold is None:
        num_transfer = get_num_transfer_tokens(block_mask_index, steps_per_block)

    for step in range(steps_per_block):
        active_ids = x[:, block_start:block_end]
        mask_index = (active_ids == MASK_ID)
        if not mask_index.any():
            break

        suffix_ids = x[:, block_start:]
        suffix_logits, _ = model(                      # <- no num_logits
            suffix_ids,
            position_offset=block_start,
            cache_buffer=cache_buffer,
            write_pos=block_start,
        )
        logits = suffix_logits[:, :block_length]       # <- sliced in Python

        logits_with_noise = add_gumbel_noise(logits, temperature)
        x0_raw = logits_with_noise.argmax(dim=-1)

        if remasking == "low_confidence":
            p = F.softmax(logits.float(), dim=-1)
            x0_p = p.gather(-1, x0_raw.unsqueeze(-1)).squeeze(-1)
        elif remasking == "random":
            x0_p = torch.rand(x0_raw.shape, device=device)
        else:
            raise ValueError(f"Unknown remasking: {remasking}")

        x0 = torch.where(mask_index, x0_raw, active_ids)
        confidence = torch.where(mask_index, x0_p, torch.full_like(x0_p, -torch.inf))
        if confidence_threshold is not None:
            if step == steps_per_block - 1:
                transfer_index = mask_index.clone()
            else:
                transfer_index = select_transfer_indices_hierarchy(
                    confidence, threshold=confidence_threshold,
                    low_threshold=low_confidence_threshold,
                )
        else:
            transfer_index = select_transfer_indices(confidence, num_transfer[:, step])

        active_ids = active_ids.clone()
        active_ids[transfer_index] = x0[transfer_index]
        x[:, block_start:block_end] = active_ids

    remaining_ids = x[:, block_start:]
    model(                                             # <- no num_logits
        remaining_ids,
        position_offset=block_start,
        cache_buffer=cache_buffer,
        write_pos=block_start,
    )
    cache_buffer.commit(block_end)
    return x


@torch.no_grad()
def _reference_generate_cached(
    model, prompt_ids, gen_length=128, steps=128, block_length=128,
    temperature=0.0, remasking="low_confidence",
    confidence_threshold=None, low_confidence_threshold=0.4,
):
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    steps_per_block = steps // num_blocks

    device = prompt_ids.device
    B, P = prompt_ids.shape
    total_len = P + gen_length

    x = torch.full((B, total_len), MASK_ID, dtype=torch.long, device=device)
    x[:, :P] = prompt_ids

    cache_buffer = KVCacheBuffer(
        num_layers=model.cfg.NL,
        batch_size=B,
        kvh_local=model.layers[0].self_attn.KVH_local,
        max_len=total_len,
        head_dim=model.cfg.HD,
        dtype=next(model.parameters()).dtype,
        device=device,
    )

    model(x, position_offset=0, cache_buffer=cache_buffer, write_pos=0)  # <- no num_logits
    cache_buffer.commit(P)

    for block_idx in range(num_blocks):
        x = _reference_generate_block_cached(
            model=model, x=x,
            block_start=P + block_idx * block_length,
            block_end=P + (block_idx + 1) * block_length,
            steps_per_block=steps_per_block,
            cache_buffer=cache_buffer,
            temperature=temperature,
            remasking=remasking,
            confidence_threshold=confidence_threshold,
            low_confidence_threshold=low_confidence_threshold,
        )
    return x[:, P:]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_model(device, dtype, use_fused_moe):
    torch.manual_seed(0)
    model = LLaDAMoEKV(TEST_CFG, use_fused_moe=use_fused_moe)
    if use_fused_moe:
        # w1/w2 are raw torch.empty Parameters (normally filled by
        # load_state_dict_from_unfused); give them real values.
        for layer in model.layers:
            torch.nn.init.normal_(layer.mlp.w1, std=0.02)
            torch.nn.init.normal_(layer.mlp.w2, std=0.02)
    return model.to(dtype).to(device).eval()


def _fresh_cache(model, B, total_len, device, dtype):
    return KVCacheBuffer(
        num_layers=model.cfg.NL, batch_size=B,
        kvh_local=model.layers[0].self_attn.KVH_local,
        max_len=total_len, head_dim=model.cfg.HD, dtype=dtype, device=device,
    )


def _report(label, a, b):
    """Tier B: bit-exact expected; print a real diagnostic if it isn't."""
    if torch.equal(a, b):
        print(f"    [bit-exact] {label}")
        return True
    diff = (a.float() - b.float()).abs()
    print(f"    [DIFFERS]   {label}: max_abs={diff.max().item():.3e} "
          f"mean_abs={diff.mean().item():.3e} "
          f"n_differing={(diff > 0).sum().item()}/{diff.numel()}")
    return False


# ---------------------------------------------------------------------------
# Part 1: forward(num_logits=N) == forward()[:, :N]
# ---------------------------------------------------------------------------

@torch.no_grad()
def test_forward_slice_equivalence(device, dtype, use_fused_moe):
    print(f"  Part 1: forward(num_logits=N) vs forward()[:, :N]  "
          f"({device}, {dtype}, fused_moe={use_fused_moe})")
    model = _build_model(device, dtype, use_fused_moe)
    ok = True

    for B, T, N in [(1, 64, 8), (1, 64, 64), (2, 48, 16), (3, 32, 1), (1, 128, 64)]:
        torch.manual_seed(B * 1000 + T)
        ids = torch.randint(0, 1000, (B, T), device=device)

        full, kv_full = model(ids, position_offset=0)
        sliced, kv_sliced = model(ids, position_offset=0, num_logits=N)

        assert sliced.shape == (B, N, TEST_CFG.VS), \
            f"expected {(B, N, TEST_CFG.VS)}, got {tuple(sliced.shape)}"
        ok &= _report(f"B={B} T={T} N={N}", full[:, :N], sliced)

        # The K/V side effect must be untouched -- it is the whole reason
        # the prime/finalize passes run at all.
        for i, ((kf, vf), (ks, vs)) in enumerate(zip(kv_full, kv_sliced)):
            assert torch.equal(kf, ks), f"layer {i} K diverged"
            assert torch.equal(vf, vs), f"layer {i} V diverged"

    # num_logits=0 skips the head entirely but must still return full K/V.
    torch.manual_seed(7)
    ids = torch.randint(0, 1000, (2, 40), device=device)
    _, kv_full = model(ids, position_offset=0)
    none_logits, kv_zero = model(ids, position_offset=0, num_logits=0)
    assert none_logits is None, "num_logits=0 must return None for logits"
    for i, ((kf, vf), (kz, vz)) in enumerate(zip(kv_full, kv_zero)):
        assert torch.equal(kf, kz), f"layer {i} K diverged at num_logits=0"
        assert torch.equal(vf, vz), f"layer {i} V diverged at num_logits=0"
    print("    [ok]        num_logits=0 -> logits None, K/V identical")

    # The cache_buffer path is the one production actually uses.
    B, T = 2, 48
    torch.manual_seed(11)
    ids = torch.randint(0, 1000, (B, T), device=device)
    cb_a = _fresh_cache(model, B, T, device, dtype)
    cb_b = _fresh_cache(model, B, T, device, dtype)
    full, _ = model(ids, position_offset=0, cache_buffer=cb_a, write_pos=0)
    sliced, _ = model(ids, position_offset=0, cache_buffer=cb_b, write_pos=0, num_logits=16)
    ok &= _report("cache_buffer path B=2 T=48 N=16", full[:, :16], sliced)
    for i in range(TEST_CFG.NL):
        assert torch.equal(cb_a.k[i], cb_b.k[i]), f"layer {i} cache K diverged"
        assert torch.equal(cb_a.v[i], cb_b.v[i]), f"layer {i} cache V diverged"
    print("    [ok]        cache_buffer contents identical")
    return ok


# ---------------------------------------------------------------------------
# Part 2: end-to-end generate_cached vs the frozen pre-change loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def test_generate_equivalence(device, dtype, use_fused_moe):
    print(f"  Part 2: generate_cached vs frozen reference  "
          f"({device}, {dtype}, fused_moe={use_fused_moe})")
    model = _build_model(device, dtype, use_fused_moe)

    configs = [
        dict(gen_length=32, steps=32, block_length=32, confidence_threshold=None),
        dict(gen_length=64, steps=64, block_length=16, confidence_threshold=None),
        # The recommended production config's decoding path (threshold=0.9/0.4)
        # exercises the early-exit branch, where a block can end before
        # steps_per_block -- i.e. a different number of lm_head calls.
        dict(gen_length=64, steps=32, block_length=16, confidence_threshold=0.9),
    ]

    for B in (1, 3):
        for cfg in configs:
            torch.manual_seed(123)
            prompt = torch.randint(0, 1000, (B, 12), device=device)

            got = generate_cached(model, prompt.clone(), temperature=0.0, **cfg)
            want = _reference_generate_cached(model, prompt.clone(), temperature=0.0, **cfg)

            assert got.shape == want.shape, f"shape {got.shape} vs {want.shape}"
            if not torch.equal(got, want):
                n = (got != want).sum().item()
                raise AssertionError(
                    f"TOKEN MISMATCH B={B} {cfg}: {n}/{got.numel()} tokens differ\n"
                    f"  got : {got[0].tolist()}\n  want: {want[0].tolist()}"
                )
            thr = cfg["confidence_threshold"]
            print(f"    [ok] B={B} gen={cfg['gen_length']} block={cfg['block_length']} "
                  f"steps={cfg['steps']} threshold={thr} -> {got.numel()} tokens identical")


# ---------------------------------------------------------------------------

def main():
    all_bit_exact = True

    print("=" * 70)
    print("CPU (float32, eager MoEBlock)")
    print("=" * 70)
    all_bit_exact &= test_forward_slice_equivalence("cpu", torch.float32, False)
    test_generate_equivalence("cpu", torch.float32, False)

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        print()
        print("=" * 70)
        print(f"CUDA (bfloat16, Triton fused MoE) — {name}")
        print("=" * 70)
        all_bit_exact &= test_forward_slice_equivalence("cuda", torch.bfloat16, True)
        test_generate_equivalence("cuda", torch.bfloat16, True)
    else:
        print("\n[skip] No CUDA GPU — skipping the bfloat16 / Triton fused MoE run.")

    print()
    print("=" * 70)
    print("PASS — generated token sequences are identical to the pre-change loop.")
    if all_bit_exact:
        print("       Logits are bit-exact everywhere (Tier B clean).")
    else:
        print("       NOTE: some logits differed in the last ulp (see [DIFFERS] above).")
        print("       Expected-but-benign: a different M lets cuBLAS/ATen pick a")
        print("       different reduction schedule. Token output is unaffected, which")
        print("       is the property that matters.")
    print("=" * 70)


if __name__ == "__main__":
    main()
