"""
Regression + benchmark for the SiLU(gate)*up epilogue fused into GEMM1 of
dminfr/engine/fused_moe_triton.py::fused_moe.

The unfused path materialized GEMM1's full [M, top_k, 2*EI] product, read it
back, ran SiLU(gate)*up as a separate elementwise pass, and wrote an
[M*top_k, EI] result. At B=57 that is ~15MB written + 15MB read + 7.5MB
written per MoE layer per forward, ~1.2GB per forward across 16 layers -- on
a kernel ncu measures at 96.4% L2 throughput (DRAM only 66-68%, so this is an
L2-traffic problem, and removing intermediate traffic is exactly the lever).

The fused path has each program compute two B tiles (gate at offs_bn, up at
offs_bn + N) and apply the activation from registers.

WHY THIS SHOULD BE BIT-EXACT: the K-loop is untouched, so the fp32
accumulators are identical. The epilogue then deliberately reproduces the
unfused op ORDER rather than the more accurate one -- accumulators rounded to
bf16 first (the intermediate_cache1 store), SiLU as x/(1+exp(-x)) in fp32
rounded back to bf16, then the multiply. Computing from the raw fp32
accumulators would be more precise and therefore NOT bit-identical.

WHAT THIS CANNOT PROVE BY CONSTRUCTION: that Triton's tl.exp lowers to the
same instruction as ATen's expf. A last-ulp fp32 difference there would
usually vanish in the round to bf16 -- but "usually" is not "never", and this
project has documented that its near-uniform router turns bf16-level noise
into discrete top-8 expert flips. So this file MEASURES the difference rather
than assuming it away, and reports exactly how many elements differ and by
how much if any do.

Part 1 (correctness): fused vs unfused output of fused_moe() at real
FULL_CFG-derived shapes, across the token counts the tuner buckets for.
Part 2 (end-to-end): generate_cached with the fusion on vs off, asserting
identical token sequences.
Part 3 (benchmark): wall-clock per call at each M, both paths.

CUDA-only (Triton). Skips with a message if no GPU is available.
"""

import os
import sys
import time

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dminfr.engine.fused_moe_triton as fmt
from dminfr.engine.fused_moe_triton import fused_moe
from dminfr.engine.model import Cfg, LLaDAMoEKV
from dminfr.engine.generate import generate_cached


# FULL_CFG's MoE geometry: 64 experts, top-8, H=2048, EI=1024 (so w1's N=2048).
E, TOPK, H, EI = 64, 8, 2048, 1024


def _make_inputs(M, device, seed):
    g = torch.Generator(device=device).manual_seed(seed)
    hidden = torch.randn(M, H, generator=g, device=device, dtype=torch.bfloat16) * 0.05
    w1 = torch.randn(E, 2 * EI, H, generator=g, device=device, dtype=torch.bfloat16) * 0.02
    w2 = torch.randn(E, H, EI, generator=g, device=device, dtype=torch.bfloat16) * 0.02

    # Route the way the real model does: fp32 softmax over all experts, flat
    # top-k, no renormalization. Near-uniform logits on purpose -- that is this
    # checkpoint's actual regime and the one most sensitive to tie-breaking.
    logits = torch.randn(M, E, generator=g, device=device, dtype=torch.float32) * 0.05
    weights = torch.softmax(logits, dim=-1, dtype=torch.float32)
    topk_w, topk_ids = torch.topk(weights, TOPK, dim=-1)
    return hidden, w1, w2, topk_w.to(torch.bfloat16), topk_ids.to(torch.int32)


def test_correctness(device):
    print("  Part 1: fused vs unfused epilogue, real FULL_CFG MoE shapes")
    all_exact = True

    for M in (1, 8, 16, 32, 57, 64, 256, 512, 1024, 2048):
        hidden, w1, w2, topk_w, topk_ids = _make_inputs(M, device, seed=M)

        out_unfused = fused_moe(hidden, w1, w2, topk_w, topk_ids, fuse_silu=False)
        out_fused = fused_moe(hidden, w1, w2, topk_w, topk_ids, fuse_silu=True)

        assert out_fused.shape == out_unfused.shape, \
            f"M={M}: shape {out_fused.shape} vs {out_unfused.shape}"

        if torch.equal(out_fused, out_unfused):
            print(f"    [bit-exact] M={M:5d}  (tokens={M}, token-expert pairs={M * TOPK})")
            continue

        all_exact = False
        diff = (out_fused.float() - out_unfused.float()).abs()
        ref = out_unfused.float().abs()
        n_diff = int((diff > 0).sum())
        # Relative size of the worst disagreement, to distinguish "last ulp of
        # bf16" from "actually different math".
        rel = (diff / ref.clamp(min=1e-30)).max().item()
        print(f"    [DIFFERS]   M={M:5d}  n={n_diff}/{diff.numel()} "
              f"({100.0 * n_diff / diff.numel():.4f}%)  max_abs={diff.max().item():.3e}  "
              f"max_rel={rel:.3e}")
    return all_exact


def test_end_to_end(device):
    print("  Part 2: generate_cached with the fusion on vs off")
    cfg = Cfg(H=256, NH=4, KVH=4, NL=2, NE=16, TOPK=4, EI=128, VS=157184)

    torch.manual_seed(0)
    model = LLaDAMoEKV(cfg, use_fused_moe=True)
    for layer in model.layers:
        torch.nn.init.normal_(layer.mlp.w1, std=0.02)
        torch.nn.init.normal_(layer.mlp.w2, std=0.02)
    model = model.to(torch.bfloat16).to(device).eval()

    for B in (1, 3):
        for gen_length, steps, block_length, thr in [
            (32, 32, 32, None),
            (64, 64, 16, None),
            (64, 32, 16, 0.9),
        ]:
            torch.manual_seed(123)
            prompt = torch.randint(0, 1000, (B, 12), device=device)

            fmt.FUSE_SILU = False
            want = generate_cached(model, prompt.clone(), gen_length=gen_length, steps=steps,
                                   block_length=block_length, temperature=0.0,
                                   confidence_threshold=thr)
            fmt.FUSE_SILU = True
            got = generate_cached(model, prompt.clone(), gen_length=gen_length, steps=steps,
                                  block_length=block_length, temperature=0.0,
                                  confidence_threshold=thr)

            if not torch.equal(got, want):
                n = int((got != want).sum())
                raise AssertionError(
                    f"TOKEN MISMATCH B={B} gen={gen_length} block={block_length} "
                    f"steps={steps} threshold={thr}: {n}/{got.numel()} tokens differ\n"
                    f"  fused  : {got[0].tolist()}\n  unfused: {want[0].tolist()}"
                )
            print(f"    [ok] B={B} gen={gen_length} block={block_length} steps={steps} "
                  f"threshold={thr} -> {got.numel()} tokens identical")


def _bench(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3  # ms


def test_benchmark(device):
    print("  Part 3: per-call wall clock")
    print(f"    {'M':>6}  {'unfused':>10}  {'fused':>10}  {'delta':>9}")
    for M in (1, 32, 57, 256, 1024, 2048):
        hidden, w1, w2, topk_w, topk_ids = _make_inputs(M, device, seed=M)
        t_un = _bench(lambda: fused_moe(hidden, w1, w2, topk_w, topk_ids, fuse_silu=False))
        t_fu = _bench(lambda: fused_moe(hidden, w1, w2, topk_w, topk_ids, fuse_silu=True))
        pct = (t_un - t_fu) / t_un * 100.0
        print(f"    {M:6d}  {t_un:9.4f}ms  {t_fu:9.4f}ms  {pct:+8.1f}%")


def main():
    if not torch.cuda.is_available():
        print("[skip] No CUDA GPU — the Triton fused MoE kernel cannot run.")
        return

    print("=" * 74)
    print(f"Fused SiLU epilogue — {torch.cuda.get_device_name(0)}")
    print("=" * 74)

    exact = test_correctness("cuda")
    print()
    test_end_to_end("cuda")
    print()
    test_benchmark("cuda")

    print()
    print("=" * 74)
    if exact:
        print("PASS — fused epilogue is BIT-EXACT vs the unfused path at every shape,")
        print("       and end-to-end token sequences are identical.")
    else:
        print("PARTIAL — end-to-end token sequences are identical, but some kernel")
        print("          outputs differed (see [DIFFERS] above). Most likely tl.exp vs")
        print("          ATen's expf disagreeing in the last fp32 ulp. Decide from the")
        print("          max_rel column whether that is acceptable; LLADA_MOE_FUSED_SILU=0")
        print("          disables the fusion.")
    print("=" * 74)


if __name__ == "__main__":
    main()
