"""Fused RoPE vs the eager path: bit-exactness and speed.

The fusion removes `rotate_half`'s `torch.cat([-x2, x1])`, which ncu
attributed ~4% of GPU time to at the production shape (aten::cat 3.04% +
aten::neg 0.90%, both 2-per-layer-per-forward). See h100x2_bench.md 10.

Unlike the RMSNorm and decode-tail fusions this one reassociates nothing --
it is an index permutation, so it can be and must be bit-exact. This test
asserts that rather than a tolerance, because a tolerance would hide the
one failure mode that matters: getting the op order wrong and silently
computing something more accurate than the model was tuned against.

Run: python -m eval.test_fused_rope
"""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_update.model import rotate_half  # noqa: E402
from model_update import fused_ops  # noqa: E402


def eager_rope(q, k, cos, sin):
    """The pre-fusion apply_rope body, copied verbatim."""
    if cos.dim() == 2:
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
    else:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


def make(B, NH, KVH, T, HD, per_row, device, dtype):
    torch.manual_seed(0)
    q = torch.randn(B, NH, T, HD, device=device, dtype=dtype)
    k = torch.randn(B, KVH, T, HD, device=device, dtype=dtype)
    if per_row:
        pos = torch.arange(T, device=device).unsqueeze(0).expand(B, T).contiguous()
        inv = 1.0 / (500000.0 ** (torch.arange(0, HD, 2, device=device).float() / HD))
        freqs = pos.float().unsqueeze(-1) * inv
        emb = torch.cat([freqs, freqs], dim=-1)
    else:
        pos = torch.arange(T, device=device).float()
        inv = 1.0 / (500000.0 ** (torch.arange(0, HD, 2, device=device).float() / HD))
        emb = torch.cat([torch.outer(pos, inv)] * 2, dim=-1)
    return q, k, emb.cos().to(dtype), emb.sin().to(dtype)


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA required."); return 1
    if not fused_ops.HAS_TRITON:
        print("Triton required."); return 1

    dev, dt = "cuda", torch.bfloat16
    print("=" * 70)
    print(f"Fused RoPE - {torch.cuda.get_device_name(0)}")
    print("=" * 70)

    # Both cos/sin layouts, and shapes spanning batch-1 decode through the
    # production batch. KVH < NH exercises grouped-query attention, where q
    # and k have different head counts and so different grids.
    cases = [
        # B, NH, KVH,  T,  HD, per_row
        (1, 16, 16, 32, 128, False),
        (1, 16, 16, 32, 128, True),
        (4, 16, 16, 64, 128, False),
        (4, 16, 16, 64, 128, True),
        (64, 16, 16, 32, 128, False),   # production: BATCH_MAX_SIZE=64, block 32
        (64, 16, 16, 32, 128, True),
        (8, 16, 4, 128, 128, True),     # GQA: KVH != NH
        (2, 8, 8, 17, 64, True),        # odd T, smaller HD
    ]

    print(f"\n  Part 1: bit-exactness vs eager\n")
    print(f"    {'shape':<34} {'cos layout':<10} {'q':<12} {'k':<12}")
    all_exact = True
    for B, NH, KVH, T, HD, per_row in cases:
        q, k, cos, sin = make(B, NH, KVH, T, HD, per_row, dev, dt)
        eq, ek = eager_rope(q, k, cos, sin)
        fq, fk = fused_ops.fused_rope(q, k, cos, sin)
        okq = torch.equal(eq, fq)
        okk = torch.equal(ek, fk)
        all_exact &= okq and okk
        shape = f"B{B} NH{NH} KVH{KVH} T{T} HD{HD}"
        layout = "[B,T,HD]" if per_row else "[T,HD]"
        print(f"    {shape:<34} {layout:<10} "
              f"{'bit-exact' if okq else 'DIFFERS':<12} "
              f"{'bit-exact' if okk else 'DIFFERS':<12}")
        if not okq:
            print(f"        q max abs diff: {(eq.float() - fq.float()).abs().max():.3e}")
        if not okk:
            print(f"        k max abs diff: {(ek.float() - fk.float()).abs().max():.3e}")

    print(f"\n  Part 2: wall clock at the production shape\n")
    print(f"    {'shape':<26} {'eager':>10} {'fused':>10} {'speedup':>9}")
    for B, NH, KVH, T, HD, per_row in [(64, 16, 16, 32, 128, False),
                                       (64, 16, 16, 32, 128, True),
                                       (32, 16, 16, 64, 128, True),
                                       (1, 16, 16, 32, 128, False)]:
        q, k, cos, sin = make(B, NH, KVH, T, HD, per_row, dev, dt)

        def timed(fn, n=200):
            for _ in range(20):
                fn(q, k, cos, sin)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(n):
                fn(q, k, cos, sin)
            torch.cuda.synchronize()
            return (time.perf_counter() - t0) / n * 1000

        te = timed(eager_rope)
        tf = timed(fused_ops.fused_rope)
        shape = f"B{B} T{T} {'per-row' if per_row else 'shared'}"
        print(f"    {shape:<26} {te:>9.4f}ms {tf:>9.4f}ms {te/tf:>8.2f}x")

    print("\n" + "=" * 70)
    if all_exact:
        print("PASS - fused RoPE is BIT-EXACT vs eager at every shape and layout.")
    else:
        print("FAIL - fused RoPE diverges from eager. Do not ship.")
    print("=" * 70)
    return 0 if all_exact else 1


if __name__ == "__main__":
    raise SystemExit(main())
