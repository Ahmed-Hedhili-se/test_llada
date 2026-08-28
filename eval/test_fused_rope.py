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


def make(B, NH, KVH, T, HD, per_row, device, dtype, transposed=False):
    """transposed=True reproduces what Attention.forward actually passes: a
    transpose(1, 2) VIEW of [B, T, NH, HD], which is not contiguous. The
    fused path must handle that without silently copying, so it is tested
    explicitly rather than only on the easy contiguous case."""
    torch.manual_seed(0)
    if transposed:
        q = torch.randn(B, T, NH, HD, device=device, dtype=dtype).transpose(1, 2)
        k = torch.randn(B, T, KVH, HD, device=device, dtype=dtype).transpose(1, 2)
        assert not q.is_contiguous() and q.stride(-1) == 1
    else:
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
        # B, NH, KVH,  T,  HD, per_row, transposed
        (1, 16, 16, 32, 128, False, False),
        (1, 16, 16, 32, 128, True, False),
        (4, 16, 16, 64, 128, False, False),
        (4, 16, 16, 64, 128, True, False),
        (64, 16, 16, 32, 128, False, False),  # production: BATCH_MAX_SIZE=64, block 32
        (64, 16, 16, 32, 128, True, False),
        (8, 16, 4, 128, 128, True, False),    # GQA: KVH != NH
        (2, 8, 8, 17, 64, True, False),       # odd T, smaller HD
        # The layout Attention.forward actually passes: a non-contiguous
        # transpose(1, 2) view. This is the case that matters in production.
        (64, 16, 16, 32, 128, False, True),
        (64, 16, 16, 32, 128, True, True),
        (8, 16, 4, 128, 128, True, True),     # GQA, transposed
    ]

    print(f"\n  Part 1: bit-exactness vs eager\n")
    print(f"    {'shape':<30} {'cos':<10} {'layout':<12} {'q':<11} {'k':<11}")
    all_exact = True
    for B, NH, KVH, T, HD, per_row, tr in cases:
        q, k, cos, sin = make(B, NH, KVH, T, HD, per_row, dev, dt, transposed=tr)
        eq, ek = eager_rope(q, k, cos, sin)
        fq, fk = fused_ops.fused_rope(q, k, cos, sin)
        okq = torch.equal(eq, fq)
        okk = torch.equal(ek, fk)
        all_exact &= okq and okk
        shape = f"B{B} NH{NH} KVH{KVH} T{T} HD{HD}"
        layout = "[B,T,HD]" if per_row else "[T,HD]"
        mem = "transposed" if tr else "contiguous"
        print(f"    {shape:<30} {layout:<10} {mem:<12} "
              f"{'bit-exact' if okq else 'DIFFERS':<11} "
              f"{'bit-exact' if okk else 'DIFFERS':<11}")
        if not okq:
            print(f"        q max abs diff: {(eq.float() - fq.float()).abs().max():.3e}")
        if not okk:
            print(f"        k max abs diff: {(ek.float() - fk.float()).abs().max():.3e}")

    print(f"\n  Part 2: wall clock at the production shape\n")
    print(f"    {'shape':<26} {'eager':>10} {'fused':>10} {'speedup':>9}")
    for B, NH, KVH, T, HD, per_row, tr in [(64, 16, 16, 32, 128, False, True),
                                           (64, 16, 16, 32, 128, True, True),
                                           (32, 16, 16, 64, 128, True, True),
                                           (1, 16, 16, 32, 128, False, True)]:
        q, k, cos, sin = make(B, NH, KVH, T, HD, per_row, dev, dt, transposed=tr)

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
        shape = f"B{B} T{T} {'per-row' if per_row else 'shared'} (transposed)"
        print(f"    {shape:<38} {te:>9.4f}ms {tf:>9.4f}ms {te/tf:>8.2f}x")

    print("\n" + "=" * 70)
    if all_exact:
        print("PASS - fused RoPE is BIT-EXACT vs eager at every shape and layout.")
    else:
        print("FAIL - fused RoPE diverges from eager. Do not ship.")
    print("=" * 70)
    return 0 if all_exact else 1


if __name__ == "__main__":
    raise SystemExit(main())
