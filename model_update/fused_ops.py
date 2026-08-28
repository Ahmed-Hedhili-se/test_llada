"""Two Triton fusions, both opt-in and both individually revertible.

Neither is bit-exact, which is why each sits behind its own switch rather than
replacing the eager path outright:

  LLADA_FUSE_RMSNORM=1   fuse RMSNorm's ~8 aten kernels into one
  LLADA_FUSE_DECODE=1    fuse the decode tail's argmax + confidence gather

RMSNorm
-------
The eager version is ``x.float()``, ``pow(2)``, ``mean(-1)``, ``+eps``,
``rsqrt``, two multiplies and ``.to(dtype)`` -- eight launches for one
normalisation, and the model runs 65 of them per forward pass (16 layers x 4,
plus the final norm). That is roughly 30% of all kernel launches in a forward,
and the profiler attributes ~9% of GPU time to them plus more inside the cast
kernels.

``q_norm``/``k_norm`` are the worst of it: they reduce over HD=128 on a
reshaped ``[B*T*NH, 128]`` tensor, so almost none of their cost is arithmetic.

Not bit-exact: ``tl.sum`` over the feature axis will not reproduce ATen's
reduction order. The arithmetic is the same (fp32 accumulation, same eps
placement, same cast points); only the summation tree differs.

Decode tail
-----------
``generate.py`` computes, per step: ``logits.float()`` (a full fp32 copy of a
[B, block, 157184] tensor), ``softmax`` over the whole vocabulary, then a
``gather`` of exactly one probability per position -- plus a separate
``argmax``. Three passes over a 157k-wide fp32 tensor to produce two numbers
per position.

One kernel does it in a single pass over the bf16 logits, computing the max,
the exp-sum and the argmax together.

The argmax half is exactly identical (comparisons, no arithmetic). The
probability can differ in the last ulp because the exp-sum accumulates in a
different order, and it is consumed only as a ranking/threshold value -- but
"only a ranking" is not a proof, so eval/test_fusions.py checks whether the
selected token set actually changes.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # pragma: no cover - platform dependent
    HAS_TRITON = False

# Both default ON. MEASURED on an A6000, batched throughput at concurrency 32,
# fixed prompts, 12192 output tokens in every arm:
#
#     unfused          52.5s   232.1 tok/s
#     fused            43.9s   277.9 tok/s   (+19.7%)
#     unfused repeat   52.6s   231.6 tok/s   (baselines agree to 0.2%)
#
# Accuracy gate: GSM8K n=50 seed=42, 72.0% unfused vs 74.0% fused -- one
# question, noise at that sample size. Stronger evidence is the token-identity
# check in eval/test_fusions.py, where both fusions reproduce the unfused
# generation exactly on a small config.
#
# Neither is bit-exact (both reassociate a reduction), so both keep a kill
# switch: LLADA_FUSE_RMSNORM=0 / LLADA_FUSE_DECODE=0.
#
# Note the gain is a BATCHED gain. At batch 1 the same A/B measures ~2%,
# because MoE weight streaming dominates there and these two are a small share
# of it; at batch 32 they are a much larger fraction and the launch savings cut
# into idle time as well.
FUSE_RMSNORM = os.environ.get("LLADA_FUSE_RMSNORM", "1") != "0"
FUSE_DECODE = os.environ.get("LLADA_FUSE_DECODE", "1") != "0"

# RoPE
# ----
# Unlike the two above, this one IS bit-exact, so it has no numerical reason
# to be opt-in -- the switch exists only as a kill switch.
#
# `rotate_half` is `torch.cat([-x2, x1], dim=-1)`: it materialises a full
# rotated copy of q and k, every layer, every denoising step, purely to
# express a permutation of the head dimension. ncu attributed ~4% of GPU time
# to the resulting `aten::cat` (3.04%) + `aten::neg` (0.90%) at the production
# shape -- second only to the MoE kernel and the cuBLAS GEMMs. See
# h100x2_bench.md section 10.
#
# The rotation is an index permutation, so a kernel can read the partner
# element directly and never build the copy:
#
#     d <  HD/2 :  out[d] = x[d]*cos[d] + (-x[d + HD/2])*sin[d]
#     d >= HD/2 :  out[d] = x[d]*cos[d] +   x[d - HD/2] *sin[d]
#
# Bit-exactness: cos/sin are cast to the model dtype before apply_rope (see
# model.py), so the eager path is three bf16 ops, each computed in fp32 by
# ATen's opmath and rounded back. The kernel reproduces that order exactly --
# multiply, round, multiply, round, add, round -- rather than taking the more
# accurate fully-fp32 route, for the same reason the SiLU epilogue does.
FUSE_ROPE = os.environ.get("LLADA_FUSE_ROPE", "1") != "0"


if HAS_TRITON:

    @triton.jit
    def _rmsnorm_kernel(x_ptr, w_ptr, out_ptr, stride, N, eps,
                        BLOCK: tl.constexpr):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(x_ptr + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
        # Mirrors the eager order: mean of squares in fp32, then eps, then
        # rsqrt, then scale, then weight, then cast once at the end.
        var = tl.sum(x * x, axis=0) / N
        x_hat = x * tl.math.rsqrt(var + eps)
        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + row * stride + cols, (w * x_hat).to(out_ptr.dtype.element_ty),
                 mask=mask)

    @triton.jit
    def _rope_kernel(x_ptr, out_ptr, cos_ptr, sin_ptr,
                     s_xb, s_xh, s_xt,
                     s_cb, s_ct,
                     H, T,
                     HD: tl.constexpr, HALF: tl.constexpr, BLOCK: tl.constexpr):
        # One program per (batch, head, token) row; the whole head dim fits in
        # one block (HD=128 on FULL_CFG), so the partner element the rotation
        # needs is already in registers-worth of the same contiguous load.
        pid = tl.program_id(0)
        t = pid % T
        h = (pid // T) % H
        b = pid // (T * H)

        cols = tl.arange(0, BLOCK)
        mask = cols < HD

        x_row = x_ptr + b * s_xb + h * s_xh + t * s_xt
        o_row = out_ptr + b * s_xb + h * s_xh + t * s_xt
        # s_cb is 0 when cos/sin are shared across the batch ([T, HD]), which
        # collapses the batch term without a second kernel or a broadcast copy.
        c_row = cos_ptr + b * s_cb + t * s_ct
        s_row = sin_ptr + b * s_cb + t * s_ct

        x = tl.load(x_row + cols, mask=mask, other=0.0)
        c = tl.load(c_row + cols, mask=mask, other=0.0)
        s = tl.load(s_row + cols, mask=mask, other=0.0)

        # rotate_half without materialising it: partner index, then sign.
        partner = tl.where(cols < HALF, cols + HALF, cols - HALF)
        xr = tl.load(x_row + partner, mask=mask, other=0.0)
        xr = tl.where(cols < HALF, -xr, xr)

        # Reproduce ATen's op order exactly: each bf16 op computes in fp32 and
        # rounds. Doing the whole expression in fp32 and rounding once would be
        # more accurate and would NOT match, which is the point.
        ty = out_ptr.dtype.element_ty
        t1 = (x.to(tl.float32) * c.to(tl.float32)).to(ty)
        t2 = (xr.to(tl.float32) * s.to(tl.float32)).to(ty)
        out = (t1.to(tl.float32) + t2.to(tl.float32)).to(ty)
        tl.store(o_row + cols, out, mask=mask)

    @triton.jit
    def _decode_tail_kernel(logits_ptr, argmax_ptr, conf_ptr,
                            n_rows, V, stride,
                            BLOCK: tl.constexpr):
        """One pass over a row of logits -> (argmax index, softmax prob of it).

        Two passes over the row internally (max, then exp-sum) but one pass
        over HBM, versus the eager path's fp32 copy + softmax write + gather
        read of a tensor 2x wider in bytes.
        """
        row = tl.program_id(0)
        if row >= n_rows:
            return
        base = logits_ptr + row * stride

        best = -float("inf")
        best_idx = 0
        for off in range(0, V, BLOCK):
            cols = off + tl.arange(0, BLOCK)
            m = cols < V
            vals = tl.load(base + cols, mask=m, other=-float("inf")).to(tl.float32)
            chunk_max = tl.max(vals, axis=0)
            if chunk_max > best:
                best = chunk_max
                # argmax within the chunk: lowest index attaining the max, which
                # is the tie-break torch.argmax uses.
                best_idx = off + tl.argmax(vals, axis=0)

        acc = 0.0
        for off in range(0, V, BLOCK):
            cols = off + tl.arange(0, BLOCK)
            m = cols < V
            vals = tl.load(base + cols, mask=m, other=-float("inf")).to(tl.float32)
            acc += tl.sum(tl.where(m, tl.exp(vals - best), 0.0), axis=0)

        tl.store(argmax_ptr + row, best_idx)
        tl.store(conf_ptr + row, 1.0 / acc)  # exp(best-best)=1, so p = 1/sum


def fused_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Drop-in for RMSNorm.forward on a 2-D [rows, N] view."""
    assert x.is_cuda and x.dim() == 2
    N = x.shape[-1]
    out = torch.empty_like(x)
    BLOCK = triton.next_power_of_2(N)
    num_warps = 4 if BLOCK <= 512 else (8 if BLOCK <= 2048 else 16)
    _rmsnorm_kernel[(x.shape[0],)](
        x, weight, out, x.stride(0), N, eps, BLOCK=BLOCK, num_warps=num_warps
    )
    return out


def _rope_one(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to one [B, H, T, HD] tensor. cos/sin are [T, HD] or [B, T, HD]."""
    B, H, T, HD = x.shape
    assert HD % 2 == 0, f"RoPE needs an even head dim, got {HD}"
    # The kernel indexes the head dim with a bare `cols`, so the last axis must
    # be contiguous; the transpose(1, 2) upstream leaves it that way, but a
    # future caller might not.
    x = x.contiguous()
    cos = cos.contiguous()
    sin = sin.contiguous()
    out = torch.empty_like(x)

    if cos.dim() == 2:                       # [T, HD] shared across the batch
        s_cb, s_ct = 0, cos.stride(0)
    else:                                    # [B, T, HD] per-row, left-padded
        s_cb, s_ct = cos.stride(0), cos.stride(1)

    BLOCK = triton.next_power_of_2(HD)
    _rope_kernel[(B * H * T,)](
        x, out, cos, sin,
        x.stride(0), x.stride(1), x.stride(2),
        s_cb, s_ct,
        H, T,
        HD=HD, HALF=HD // 2, BLOCK=BLOCK,
        num_warps=4,
    )
    return out


def fused_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor,
               sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Drop-in for apply_rope. Bit-exact vs the eager path."""
    return _rope_one(q, cos, sin), _rope_one(k, cos, sin)


def fused_decode_tail(logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """(argmax, softmax-prob-of-argmax) for [B, T, V] logits, in one pass.

    Replaces: logits.float() -> softmax -> gather, plus a separate argmax.
    """
    assert logits.is_cuda and logits.dim() == 3
    B, T, V = logits.shape
    flat = logits.reshape(B * T, V)
    if not flat.is_contiguous():
        flat = flat.contiguous()
    idx = torch.empty(B * T, device=logits.device, dtype=torch.int64)
    conf = torch.empty(B * T, device=logits.device, dtype=torch.float32)
    BLOCK = 4096
    _decode_tail_kernel[(B * T,)](
        flat, idx, conf, B * T, V, flat.stride(0), BLOCK=BLOCK, num_warps=8
    )
    return idx.view(B, T), conf.view(B, T)
