"""Correctness and speed for the three optional fusions.

    (QKV fusion was measured here and REJECTED -- see Attention._project_qkv)
    LLADA_FUSE_RMSNORM=1  one Triton kernel instead of ~8 aten launches
    LLADA_FUSE_DECODE=1   argmax + confidence in one pass over the logits

Each is measured separately, because they have different risk profiles and
should be adopted or dropped independently.

Only QKV can plausibly be bit-exact -- every output row is the same dot product
over the same K, so the arithmetic is unchanged and only cuBLAS kernel
selection differs. The other two reassociate a reduction and cannot be, so for
them the question is not "is it identical" but "does the difference reach the
output" -- which for a diffusion decoder means: does the set of tokens selected
per step change. A logit that moves in the last ulp is irrelevant; a token that
flips is not.

That is why the decode-tail test compares SELECTED TOKENS rather than
probabilities, and the RMSNorm test compares end-to-end generation rather than
per-tensor error. Both are the property that actually matters.

CUDA only. Skips with a message otherwise.
"""

from __future__ import annotations

import os
import sys
import time

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _bench(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


# --------------------------------------------------------------------------
# 1. RMSNorm
# --------------------------------------------------------------------------

def test_rmsnorm():
    from dminfr.engine.fused_ops import fused_rmsnorm
    from dminfr.engine.model import RMSNorm

    print("\n  --- RMSNorm ---")
    print(f"  {'shape':>18}  {'rel_L2':>10}  {'max_abs':>10}  "
          f"{'eager':>9}  {'fused':>9}  {'speedup':>8}")
    ok = True
    # (rows, N): the two regimes the model actually runs -- the layer norms over
    # H=2048, and q_norm/k_norm over HD=128 with far more rows.
    for rows, N in [(2048, 2048), (16384, 2048), (32768, 128), (262144, 128)]:
        torch.manual_seed(rows)
        x = torch.randn(rows, N, device="cuda", dtype=torch.bfloat16) * 0.5
        norm = RMSNorm(N, 1e-5).to("cuda").to(torch.bfloat16)
        torch.nn.init.normal_(norm.weight, mean=1.0, std=0.02)

        # eager reference, bypassing the fused branch explicitly
        xf = x.float()
        want = (norm.weight.float() * (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-5))
                ).to(torch.bfloat16)
        got = fused_rmsnorm(x, norm.weight, 1e-5)

        diff = (got.float() - want.float())
        rel = diff.norm() / want.float().norm()
        t_e = _bench(lambda: (norm.weight.float() * (x.float() * torch.rsqrt(
            x.float().pow(2).mean(-1, keepdim=True) + 1e-5))).to(torch.bfloat16))
        t_f = _bench(lambda: fused_rmsnorm(x, norm.weight, 1e-5))
        print(f"  {f'{rows}x{N}':>18}  {rel:10.2e}  {diff.abs().max():10.2e}  "
              f"{t_e:8.4f}ms  {t_f:8.4f}ms  {t_e / t_f:7.2f}x")
        if rel > 1e-2:
            ok = False
            print(f"      ^ relative error beyond bf16 rounding")
    return ok


# --------------------------------------------------------------------------
# 2. Decode tail
# --------------------------------------------------------------------------

def test_decode_tail():
    from dminfr.engine.fused_ops import fused_decode_tail

    print("\n  --- Decode tail (argmax + confidence) ---")
    print(f"  {'shape':>18}  {'argmax match':>13}  {'conf rel':>10}  "
          f"{'eager':>9}  {'fused':>9}  {'speedup':>8}")
    ok = True
    V = 157184
    for B, T in [(1, 32), (1, 64), (8, 32), (32, 32)]:
        torch.manual_seed(B * 100 + T)
        logits = torch.randn(B, T, V, device="cuda", dtype=torch.bfloat16)

        def eager():
            idx = logits.argmax(dim=-1)
            p = torch.softmax(logits.float(), dim=-1)
            return idx, p.gather(-1, idx.unsqueeze(-1)).squeeze(-1)

        w_idx, w_conf = eager()
        g_idx, g_conf = fused_decode_tail(logits)

        # The argmax is pure comparison, so it must match exactly. Anything
        # else means an indexing bug, not a rounding difference.
        match = (g_idx == w_idx).float().mean().item()
        rel = ((g_conf - w_conf).abs() / w_conf.clamp(min=1e-30)).max().item()
        t_e = _bench(lambda: eager(), iters=20, warmup=5)
        t_f = _bench(lambda: fused_decode_tail(logits), iters=20, warmup=5)
        print(f"  {f'{B}x{T}x{V}':>18}  {match * 100:12.2f}%  {rel:10.2e}  "
              f"{t_e:8.4f}ms  {t_f:8.4f}ms  {t_e / t_f:7.2f}x")
        if match < 1.0:
            ok = False
            print("      ^ argmax differs -- that is a bug, not rounding")
        if rel > 1e-2:
            ok = False
            print("      ^ confidence beyond fp32-rounding tolerance")
    return ok


# --------------------------------------------------------------------------
# 3. QKV
# --------------------------------------------------------------------------

def test_qkv():
    from dminfr.engine.model import FULL_CFG, Attention

    print("\n  --- Fused QKV ---")
    print(f"  {'tokens':>8}  {'bit-exact':>10}  {'rel_L2':>10}  "
          f"{'3 GEMMs':>9}  {'1 GEMM':>9}  {'speedup':>8}")
    ok = True
    cfg = FULL_CFG
    torch.manual_seed(0)
    attn = Attention(cfg).to("cuda").to(torch.bfloat16).eval()
    for p in attn.parameters():
        torch.nn.init.normal_(p, std=0.02)

    # fuse_qkv_ was removed after this measurement (see Attention._project_qkv);
    # rebuild the fused projection locally so the comparison stays runnable on
    # other hardware, where cuBLAS may choose differently.
    import copy
    import torch.nn as nn
    fused = copy.deepcopy(attn)
    with torch.no_grad():
        w = torch.cat([fused.q_proj.weight, fused.k_proj.weight,
                       fused.v_proj.weight], dim=0)
        lin = nn.Linear(w.shape[1], w.shape[0], bias=False).to(w.device, w.dtype)
        lin.weight.copy_(w)
    hd = cfg.HD
    sizes = [fused.NH_local * hd, fused.KVH_local * hd, fused.KVH_local * hd]
    fused._project_qkv = lambda x: lin(x).split(sizes, dim=-1)

    for M in (32, 256, 1024, 2048):
        x = torch.randn(1, M, cfg.H, device="cuda", dtype=torch.bfloat16) * 0.05
        with torch.no_grad():
            w = torch.cat(attn._project_qkv(x), dim=-1)
            g = torch.cat(fused._project_qkv(x), dim=-1)
            exact = torch.equal(w, g)
            rel = (g.float() - w.float()).norm() / w.float().norm().clamp(min=1e-30)
            t_3 = _bench(lambda: attn._project_qkv(x))
            t_1 = _bench(lambda: fused._project_qkv(x))
        print(f"  {M:8d}  {str(exact):>10}  {rel:10.2e}  "
              f"{t_3:8.4f}ms  {t_1:8.4f}ms  {t_3 / t_1:7.2f}x")
        if rel > 1e-2:
            ok = False
    return ok


# --------------------------------------------------------------------------
# 4. End-to-end: do the fusions change generated tokens?
# --------------------------------------------------------------------------

def test_end_to_end_tokens():
    """The decisive check. Per-tensor error is not the question -- whether the
    decoder picks different tokens is."""
    import dminfr.engine.fused_ops as fo
    from dminfr.engine.generate import generate_cached
    from dminfr.engine.model import Cfg, LLaDAMoEKV

    print("\n  --- End-to-end token identity ---")
    cfg = Cfg(H=512, NH=4, KVH=4, NL=4, NE=16, TOPK=4, EI=256, VS=157184)
    torch.manual_seed(0)
    model = LLaDAMoEKV(cfg, use_fused_moe=False).to(torch.bfloat16).cuda().eval()

    torch.manual_seed(7)
    prompt = torch.randint(0, 1000, (2, 16), device="cuda")
    kw = dict(gen_length=32, steps=32, block_length=16, temperature=0.0)

    fo.FUSE_RMSNORM, fo.FUSE_DECODE = False, False
    base = generate_cached(model, prompt.clone(), **kw)

    results = {}
    for name, rms, dec in [("rmsnorm", True, False),
                           ("decode", False, True),
                           ("both", True, True)]:
        fo.FUSE_RMSNORM, fo.FUSE_DECODE = rms, dec
        got = generate_cached(model, prompt.clone(), **kw)
        same = int((got == base).sum())
        total = base.numel()
        results[name] = same / total
        flag = "IDENTICAL" if same == total else f"{total - same}/{total} differ"
        print(f"  {name:>10}: {flag}")
    fo.FUSE_RMSNORM, fo.FUSE_DECODE = False, False
    return results


def main():
    if not torch.cuda.is_available():
        print("[skip] fusions need a CUDA GPU")
        return 0
    print("=" * 78)
    print(f"Fusion evaluation - {torch.cuda.get_device_name(0)}")
    print("=" * 78)
    rms_ok = test_rmsnorm()
    dec_ok = test_decode_tail()
    qkv_ok = test_qkv()
    tokens = test_end_to_end_tokens()

    print("\n" + "=" * 78)
    print("Numerics within tolerance:  "
          f"rmsnorm={rms_ok}  decode={dec_ok}  qkv={qkv_ok}")
    print("Token identity vs unfused:  "
          + "  ".join(f"{k}={v * 100:.1f}%" for k, v in tokens.items()))
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
