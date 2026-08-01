import json
import math
import itertools
import torch
import triton
import triton.language as tl

# ── Import the kernel and helpers directly from our model_update package ──────
from model_update.fused_moe_triton import (
    fused_moe,
    invoke_fused_moe_kernel,
    moe_align_block_size,
    fused_moe_kernel,
)

# ═══════════════════════════════════════════════════════════════════════════════
# 1.  REALISTIC WORKLOAD PARAMETERS
#     These match your SMALL_CFG / actual inference loop in eval/time_fraction.py
# ═══════════════════════════════════════════════════════════════════════════════
MODEL_SHAPES = {
    # (E, N, K, top_k, description)
    # E = num_experts, N = w1 output (2*expert_inner), K = hidden_dim
    "SMALL_CFG": (16,  512, 512,  4, "test model (NE=16, EI=256, H=512)"),
    "FULL_CFG":  (64, 2048, 2048, 8, "production model (NE=64, EI=1024, H=2048)"),
}

# Representative M values encountered during actual block-wise KV generation.
# M = batch_size * active_block_length.  For your setup (BS=2, BL=64) M≈128.
# We also include smaller values for prefill and speculative small batches.
REALISTIC_M_BUCKETS = [1, 2, 4, 8, 16, 32, 64, 128, 256]

# How heavily to penalise padding waste (tuned empirically; 0 = pure latency)
PADDING_PENALTY_WEIGHT = 0.5

# ═══════════════════════════════════════════════════════════════════════════════
# 2.  CONFIGURATION SEARCH SPACE
#     Restricted to configs that cannot exceed ~96 KB shared memory on an A40.
#     We add the constraint BLOCK_SIZE_M <= 64 to prevent padding explosion.
# ═══════════════════════════════════════════════════════════════════════════════
def get_config_grid(max_block_m: int = 64):
    """
    Returns a list of candidate Triton configs.

    max_block_m: Hard upper cap on BLOCK_SIZE_M.
                 For small M (< 64), a BLOCK_SIZE_M of 128 wastes 8x the
                 compute because every expert slot is padded to 128.  We
                 cap at 64 by default which already allows BLOCK_SIZE_M > M
                 for the M=32 bucket (2x overhead at most).
    """
    block_m   = [bm for bm in [16, 32, 64, 128] if bm <= max_block_m]
    block_n   = [32, 64, 128]
    block_k   = [32, 64, 128]
    group_m   = [1, 4, 8]
    num_warps = [4, 8]
    num_stages= [2, 3]          # 4 stages often hits shmem limit on A40

    configs = []
    for bm, bn, bk, gm, nw, ns in itertools.product(
            block_m, block_n, block_k, group_m, num_warps, num_stages):
        # ── Hard shmem guard ───────────────────────────────────────────────
        # Formula: tiles_a + tiles_b per stage, each element is 2 bytes (bf16)
        shmem = (bm * bk + bk * bn) * ns * 2
        if shmem > 96_000:
            continue
        configs.append({
            "BLOCK_SIZE_M": bm,
            "BLOCK_SIZE_N": bn,
            "BLOCK_SIZE_K": bk,
            "GROUP_SIZE_M": gm,
            "num_warps":    nw,
            "num_stages":   ns,
        })
    return configs


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  COMPOSITE BENCHMARK: full pipeline + padding penalty
# ═══════════════════════════════════════════════════════════════════════════════
def benchmark_full_pipeline(M, E, N, K, top_k, config, penalty_weight=0.5):
    """
    Measures the END-TO-END cost of one fused_moe call for the given config:
      GEMM1 → SiLU+mul → GEMM2 → weighted_sum

    Returns (score, latency_ms, padding_ratio) where:
      score = latency_ms * (1 + penalty_weight * padding_ratio)

    The padding_ratio captures how much extra work BLOCK_SIZE_M forces:
      If M=16, E=16, top_k=4, and BLOCK_SIZE_M=128:
        real_EM   = M * top_k = 64
        padded_EM = E * ceil(4/BLOCK_SIZE_M) * BLOCK_SIZE_M = 16*128 = 2048
        padding_ratio = (2048 - 64) / 64 = 31   ← catastrophic

    A config with BM=16:
        padded_EM = E * ceil(4/16) * 16 = 16*16 = 256
        padding_ratio = (256 - 64) / 64 = 3      ← moderate
    """
    device = torch.device("cuda")
    dtype  = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    hidden_states  = torch.randn((M, K),       device=device, dtype=dtype)
    w1             = torch.randn((E, N, K),     device=device, dtype=dtype)
    w2             = torch.randn((E, K, N // 2),device=device, dtype=dtype)
    topk_weights   = torch.rand( (M, top_k),   device=device, dtype=torch.float32)
    topk_ids       = torch.randint(0, E, (M, top_k), device=device, dtype=torch.int32)

    bm = config["BLOCK_SIZE_M"]

    # ── Compute padding ratio ─────────────────────────────────────────────────
    real_EM   = M * top_k
    # Each expert is padded to the next multiple of bm
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, bm, E
    )
    padded_EM = sorted_token_ids.shape[0]
    padding_ratio = (padded_EM - real_EM) / max(real_EM, 1)

    # Pre-allocate caches (same as production code)
    cache1 = torch.empty((M, top_k, N),       device=device, dtype=dtype)
    cache2 = torch.empty((M, top_k, K),        device=device, dtype=dtype)

    def run_full_pipeline():
        # GEMM 1: hidden @ W1
        invoke_fused_moe_kernel(
            hidden_states, w1, cache1,
            topk_weights, topk_ids,
            sorted_token_ids, expert_ids, num_tokens_post_padded,
            False, top_k, config,
        )
        # SiLU activation (same as fused_moe production code)
        gate, up = cache1.chunk(2, dim=-1)
        act_out  = (torch.nn.functional.silu(gate) * up).view(M * top_k, N // 2)
        # GEMM 2: activated @ W2
        invoke_fused_moe_kernel(
            act_out, w2, cache2,
            topk_weights, topk_ids,
            sorted_token_ids, expert_ids, num_tokens_post_padded,
            True, top_k, config,
        )
        # Weighted sum
        cache2.sum(dim=1)

    try:
        latency_ms = triton.testing.do_bench(
            run_full_pipeline, warmup=15, rep=100
        )
    except BaseException:
        return float("inf"), float("inf"), padding_ratio

    score = latency_ms * (1.0 + penalty_weight * padding_ratio)
    return score, latency_ms, padding_ratio


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  PROFILING: occupancy, register usage, shared memory
# ═══════════════════════════════════════════════════════════════════════════════
def profile_config(config, K, N):
    """
    Queries the Triton compiler metadata for the fused_moe_kernel compiled
    with the given constexpr values.  Returns a dict with:
      - shared_bytes   : bytes of shared memory per block
      - num_warps      : warps launched per block
      - occupancy_pct  : estimated occupancy (based on SM register limits)
    """
    try:
        compiled = fused_moe_kernel.warmup(
            # We pass dummy pointers; warmup only compiles and inspects metadata
            torch.empty(1, dtype=torch.bfloat16, device="cuda"),   # a_ptr
            torch.empty(1, dtype=torch.bfloat16, device="cuda"),   # b_ptr
            torch.empty(1, dtype=torch.bfloat16, device="cuda"),   # c_ptr
            None, None,    # scale ptrs
            torch.empty(1, dtype=torch.float32, device="cuda"),    # topk_weights
            torch.empty(1, dtype=torch.int32,   device="cuda"),    # sorted_token_ids
            torch.empty(1, dtype=torch.int32,   device="cuda"),    # expert_ids
            torch.empty(1, dtype=torch.int32,   device="cuda"),    # num_tokens_post_padded
            N, K, 1, 1,    # N, K, EM, num_valid_tokens
            1, 1,          # stride_am, stride_ak
            1, 1, 1,       # stride_be, stride_bk, stride_bn
            1, 1,          # stride_cm, stride_cn
            0, 0,          # stride_bse, stride_bsn
            BLOCK_SIZE_M   = config["BLOCK_SIZE_M"],
            BLOCK_SIZE_N   = config["BLOCK_SIZE_N"],
            BLOCK_SIZE_K   = config["BLOCK_SIZE_K"],
            GROUP_SIZE_M   = config["GROUP_SIZE_M"],
            MUL_ROUTED_WEIGHT = False,
            top_k          = 1,
            compute_type   = tl.bfloat16,
            use_fp8_w8a8   = False,
            use_int8_w8a16 = False,
            is_first_gemm  = True,
            num_warps      = config["num_warps"],
            num_stages     = config["num_stages"],
            grid           = (1,),
        )
        meta = compiled.metadata
        shared_bytes = getattr(meta, "shared", None)
        # Max warps per SM on Ampere/A40 = 48; max blocks = shared_mem limited
        occupancy_pct = None
        if shared_bytes is not None and shared_bytes > 0:
            max_blocks_by_shmem = 102400 // shared_bytes
            max_blocks_by_regs  = 65536  // max(meta.num_warps * 32, 1)
            blocks_per_sm       = min(max_blocks_by_shmem, max_blocks_by_regs, 16)
            occupancy_pct       = round(blocks_per_sm * meta.num_warps / 48 * 100)
        return {
            "shared_bytes":   shared_bytes,
            "num_warps":      getattr(meta, "num_warps", config["num_warps"]),
            "occupancy_pct":  occupancy_pct,
        }
    except Exception as e:
        return {"shared_bytes": None, "num_warps": config["num_warps"], "occupancy_pct": None, "error": str(e)}


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  CORRECTNESS CHECK
# ═══════════════════════════════════════════════════════════════════════════════
def verify_correctness(E, N, K, top_k, config_under_test, reference_config):
    """
    Runs fused_moe with both configs on the same inputs and asserts that the
    outputs are numerically close (cosine similarity > 0.999).
    This catches any config that inadvertently breaks the accumulation logic.
    """
    device = torch.device("cuda")
    dtype  = torch.bfloat16
    M      = 32

    torch.manual_seed(0)
    hs   = torch.randn((M, K),        device=device, dtype=dtype)
    w1   = torch.randn((E, N, K),     device=device, dtype=dtype)
    w2   = torch.randn((E, K, N // 2),device=device, dtype=dtype)
    tws  = torch.rand( (M, top_k),    device=device, dtype=torch.float32)
    tids = torch.randint(0, E, (M, top_k), device=device, dtype=torch.int32)

    # Temporarily monkey-patch get_best_config so fused_moe uses our config
    import model_update.fused_moe_triton as _ft

    def _run_with(cfg):
        orig = _ft.get_best_config
        _ft.get_best_config = lambda m, e: cfg
        try:
            out = _ft.fused_moe(hs, w1, w2, tws, tids)
        finally:
            _ft.get_best_config = orig
        return out

    with torch.no_grad():
        ref_out  = _run_with(reference_config).float()
        test_out = _run_with(config_under_test).float()

    cos = torch.nn.functional.cosine_similarity(
        ref_out.view(-1), test_out.view(-1), dim=0
    ).item()
    return cos > 0.999, cos


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  MAIN TUNING LOOP
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",        default="SMALL_CFG", choices=list(MODEL_SHAPES))
    ap.add_argument("--max-block-m",  type=int, default=64,
                    help="Hard cap on BLOCK_SIZE_M to limit padding overhead.")
    ap.add_argument("--penalty",      type=float, default=PADDING_PENALTY_WEIGHT,
                    help="Weight on padding ratio in composite score.")
    ap.add_argument("--top-configs",  type=int, default=3,
                    help="Print profiling for the top-N configs per M.")
    ap.add_argument("--output",       default="moe_tune_config.json")
    ap.add_argument("--verify",       action="store_true", default=True)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is required."); return

    E, N, K, top_k, desc = MODEL_SHAPES[args.model]
    print(f"\n{'='*70}")
    print(f"  End-to-End Aware MoE Autotuner")
    print(f"  Model : {args.model} — {desc}")
    print(f"  GPU   : {torch.cuda.get_device_name(0)}")
    print(f"  max_block_m={args.max_block_m}  penalty={args.penalty}")
    print(f"{'='*70}\n")

    reference_cfg = {
        "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 1,  "num_warps": 4,     "num_stages": 2,
    }

    all_configs = get_config_grid(max_block_m=args.max_block_m)
    print(f"Candidate configs after shmem pruning: {len(all_configs)}\n")

    best_configs = {}

    for M in REALISTIC_M_BUCKETS:
        print(f"{'─'*60}")
        print(f"  Tuning M = {M}  (E={E}, top_k={top_k}, N={N}, K={K})")
        print(f"{'─'*60}")

        results = []

        for i, cfg in enumerate(all_configs):
            score, lat_ms, pad_ratio = benchmark_full_pipeline(
                M, E, N, K, top_k, cfg, args.penalty
            )
            results.append((score, lat_ms, pad_ratio, cfg))

            if (i + 1) % 50 == 0:
                best_so_far = min(results, key=lambda x: x[0])
                print(f"  [{i+1:3d}/{len(all_configs)}] best score={best_so_far[0]:.3f} ms "
                      f"lat={best_so_far[1]:.3f} ms  pad={best_so_far[2]:.1%}")

        results.sort(key=lambda x: x[0])
        best_score, best_lat, best_pad, best_cfg = results[0]

        print(f"\n  Top {args.top_configs} configs for M={M}:")
        for rank, (score, lat, pad, cfg) in enumerate(results[:args.top_configs]):
            prof = profile_config(cfg, K, N)
            bm = cfg["BLOCK_SIZE_M"]
            print(f"    [{rank+1}] BM={bm:3d} BN={cfg['BLOCK_SIZE_N']:3d} "
                  f"BK={cfg['BLOCK_SIZE_K']:3d} GM={cfg['GROUP_SIZE_M']} "
                  f"nw={cfg['num_warps']} ns={cfg['num_stages']}  "
                  f"lat={lat:.3f}ms  pad={pad:.1%}  score={score:.3f}  "
                  f"shmem={prof.get('shared_bytes','?')}B  "
                  f"occ={prof.get('occupancy_pct','?')}%")

        # ── Correctness check on winner ───────────────────────────────────
        if args.verify:
            ok, cos = verify_correctness(E, N, K, top_k, best_cfg, reference_cfg)
            print(f"\n  Correctness check: cos_sim={cos:.6f}  {'✅ PASS' if ok else '❌ FAIL'}")
            if not ok:
                print("  WARNING: Skipping this config — outputs do not match reference!")
                best_cfg   = reference_cfg
                best_lat   = float("nan")
                best_score = float("nan")
                best_pad   = float("nan")

        best_configs[str(M)] = best_cfg
        print(f"\n  ✅ Best for M={M}: {best_cfg}")
        print(f"     Latency={best_lat:.3f}ms  Padding={best_pad:.1%}  "
              f"Score={best_score:.3f}\n")

    with open(args.output, "w") as f:
        json.dump(best_configs, f, indent=4)

    print(f"{'='*70}")
    print(f"  Tuning complete. Saved to: {args.output}")
    print(f"  To apply: ensure moe_tune_config.json is in your repo root.")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
