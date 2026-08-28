import json
import math
import itertools
import torch
import triton
import triton.language as tl

from model_update.fused_moe_triton import (
    fused_moe,
    invoke_fused_moe_kernel,
    moe_align_block_size,
    fused_moe_kernel,
    SHARED_MEM_LIMIT,
    FUSE_SILU,
    _shmem_bytes,
)

# ═══════════════════════════════════════════════════════════════════════════════
# 1.  REALISTIC WORKLOAD PARAMETERS
#     These match your SMALL_CFG / actual inference loop in eval/time_fraction.py
# ═══════════════════════════════════════════════════════════════════════════════
MODEL_SHAPES = {
    # (E, N, K, top_k, description)
    # E = num_experts, N = w1 output (2*expert_inner), K = hidden_dim
    "SMALL_CFG": (16,  512,  512,  4, "test model (NE=16, EI=256, H=512)"),
    "FULL_CFG":  (64, 2048, 2048, 8, "production model (NE=64, EI=1024, H=2048)"),
}

# Representative M values encountered during actual block-wise KV generation.
# M = batch_size * active_block_length.  For your setup (BS=1, BL=32) M≈32.
# RTX A6000 has 48 GB VRAM — we include larger M values to exploit this.
# Server-side batching (src/server.py, BATCH_MAX_SIZE=64) pushes M as high as
# 64 * block_length = 2048, so the upper buckets matter in production, not just
# on paper -- get_best_config() picks the *closest* tuned M, so a missing 2048
# bucket would silently fall back to the 512 config for a 4x larger workload.
REALISTIC_M_BUCKETS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]

PADDING_PENALTY_WEIGHT = 0.5

# ═══════════════════════════════════════════════════════════════════════════════
# 2.  CONFIGURATION SEARCH SPACE
#     Pruned against the ACTUAL shared-memory budget of the GPU being tuned on
#     (SHARED_MEM_LIMIT, queried from the device in fused_moe_triton.py) rather
#     than a hardcoded Ampere figure -- H100 allows ~227KB/block vs A40's
#     ~100KB, and pruning an H100 against 96KB throws away exactly the
#     deep-pipelined configs that make Hopper fast.
#     We keep the constraint BLOCK_SIZE_M <= 64 to prevent padding explosion.
# ═══════════════════════════════════════════════════════════════════════════════
def get_config_grid(max_block_m: int = 64, shmem_limit: int = None):
    """
    Returns a list of candidate Triton configs.

    max_block_m: Hard upper cap on BLOCK_SIZE_M.
                 For small M (< 64), a BLOCK_SIZE_M of 128 wastes 8x the
                 compute because every expert slot is padded to 128.  We
                 cap at 64 by default which already allows BLOCK_SIZE_M > M
                 for the M=32 bucket (2x overhead at most).

    shmem_limit: Per-block shared-memory budget in bytes. Defaults to the
                 running GPU's real limit (SHARED_MEM_LIMIT). MUST match the
                 guard in fused_moe_triton.py::get_best_config(), or the tuner
                 can select a config that the loader then rejects at runtime.

    num_stages 4 is included when the budget allows it: it was originally
    excluded because "4 stages often hits shmem limit on A40", which is a
    statement about Ampere's 100KB, not about the config being bad. On H100
    the shmem guard below decides that empirically instead.
    """
    if shmem_limit is None:
        shmem_limit = SHARED_MEM_LIMIT

    block_m   = [bm for bm in [16, 32, 64, 128] if bm <= max_block_m]
    block_n   = [32, 64, 128]
    block_k   = [32, 64, 128]
    group_m   = [1, 4, 8]
    num_warps = [4, 8]
    num_stages= [2, 3, 4]

    configs = []
    for bm, bn, bk, gm, nw, ns in itertools.product(
            block_m, block_n, block_k, group_m, num_warps, num_stages):
        # ── Hard shmem guard ───────────────────────────────────────────────
        # tiles_a + tiles_b per stage, 2 bytes per bf16 element -- via the
        # shared _shmem_bytes() helper so this search space and the runtime
        # guard in get_best_config() cannot drift apart. With the SiLU epilogue
        # fused, GEMM1 keeps two B tiles in flight, so the budget is stricter
        # and some configs that were legal unfused are pruned here.
        shmem = _shmem_bytes(bm, bn, bk, ns, FUSE_SILU)
        if shmem > shmem_limit:
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

    # Pre-allocate caches (same as production code). Which caches exist depends
    # on whether the SiLU epilogue is fused: with it on, GEMM1 writes the
    # EI-wide activated result directly and the 2*EI-wide cache1 never exists.
    # The tuner has to measure the pipeline production actually runs -- its
    # whole premise is scoring the real GEMM1 -> act -> GEMM2 -> sum chain
    # rather than one GEMM in isolation, and the fused epilogue changes both
    # the traffic and the shared-memory budget that scoring depends on.
    fuse_silu = FUSE_SILU
    cache2 = torch.empty((M, top_k, K), device=device, dtype=dtype)
    if fuse_silu:
        act_out = torch.empty((M * top_k, N // 2), device=device, dtype=dtype)
    else:
        cache1 = torch.empty((M, top_k, N), device=device, dtype=dtype)

    def run_full_pipeline():
        if fuse_silu:
            # GEMM 1 + SiLU(gate)*up epilogue
            invoke_fused_moe_kernel(
                hidden_states, w1, act_out,
                topk_weights, topk_ids,
                sorted_token_ids, expert_ids, num_tokens_post_padded,
                False, top_k, config, silu_epilogue=True,
            )
            act = act_out
        else:
            # GEMM 1: hidden @ W1
            invoke_fused_moe_kernel(
                hidden_states, w1, cache1,
                topk_weights, topk_ids,
                sorted_token_ids, expert_ids, num_tokens_post_padded,
                False, top_k, config,
            )
            # SiLU activation (same as fused_moe production code)
            gate, up = cache1.chunk(2, dim=-1)
            act = (torch.nn.functional.silu(gate) * up).view(M * top_k, N // 2)
        # GEMM 2: activated @ W2
        invoke_fused_moe_kernel(
            act, w2, cache2,
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
#: One-shot latch so an introspection failure prints once, not per line.
_PROFILE_WARNED = False


def profile_config(config, K, N):
    """
    Queries the Triton compiler metadata for the fused_moe_kernel compiled
    with the given constexpr values.  Returns a dict with:
      - shared_bytes   : bytes of shared memory per block
      - num_warps      : warps launched per block
      - occupancy_pct  : estimated occupancy (based on SM register limits)
    """
    # The argument list below must track fused_moe_kernel's signature exactly.
    # It silently did not, for two separate reasons, and the bare `except`
    # below turned both into `shmem=None occ=None` on every line of tuner
    # output rather than one loud failure:
    #   - SILU_EPILOGUE was never passed, from the moment that fusion landed;
    #   - a_scale_ptr/b_scale_ptr, stride_bse/stride_bsn and
    #     use_fp8_w8a8/use_int8_w8a16 were still being passed after they were
    #     removed from the kernel as dead vLLM inheritance.
    # Either one raises TypeError, which was caught and discarded. The caller
    # now surfaces the error once instead (see main()).
    try:
        compiled = fused_moe_kernel.warmup(
            # Dummy pointers; warmup only compiles and inspects metadata.
            torch.empty(1, dtype=torch.bfloat16, device="cuda"),   # a_ptr
            torch.empty(1, dtype=torch.bfloat16, device="cuda"),   # b_ptr
            torch.empty(1, dtype=torch.bfloat16, device="cuda"),   # c_ptr
            torch.empty(1, dtype=torch.float32, device="cuda"),    # topk_weights
            torch.empty(1, dtype=torch.int32,   device="cuda"),    # sorted_token_ids
            torch.empty(1, dtype=torch.int32,   device="cuda"),    # expert_ids
            torch.empty(1, dtype=torch.int32,   device="cuda"),    # num_tokens_post_padded
            N, K, 1, 1,    # N, K, EM, num_valid_tokens
            1, 1,          # stride_am, stride_ak
            1, 1, 1,       # stride_be, stride_bk, stride_bn
            1, 1,          # stride_cm, stride_cn
            BLOCK_SIZE_M   = config["BLOCK_SIZE_M"],
            BLOCK_SIZE_N   = config["BLOCK_SIZE_N"],
            BLOCK_SIZE_K   = config["BLOCK_SIZE_K"],
            GROUP_SIZE_M   = config["GROUP_SIZE_M"],
            MUL_ROUTED_WEIGHT = False,
            top_k          = 1,
            compute_type   = tl.bfloat16,
            is_first_gemm  = True,
            SILU_EPILOGUE  = FUSE_SILU,
            num_warps      = config["num_warps"],
            num_stages     = config["num_stages"],
            grid           = (1,),
        )
        meta = compiled.metadata
        shared_bytes = getattr(meta, "shared", None)
        n_regs = getattr(meta, "n_regs", None)
        n_spills = getattr(meta, "n_spills", None)

        # Device-queried, not hardcoded: the previous constants were Ampere/A40
        # (100KB shmem/SM, 48 warps/SM) and under-report H100 by ~25%.
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        shmem_per_sm = getattr(props, "shared_memory_per_multiprocessor", 102400)
        max_threads_per_sm = getattr(props, "max_threads_per_multi_processor", 1536)
        max_warps_per_sm = max(max_threads_per_sm // 32, 1)
        regs_per_sm = getattr(props, "regs_per_multiprocessor", 65536)

        num_warps = getattr(meta, "num_warps", config["num_warps"])
        threads_per_block = num_warps * 32

        # Registers are per THREAD, so the block's register footprint is
        # n_regs * threads_per_block. The old formula divided the SM's register
        # file by the thread count alone, which is a register-per-thread budget,
        # not a block count -- it produced a number ~n_regs times too large and
        # so never actually bound.
        # NOTE: Triton 3.1's warmup metadata does not expose n_regs (it is
        # populated after a real launch, not after compilation), so on that
        # version only the shared-memory limit applies and this number is an
        # UPPER BOUND. Measured against ncu on H100 at BM=64/BN=128: this
        # reports 25%, ncu measures 12.4% -- because registers (234/thread)
        # bind first and are invisible here. Treat it as "shmem does not
        # prevent N blocks", not as achieved occupancy. If a future Triton
        # exposes n_regs the register term below starts applying and the two
        # converge.
        limits = []
        if shared_bytes:
            limits.append(shmem_per_sm // shared_bytes)
        if n_regs:
            limits.append(regs_per_sm // max(n_regs * threads_per_block, 1))
        occupancy_pct = None
        if limits:
            blocks_per_sm = max(min(min(limits), 32), 0)
            occupancy_pct = round(blocks_per_sm * num_warps / max_warps_per_sm * 100)

        return {
            "shared_bytes":   shared_bytes,
            "num_warps":      num_warps,
            "n_regs":         n_regs,
            "n_spills":       n_spills,
            "occupancy_pct":  occupancy_pct,
        }
    except Exception as e:
        return {"shared_bytes": None, "num_warps": config["num_warps"],
                "n_regs": None, "n_spills": None,
                "occupancy_pct": None, "error": f"{type(e).__name__}: {e}"}


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
        # **kw absorbs get_best_config's silu_epilogue argument -- the whole
        # point of the patch is to force `cfg` regardless of what was asked for.
        _ft.get_best_config = lambda m, e, **kw: cfg
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
    global _PROFILE_WARNED
    import argparse
    import os
    ap = argparse.ArgumentParser(
        description="End-to-end aware MoE Autotuner for Triton fused kernels."
    )
    ap.add_argument("--model",        default="FULL_CFG", choices=list(MODEL_SHAPES),
                    help="Model config to tune for (default: FULL_CFG = production model).")
    ap.add_argument("--tp-size",      type=int, default=1,
                    help="Tensor/expert-parallel degree you will RUN with. The MoE "
                         "weights are expert-sharded, so each rank's w1 holds only "
                         "NE//tp_size experts and fused_moe() sees that smaller E. "
                         "Tuning at the global E would optimise padding for a "
                         "workload that never occurs. Tune once on a single GPU "
                         "with the tp_size you intend to deploy; the resulting "
                         "moe_tune_config.json is read by all ranks from the repo "
                         "root, so it does NOT need to be regenerated per GPU.")
    ap.add_argument("--max-block-m",  type=int, default=64,
                    help="Hard cap on BLOCK_SIZE_M to limit padding overhead.")
    ap.add_argument("--penalty",      type=float, default=PADDING_PENALTY_WEIGHT,
                    help="Weight on padding ratio in composite score.")
    ap.add_argument("--top-configs",  type=int, default=3,
                    help="Print profiling for the top-N configs per M.")
    ap.add_argument("--output",       default=None,
                    help="Output JSON path (default: a device-keyed file in <repo_root>, "
                         "e.g. moe_tune_config.device_name=NVIDIA_H100_PCIe.json).")
    ap.add_argument("--verify",       action="store_true", default=True)
    args = ap.parse_args()

    if args.output is None:
        # Default: save to repo root so fused_moe_triton.py picks it up automatically
        repo_root = os.path.dirname(os.path.abspath(__file__))
        # Device-keyed by default. Tile shapes are hardware-specific -- this
        # file used to be one unkeyed moe_tune_config.json, so a config tuned
        # on one GPU loaded silently on another with nothing in the filename
        # to say so. fused_moe_triton.py prefers the keyed name and warns if
        # it falls back to the legacy one.
        from model_update.fused_moe_triton import _device_tag
        args.output = os.path.join(
            repo_root, f"moe_tune_config.device_name={_device_tag()}.json")

    if not torch.cuda.is_available():
        print("CUDA is required."); return

    E, N, K, top_k, desc = MODEL_SHAPES[args.model]

    global_E = E
    if args.tp_size > 1:
        if E % args.tp_size != 0:
            print(f"ERROR: NE={E} is not divisible by --tp-size {args.tp_size}; "
                  f"expert sharding in distributed.py requires an exact split.")
            return
        E = E // args.tp_size

    props = torch.cuda.get_device_properties(0)
    print(f"\n{'='*70}")
    print(f"  End-to-End Aware MoE Autotuner")
    print(f"  Model : {args.model} — {desc}")
    print(f"  GPU   : {torch.cuda.get_device_name(0)} "
          f"(sm_{props.major}{props.minor}, {props.multi_processor_count} SMs)")
    print(f"  shmem : {SHARED_MEM_LIMIT/1024:.0f} KB/block usable (device-queried)")
    if args.tp_size > 1:
        print(f"  TP    : tp_size={args.tp_size} → tuning for E={E} local experts "
              f"(global NE={global_E})")
    else:
        print(f"  TP    : tp_size=1 → tuning for all E={E} experts on one GPU")
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
            # Surface an introspection failure once, loudly, instead of
            # printing shmem=None occ=None on every line forever. The whole
            # point of this block is register/shared-mem pressure; silently
            # reporting nothing is worse than not reporting.
            if prof.get("error") and not _PROFILE_WARNED:
                print(f"    [warn] kernel introspection unavailable: {prof['error']}")
                print( "           shmem/occupancy/register columns will read '?'.")
                _PROFILE_WARNED = True
            bm = cfg["BLOCK_SIZE_M"]
            spill = prof.get("n_spills")
            spill_s = f" spill={spill}" if spill else ""
            print(f"    [{rank+1}] BM={bm:3d} BN={cfg['BLOCK_SIZE_N']:3d} "
                  f"BK={cfg['BLOCK_SIZE_K']:3d} GM={cfg['GROUP_SIZE_M']} "
                  f"nw={cfg['num_warps']} ns={cfg['num_stages']}  "
                  f"lat={lat:.3f}ms  pad={pad:.1%}  score={score:.3f}  "
                  f"shmem={prof.get('shared_bytes') or '?'}B  "
                  f"regs={prof.get('n_regs') or '?'}  "
                  f"occ={prof.get('occupancy_pct') or '?'}%{spill_s}")

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
    print("  To apply: keep this file in the repo root; fused_moe_triton.py")
    print("            loads the device-keyed name matching the GPU it runs on.")
    print(f"  All ranks read this one file at import time — tuning on a single")
    print(f"  GPU is sufficient; do NOT re-run it per GPU.")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
