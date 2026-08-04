"""
Inference Time Benchmark: Baseline vs Optimized.

Compares:
  1. Dense Baseline (src/)  — unfused MoE, no KV cache, topk=8
  2. Optimized     (model_update/) — Triton fused MoE, block-wise KV cache, topk=8
"""

import argparse
import gc
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

os.environ.setdefault("PYTHONUNBUFFERED", "1")

workspace_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(workspace_root))

from model_update.distributed import init_distributed, get_tp_rank, get_tp_size
init_distributed()

MASK_ID = 156895


# ── helpers ──────────────────────────────────────────────────────────────────

def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_stats(latencies):
    l_sorted = sorted(latencies)
    mean_val = sum(l_sorted) / len(l_sorted)
    median_val = l_sorted[len(l_sorted) // 2]
    p95_val = l_sorted[int(len(l_sorted) * 0.95)]
    return mean_val, median_val, p95_val


def free_model(model, device):
    try:
        model.to("cpu")
    except Exception:
        pass
    del model
    gc.collect()
    if "cuda" in device:
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


# ── loaders ──────────────────────────────────────────────────────────────────

def load_baseline(weight_dir, device):
    """Load the original dense (unfused) model from src/."""
    from src.model import LLaDAMoE, load_weights
    if "cuda" in device:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    model = LLaDAMoE().to(torch.bfloat16).eval()
    load_weights(model, weight_dir, verbose=False)
    model = model.to(device)
    if "cuda" in device:
        torch.cuda.synchronize()
    return model, time.perf_counter() - t0


def load_optimized(weight_dir, device, compile_attn=False):
    """Load the optimized model from model_update/ with Triton fused MoE."""
    from model_update.model import LLaDAMoEKV, FULL_CFG, TritonFusedMoEBlock
    from model_update.distributed import load_weights_tp
    if "cuda" in device:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    model = LLaDAMoEKV(FULL_CFG, use_fused_moe=False).to(torch.bfloat16).eval()
    try:
        load_weights_tp(model, weight_dir, verbose=(get_tp_rank() == 0))
        for i, layer in enumerate(model.layers):
            fused_mlp = TritonFusedMoEBlock(layer.mlp.cfg).to(torch.bfloat16)
            fused_mlp.load_state_dict_from_unfused(layer.mlp)
            layer.mlp = fused_mlp
    except Exception as e:
        print(f"  [Rank {get_tp_rank()}] Warning: Failed to load weights: {e}")
    model = model.to(device)
    if compile_attn:
        model.compile_attention()
    if "cuda" in device:
        torch.cuda.synchronize()
    return model, time.perf_counter() - t0


# ── generation wrappers ──────────────────────────────────────────────────────

def baseline_generate(model, prompt_ids, gen_length, steps, block_length):
    """LLaDA iterative unmasking without KV cache (baseline src/ model)."""
    device = prompt_ids.device
    P = prompt_ids.shape[1]
    x = torch.full((1, P + gen_length), MASK_ID, dtype=torch.long, device=device)
    x[:, :P] = prompt_ids

    num_blocks = gen_length // block_length
    steps_per_block = steps // num_blocks

    for block_idx in range(num_blocks):
        bs = P + block_idx * block_length
        be = P + (block_idx + 1) * block_length
        block_mask = (x[:, bs:be] == MASK_ID)
        mask_num = block_mask.sum(dim=1, keepdim=True)
        base = mask_num // steps_per_block
        rem = mask_num % steps_per_block
        ntok = torch.zeros(1, steps_per_block, device=device, dtype=torch.long) + base
        for i in range(1):
            ntok[i, :rem[i]] += 1

        for step in range(steps_per_block):
            mask_index = (x == MASK_ID)
            with torch.no_grad():
                logits = model(x)
            x0 = logits.argmax(dim=-1)
            p = F.softmax(logits.float(), dim=-1)
            x0_p = p.gather(-1, x0.unsqueeze(-1)).squeeze(-1)
            x0_p[:, be:] = -np.inf
            x0 = torch.where(mask_index, x0, x)
            conf = torch.where(mask_index, x0_p, torch.tensor(-np.inf, device=device))
            transfer = torch.zeros_like(x0, dtype=torch.bool)
            k = ntok[0, step].item()
            if k > 0:
                _, sel = torch.topk(conf[0], k=int(k))
                transfer[0, sel] = True
            x[transfer] = x0[transfer]

    return x[0, P:]


def optimized_generate(model, prompt_ids, gen_length, steps, block_length):
    """Generation with block-wise KV cache, fused MoE, static top-8 experts."""
    from model_update.generate import generate_cached
    return generate_cached(model, prompt_ids, gen_length, steps, block_length)


# ── benchmarking ─────────────────────────────────────────────────────────────

def benchmark(gen_fn, device, num_warmup, num_runs):
    """Run warmup + timed runs and return list of latencies."""
    for _ in range(num_warmup):
        gen_fn()
    if "cuda" in device:
        torch.cuda.synchronize()

    latencies = []
    for _ in range(num_runs):
        if "cuda" in device:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        gen_fn()
        if "cuda" in device:
            torch.cuda.synchronize()
        latencies.append(time.perf_counter() - t0)
    return latencies


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Baseline vs Optimized inference time comparison")
    ap.add_argument("--weight-dir", default="weights")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--num-warmup", type=int, default=1)
    ap.add_argument("--num-runs", type=int, default=3)
    ap.add_argument("--gen-length", type=int, default=32)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--block-length", type=int, default=16)
    ap.add_argument("--mode", choices=["both", "baseline", "optimized"], default="both",
                    help="Which model(s) to benchmark")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile the optimized model's Attention blocks "
                         "(the only slice of the step outside the Triton MoE kernel "
                         "large enough to matter; see eval/time_fraction.py)")
    args = ap.parse_args()

    # In distributed mode, override device with local rank
    if get_tp_size() > 1:
        args.device = f"cuda:{get_tp_rank()}"
        
    if "cuda" in args.device:
        torch.cuda.set_device(args.device)
        
    if get_tp_size() > 1:
        if args.mode in ["both", "baseline"]:
            if get_tp_rank() == 0:
                print("Warning: Baseline model does not support tensor parallelism! Switching mode to 'optimized'.")
            args.mode = "optimized"

    is_master = (get_tp_rank() == 0)

    if is_master:
        print("=" * 80)
        print("  Benchmark: Baseline (src/) vs Optimized (model_update/)")
        print("=" * 80)
        print(f"  Device           : {args.device}")
        print(f"  PyTorch Version  : {torch.__version__}")
        print(f"  Weight Directory : {args.weight_dir}")
        print(f"  Gen Length       : {args.gen_length}")
        print(f"  Steps            : {args.steps}")
        print(f"  Block Length     : {args.block_length}")
        print(f"  Warmup Runs      : {args.num_warmup}")
        print(f"  Benchmark Runs   : {args.num_runs}")
        print(f"  Compile Attn     : {args.compile}")
        print("=" * 80 + "\n")

    if not os.path.isdir(args.weight_dir):
        if is_master:
            print(f"Error: Weight directory '{args.weight_dir}' does not exist.")
        sys.exit(1)

    print(f"[Rank {get_tp_rank()}] Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(args.weight_dir, trust_remote_code=True)
    if is_master:
        print("Done.\n")

    test_prompt = "The chemical symbol for gold is Au and for silver is"
    prompt_ids = tok(test_prompt, return_tensors="pt")["input_ids"].to(args.device)

    # ── 1. Dense Baseline ────────────────────────────────────────────────────
    baseline_tokens = None
    bl_mean = 0.0
    bl_tps = 0.0

    if args.mode in ["both", "baseline"]:
        if is_master:
            print("=" * 60)
            print("  1. DENSE BASELINE  (src/, unfused MoE, topk=8, no cache)")
            print("=" * 60)
        baseline_model, baseline_load_time = load_baseline(args.weight_dir, args.device)
        if is_master:
            print(f"  Loaded in {baseline_load_time:.2f}s")

        set_seed(42)
        baseline_tokens = baseline_generate(
            baseline_model, prompt_ids, args.gen_length, args.steps, args.block_length
        ).cpu()

        baseline_lats = benchmark(
            lambda: baseline_generate(baseline_model, prompt_ids, args.gen_length, args.steps, args.block_length),
            args.device, args.num_warmup, args.num_runs,
        )
        bl_mean, bl_med, bl_p95 = get_stats(baseline_lats)
        bl_tps = args.gen_length / bl_mean
        if is_master:
            print(f"  Mean: {bl_mean:.2f}s | Median: {bl_med:.2f}s | P95: {bl_p95:.2f}s | {bl_tps:.2f} tok/s\n")

        free_model(baseline_model, args.device)

    # ── 2. Optimized ─────────────────────────────────────────────────────────
    opt_tokens = None
    opt_mean = 0.0
    opt_tps = 0.0

    if args.mode in ["both", "optimized"]:
        if is_master:
            print("=" * 60)
            print("  2. OPTIMIZED  (model_update/, fused MoE, topk=8, KV cache)")
            print("=" * 60)
        opt_model, opt_load_time = load_optimized(args.weight_dir, args.device, compile_attn=args.compile)
        if is_master:
            print(f"  Loaded in {opt_load_time:.2f}s")

        set_seed(42)
        opt_tokens = optimized_generate(
            opt_model, prompt_ids, args.gen_length, args.steps, args.block_length
        )[0].cpu()

        opt_lats = benchmark(
            lambda: optimized_generate(opt_model, prompt_ids, args.gen_length, args.steps, args.block_length),
            args.device, args.num_warmup, args.num_runs,
        )
        opt_mean, opt_med, opt_p95 = get_stats(opt_lats)
        opt_tps = args.gen_length / opt_mean
        if is_master:
            print(f"  Mean: {opt_mean:.2f}s | Median: {opt_med:.2f}s | P95: {opt_p95:.2f}s | {opt_tps:.2f} tok/s\n")

        free_model(opt_model, args.device)

    # Worker ranks are done — only master prints the final table
    if not is_master:
        return

    # ── Results ──────────────────────────────────────────────────────────────
    speedup = bl_mean / opt_mean if opt_mean > 0 and bl_mean > 0 else float("inf")

    print("=" * 100)
    print("                              RESULTS")
    print("=" * 100)
    header = f"| {'Configuration':<50} | {'Time (s)':>10} | {'Tok/s':>10} | {'Speedup':>10} |"
    sep    = f"|{'-'*50}|{'-'*12}|{'-'*12}|{'-'*12}|"
    print(header)
    print(sep)

    if args.mode in ["both", "baseline"]:
        print(f"| {'Baseline (src/, topk=8, no cache)':<50} | {bl_mean:>10.2f} | {bl_tps:>10.2f} | {'1.00x':>10} |")
    if args.mode in ["both", "optimized"]:
        speed_text = f"{speedup:.2f}x" if args.mode == "both" else "N/A"
        print(f"| {'Optimized (model_update/, topk=8, KV cache)':<50} | {opt_mean:>10.2f} | {opt_tps:>10.2f} | {speed_text:>10} |")
    print("=" * 100)

    if args.mode == "both" and baseline_tokens is not None and opt_tokens is not None:
        min_len = min(len(baseline_tokens), len(opt_tokens))
        diff_count = (baseline_tokens[:min_len] != opt_tokens[:min_len]).sum().item()
        div_pct = (diff_count / min_len) * 100
        print(f"\n  Token Divergence : {diff_count}/{min_len} tokens ({div_pct:.2f}%)")
        print(f"  Speedup          : {speedup:.2f}x")
        if speedup > 1:
            print(f"\n  ✅ Optimized model is {speedup:.2f}x faster than baseline.")
        else:
            print(f"\n  ⚠️  Optimized model is {1/speedup:.2f}x slower than baseline.")
        decoded_baseline = tok.decode(baseline_tokens, skip_special_tokens=True)
        decoded_opt = tok.decode(opt_tokens, skip_special_tokens=True)
        print(f"\n  Baseline output : {decoded_baseline[:200]}")
        print(f"  Optimized output: {decoded_opt[:200]}")
    elif args.mode == "optimized" and opt_tokens is not None:
        decoded_opt = tok.decode(opt_tokens, skip_special_tokens=True)
        print(f"\n  Optimized output: {decoded_opt[:200]}")


if __name__ == "__main__":
    main()