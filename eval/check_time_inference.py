"""
Option A Benchmark: Conservative speedup with accuracy safety.

Compares:
  1. Dense Baseline (src/)
  2. Block-wise KV Cache (model_update/)
  3. Dynamic Experts Search (DES) (model_update/)
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

os.environ.setdefault("PYTHONUNBUFFERED", "1")

workspace_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(workspace_root))

MASK_ID = 156895


def load_baseline(weight_dir, device):
    from src.model import LLaDAMoE, load_weights
    if "cuda" in device:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    model = LLaDAMoE().to(torch.bfloat16).to(device).eval()
    load_weights(model, weight_dir, verbose=False)
    if "cuda" in device:
        torch.cuda.synchronize()
    return model, time.perf_counter() - t0


def load_new_approach(weight_dir, device):
    from model_update.model import LLaDAMoEKV, FULL_CFG
    from src.model import load_weights
    if "cuda" in device:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    model = LLaDAMoEKV(FULL_CFG).to(torch.bfloat16).to(device).eval()
    try:
        load_weights(model, weight_dir, verbose=False)
    except Exception as e:
        print(f"Warning: Failed to load weights: {e}")
    if "cuda" in device:
        torch.cuda.synchronize()
    return model, time.perf_counter() - t0


def benchmark_generation(model, device, prompt_ids, gen_length, steps, block_length,
                         num_warmup, num_runs, is_new=False, **kwargs):
    if is_new:
        from model_update.generate import generate_cached
        gen_fn = lambda: generate_cached(
            model, prompt_ids, gen_length, steps, block_length, **kwargs
        )
    else:
        def diffusion_generate(model, prompt_ids, gen_length=64, steps=64, block_length=32):
            import numpy as np
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
        gen_fn = lambda: diffusion_generate(model, prompt_ids, gen_length, steps, block_length)

    for _ in range(num_warmup):
        _ = gen_fn()
    if "cuda" in device:
        torch.cuda.synchronize()

    latencies = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        _ = gen_fn()
        if "cuda" in device:
            torch.cuda.synchronize()
        latencies.append(time.perf_counter() - t0)
    return latencies


def get_stats(latencies):
    l_sorted = sorted(latencies)
    mean_val = sum(l_sorted) / len(l_sorted)
    return mean_val, l_sorted[len(l_sorted) // 2], l_sorted[int(len(l_sorted) * 0.95)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight-dir", default="weights")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--num-warmup", type=int, default=1)
    ap.add_argument("--num-runs", type=int, default=3)
    ap.add_argument("--gen-length", type=int, default=32)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--block-length", type=int, default=16)
    args = ap.parse_args()

    print(f"================================================================")
    print(f" Benchmark: Inference Time Comparison")
    print(f"================================================================")
    print(f"  Device           : {args.device}")
    print(f"  PyTorch Version  : {torch.__version__}")
    print(f"  Weight Directory : {args.weight_dir}")
    print(f"  Warmup Runs      : {args.num_warmup}")
    print(f"  Benchmark Runs   : {args.num_runs}")
    print(f"================================================================\n")

    if not os.path.exists(args.weight_dir) or not os.path.isdir(args.weight_dir):
        print(f"Error: Weight directory '{args.weight_dir}' does not exist.")
        sys.exit(1)

    print("Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(args.weight_dir, trust_remote_code=True)
    print("Done.\n")

    test_prompt = "The chemical symbol for gold is Au and for silver is"
    prompt_ids = tok(test_prompt, return_tensors="pt")["input_ids"].to(args.device)

    # 1. Baseline
    print("================ 1. DENSE BASELINE ================")
    baseline, baseline_load_time = load_baseline(args.weight_dir, args.device)
    print(f"  Baseline loaded in {baseline_load_time:.2f} seconds.")
    baseline_gen_lats = benchmark_generation(
        baseline, args.device, prompt_ids, args.gen_length, args.steps,
        args.block_length, args.num_warmup, args.num_runs, is_new=False
    )
    baseline_gen_mean, _, _ = get_stats(baseline_gen_lats)
    baseline_tok_per_sec = args.gen_length / baseline_gen_mean
    print(f"  Mean latency: {baseline_gen_mean:.2f}s ({baseline_tok_per_sec:.2f} tok/s)\n")

    del baseline
    import gc
    gc.collect()
    if "cuda" in args.device:
        torch.cuda.empty_cache()

    # 2. New Approach
    print("================ LOADING NEW APPROACH ================")
    new_model, new_load_time = load_new_approach(args.weight_dir, args.device)
    print(f"  New Approach loaded in {new_load_time:.2f} seconds.\n")

    configs = []

    configs.append(("2. CACHE ONLY (Block-wise)", {
        "is_new": True, "use_dynamic_experts": False,
    }))
    configs.append(("3. CACHE + DYNAMIC EXPERTS (min_k=4)", {
        "is_new": True, "use_dynamic_experts": True, "min_k": 4, "base_k": 8, "expert_threshold": 0.0,
    }))
    configs.append(("4. CACHE + DYNAMIC EXPERTS (min_k=5)", {
        "is_new": True, "use_dynamic_experts": True, "min_k": 5, "base_k": 8, "expert_threshold": 0.0,
    }))
    configs.append(("5. CACHE + DYNAMIC EXPERTS (min_k=6)", {
        "is_new": True, "use_dynamic_experts": True, "min_k": 6, "base_k": 8, "expert_threshold": 0.0,
    }))

    results = [("1. DENSE BASELINE", baseline_gen_mean, baseline_tok_per_sec, "0.00%")]

    def set_seed(seed=42):
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # Cache-only output for divergence comparison
    set_seed(42)
    from model_update.generate import generate_cached
    ref_tokens = generate_cached(
        new_model, prompt_ids, args.gen_length, args.steps, args.block_length, use_dynamic_experts=False
    )[0].cpu()

    for name, kwargs in configs:
        print(f"================ {name} ================")
        set_seed(42)
        gen_kwargs = {k: v for k, v in kwargs.items() if k != "is_new"}
        test_tokens = generate_cached(
            new_model, prompt_ids, args.gen_length, args.steps, args.block_length, **gen_kwargs
        )[0].cpu()
        
        diff_count = (ref_tokens != test_tokens).sum().item()
        div_pct = f"{(diff_count / len(ref_tokens)) * 100:.2f}%"

        gen_lats = benchmark_generation(
            new_model, args.device, prompt_ids, args.gen_length, args.steps,
            args.block_length, args.num_warmup, args.num_runs, **kwargs
        )
        gen_mean, _, _ = get_stats(gen_lats)
        tok_per_sec = args.gen_length / gen_mean
        print(f"  Mean latency: {gen_mean:.2f}s ({tok_per_sec:.2f} tok/s) | Divergence vs Cache-Only: {div_pct}\n")
        results.append((name, gen_mean, tok_per_sec, div_pct))

    del new_model
    gc.collect()
    if "cuda" in args.device:
        torch.cuda.empty_cache()

    # Output Results Table
    print("=" * 140)
    print("                                   BENCHMARK & DIVERGENCE STANDING CHECK")
    print("=" * 140)
    print(f"| {'Configuration':<48} | {'Time (sec)':>12} | {'Tok/sec':>10} | {'Speedup':>10} | {'Token Div %':>12} |")
    print(f"|{'-'*48}|{'-'*14}|{'-'*12}|{'-'*12}|{'-'*14}|")
    baseline_time = results[0][1]
    for name, t, tps, div in results:
        speedup = baseline_time / t if t > 0 else 0
        print(f"| {name:<48} | {t:>12.2f} | {tps:>10.2f} | {speedup:>9.2f}x | {div:>12} |")
    print("=" * 140)

    best_name, best_time, best_tps, best_div = min(results[1:], key=lambda x: x[1])
    best_speedup = baseline_time / best_time
    print(f"\n🏆 FASTEST CONFIG: {best_name}")
    print(f"   Speedup: {best_speedup:.2f}x vs baseline")
    print(f"   Time: {best_time:.2f}s | Throughput: {best_tps:.2f} tok/s | Token Divergence: {best_div}")

if __name__ == "__main__":
    main()