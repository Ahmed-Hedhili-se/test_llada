"""
Inference time and latency benchmark comparing Baseline (src/) vs multiple Sparse-dLLM configurations (model_update/).
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

# Set unbuffered output
os.environ.setdefault("PYTHONUNBUFFERED", "1")

# Add the workspace root to sys.path
workspace_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(workspace_root))

MASK_ID = 156895

def load_baseline(weight_dir: str, device: str):
    from src.model import LLaDAMoE, load_weights
    
    if "cuda" in device:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    
    model = LLaDAMoE().to(torch.bfloat16).to(device).eval()
    load_weights(model, weight_dir, verbose=False)
    
    if "cuda" in device:
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    
    return model, t1 - t0

def load_new_approach(weight_dir: str, device: str):
    from model_update.model import LLaDAMoE, load_weights
    
    if "cuda" in device:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    
    model = LLaDAMoE().to(torch.bfloat16).to(device).eval()
    load_weights(model, weight_dir, verbose=False)
    
    if "cuda" in device:
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    
    return model, t1 - t0


def benchmark_forward(model, device: str, seq_len: int, num_warmup: int, num_runs: int):
    x = torch.full((1, seq_len), MASK_ID, dtype=torch.long, device=device)
    
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(x)
            
    if "cuda" in device:
        torch.cuda.synchronize()
        
    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = model(x)
            if "cuda" in device:
                torch.cuda.synchronize()
            latencies.append(time.perf_counter() - t0)
            
    return latencies


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
        rem  = mask_num % steps_per_block
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


def benchmark_generation(model, device: str, prompt_ids, gen_length: int, steps: int, block_length: int, num_warmup: int, num_runs: int, 
                         is_new: bool = False, cache_budget: int = None, saliency_update_interval: int = 8, sparse_pattern = None):
    if is_new:
        from model_update.generate import generate_sparse_cached
        gen_fn = lambda: generate_sparse_cached(
            model, prompt_ids, gen_length, steps, block_length, 
            temperature=0.0, cache_budget=cache_budget, 
            saliency_update_interval=saliency_update_interval, 
            sparse_pattern=sparse_pattern
        )
    else:
        gen_fn = lambda: diffusion_generate(model, prompt_ids, gen_length, steps, block_length)

    # Warmup
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
    median_val = l_sorted[len(l_sorted) // 2]
    p95_val = l_sorted[int(len(l_sorted) * 0.95)]
    return mean_val, median_val, p95_val


def create_dummy_sparse_pattern(num_layers, num_heads):
    from model_update.kv_cache import SparsePattern
    window = torch.full((num_layers, num_heads), 64, dtype=torch.long)
    stride = torch.full((num_layers, num_heads), 16, dtype=torch.long)
    return SparsePattern(num_layers, num_heads, window, stride)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight-dir", default="weights", help="Directory where model weights are stored")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device to run benchmark on")
    ap.add_argument("--num-warmup", type=int, default=1, help="Number of warmup iterations")
    ap.add_argument("--num-runs", type=int, default=3, help="Number of measured benchmark iterations")
    ap.add_argument("--gen-length", type=int, default=64, help="Tokens to generate in generation benchmark")
    ap.add_argument("--steps", type=int, default=64, help="Steps for diffusion generation benchmark")
    ap.add_argument("--block-length", type=int, default=32, help="Block length for generation benchmark")
    args = ap.parse_args()

    print(f"================================================================")
    print(f" LLaDA-MoE Inference Speed Benchmark: 4-Way Comparison")
    print(f"================================================================")
    print(f"  Device           : {args.device}")
    print(f"  PyTorch Version  : {torch.__version__}")
    print(f"  Weight Directory : {args.weight_dir}")
    print(f"  Warmup Runs      : {args.num_warmup}")
    print(f"  Benchmark Runs   : {args.num_runs}")
    print(f"================================================================\n")

    # Verify weights directory
    if not os.path.exists(args.weight_dir) or not os.path.isdir(args.weight_dir):
        print(f"Error: Weight directory '{args.weight_dir}' does not exist.")
        sys.exit(1)

    print("Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(args.weight_dir, trust_remote_code=True)
    print("Done.\n")

    test_prompt = "The chemical symbol for gold is Au and for silver is"
    prompt_ids = tok(test_prompt, return_tensors="pt")["input_ids"].to(args.device)

    # 1. Load and Benchmark Baseline Model
    print("================ 1. DENSE BASELINE ================")
    baseline, baseline_load_time = load_baseline(args.weight_dir, args.device)
    print(f"  Baseline loaded in {baseline_load_time:.2f} seconds.")
    
    print("Benchmarking full generation (diffusion decode)...")
    baseline_gen_lats = benchmark_generation(
        baseline, args.device, prompt_ids, args.gen_length, args.steps, 
        args.block_length, args.num_warmup, args.num_runs, is_new=False
    )
    baseline_gen_mean, _, _ = get_stats(baseline_gen_lats)
    baseline_tok_per_sec = args.gen_length / baseline_gen_mean
    baseline_ms_per_step = (baseline_gen_mean * 1000) / args.steps
    print(f"  Mean latency: {baseline_gen_mean:.2f}s ({baseline_tok_per_sec:.2f} tok/s)\n")

    print("Unloading baseline model to free memory...")
    del baseline
    import gc
    gc.collect()
    if "cuda" in args.device:
        torch.cuda.empty_cache()
    print("Done.\n")

    # 2. Load New Approach Model
    print("================ LOADING NEW APPROACH ================")
    new_model, new_load_time = load_new_approach(args.weight_dir, args.device)
    print(f"  New Approach loaded in {new_load_time:.2f} seconds.\n")

    NL = len(new_model.layers)
    dummy_pattern = create_dummy_sparse_pattern(NL, 16)

    # Cache Only
    print("================ 2. CACHE ONLY (NO SPARSITY) ================")
    print("  Settings: cache_budget=2048, saliency_update_interval=8")
    cache_only_gen_lats = benchmark_generation(
        new_model, args.device, prompt_ids, args.gen_length, args.steps, 
        args.block_length, args.num_warmup, args.num_runs, is_new=True, 
        cache_budget=2048, saliency_update_interval=8, sparse_pattern=None
    )
    cache_only_gen_mean, _, _ = get_stats(cache_only_gen_lats)
    cache_only_tok_per_sec = args.gen_length / cache_only_gen_mean
    print(f"  Mean latency: {cache_only_gen_mean:.2f}s ({cache_only_tok_per_sec:.2f} tok/s)\n")

    # Cache + SparseD
    print("================ 3. CACHE + SPARSED ================")
    print("  Settings: cache_budget=2048, saliency_update_interval=8, sparse_pattern=dummy")
    cache_sparse_gen_lats = benchmark_generation(
        new_model, args.device, prompt_ids, args.gen_length, args.steps, 
        args.block_length, args.num_warmup, args.num_runs, is_new=True, 
        cache_budget=2048, saliency_update_interval=8, sparse_pattern=dummy_pattern
    )
    cache_sparse_gen_mean, _, _ = get_stats(cache_sparse_gen_lats)
    cache_sparse_tok_per_sec = args.gen_length / cache_sparse_gen_mean
    print(f"  Mean latency: {cache_sparse_gen_mean:.2f}s ({cache_sparse_tok_per_sec:.2f} tok/s)\n")

    # Full Aggressive
    print("================ 4. AGGRESSIVE (FULL COMBO) ================")
    print("  Settings: cache_budget=1024, saliency_update_interval=16, sparse_pattern=dummy")
    full_combo_gen_lats = benchmark_generation(
        new_model, args.device, prompt_ids, args.gen_length, args.steps, 
        args.block_length, args.num_warmup, args.num_runs, is_new=True, 
        cache_budget=1024, saliency_update_interval=16, sparse_pattern=dummy_pattern
    )
    full_combo_gen_mean, _, _ = get_stats(full_combo_gen_lats)
    full_combo_tok_per_sec = args.gen_length / full_combo_gen_mean
    print(f"  Mean latency: {full_combo_gen_mean:.2f}s ({full_combo_tok_per_sec:.2f} tok/s)\n")

    print("Unloading new approach model to free memory...")
    del new_model
    gc.collect()
    if "cuda" in args.device:
        torch.cuda.empty_cache()
    print("Done.\n")

    # Output Results Table
    print("=" * 115)
    print("                                   BENCHMARK RESULTS COMPARISON")
    print("=" * 115)
    print(f"| Metric                      | Baseline       | Cache Only     | Cache + SparseD | Aggressive Full |")
    print(f"|-----------------------------|----------------|----------------|-----------------|-----------------|")
    print(f"| Total Time (sec)            | {baseline_gen_mean:14.2f} | {cache_only_gen_mean:14.2f} | {cache_sparse_gen_mean:15.2f} | {full_combo_gen_mean:15.2f} |")
    print(f"| Throughput (tokens/sec)     | {baseline_tok_per_sec:14.2f} | {cache_only_tok_per_sec:14.2f} | {cache_sparse_tok_per_sec:15.2f} | {full_combo_tok_per_sec:15.2f} |")
    print(f"| Speedup vs Baseline         |           1.0x | {baseline_gen_mean/cache_only_gen_mean:13.2f}x | {baseline_gen_mean/cache_sparse_gen_mean:14.2f}x | {baseline_gen_mean/full_combo_gen_mean:14.2f}x |")
    print("=" * 115)

if __name__ == "__main__":
    main()
