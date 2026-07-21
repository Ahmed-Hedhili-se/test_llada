"""
Inference time and latency benchmark comparing Baseline (src/) vs New Approach (model_update/).
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


def benchmark_generation(model, device: str, prompt_ids, gen_length: int, steps: int, block_length: int, num_warmup: int, num_runs: int, is_new: bool = False):
    if is_new:
        from model_update.generate import generate_sparse_cached
        gen_fn = lambda: generate_sparse_cached(model, prompt_ids, gen_length, steps, block_length, temperature=0.0)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight-dir", default="weights", help="Directory where model weights are stored")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device to run benchmark on")
    ap.add_argument("--num-warmup", type=int, default=3, help="Number of warmup iterations")
    ap.add_argument("--num-runs", type=int, default=10, help="Number of measured benchmark iterations")
    ap.add_argument("--gen-length", type=int, default=64, help="Tokens to generate in generation benchmark")
    ap.add_argument("--steps", type=int, default=64, help="Steps for diffusion generation benchmark")
    ap.add_argument("--block-length", type=int, default=32, help="Block length for generation benchmark")
    args = ap.parse_args()

    print(f"================================================================")
    print(f" LLaDA-MoE Inference Speed Benchmark: Baseline vs New")
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

    seq_lengths = [128, 256, 512, 1024]
    forward_results = {seq_len: {} for seq_len in seq_lengths}

    test_prompt = "The chemical symbol for gold is Au and for silver is"
    prompt_ids = tok(test_prompt, return_tensors="pt")["input_ids"].to(args.device)

    # 1. Load and Benchmark Baseline Model
    print("Benchmarking model loading time...")
    print("  Loading baseline model (src/)...")
    baseline, baseline_load_time = load_baseline(args.weight_dir, args.device)
    print(f"  Baseline loaded in {baseline_load_time:.2f} seconds.")
    
    baseline_peak_mem = 0
    if "cuda" in args.device:
        baseline_peak_mem = torch.cuda.max_memory_allocated(device=args.device) / (1024 ** 3)
        torch.cuda.reset_peak_memory_stats(device=args.device)

    print("Benchmarking single forward pass latency for baseline model...")
    for seq_len in seq_lengths:
        print(f"  Running sequence length {seq_len}...")
        baseline_lats = benchmark_forward(baseline, args.device, seq_len, args.num_warmup, args.num_runs)
        baseline_mean, baseline_med, baseline_p95 = get_stats(baseline_lats)
        forward_results[seq_len]["baseline"] = (baseline_mean * 1000, baseline_med * 1000, baseline_p95 * 1000)
    print("Done.\n")

    print("Benchmarking full generation (diffusion decode) for baseline model...")
    print(f"  Prompt: {repr(test_prompt)}")
    print(f"  Config: Gen Length={args.gen_length}, Steps={args.steps}, Block Length={args.block_length}")
    
    baseline_gen_lats = benchmark_generation(
        baseline, args.device, prompt_ids, args.gen_length, args.steps, 
        args.block_length, args.num_warmup, args.num_runs, is_new=False
    )
    baseline_gen_mean, baseline_gen_med, baseline_gen_p95 = get_stats(baseline_gen_lats)
    baseline_tok_per_sec = args.gen_length / baseline_gen_mean
    baseline_ms_per_step = (baseline_gen_mean * 1000) / args.steps
    print("Done.\n")

    # UNLOAD BASELINE TO FREE MEMORY
    print("Unloading baseline model to free memory...")
    del baseline
    import gc
    gc.collect()
    if "cuda" in args.device:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device=args.device)
    print("Done.\n")

    # 2. Load and Benchmark New Approach Model
    print("Benchmarking model loading time...")
    print("  Loading new approach model (model_update/)...")
    new_model, new_load_time = load_new_approach(args.weight_dir, args.device)
    print(f"  New Approach loaded in {new_load_time:.2f} seconds.")
    
    new_peak_mem = 0
    if "cuda" in args.device:
        new_peak_mem = torch.cuda.max_memory_allocated(device=args.device) / (1024 ** 3)
        torch.cuda.reset_peak_memory_stats(device=args.device)

    print("Benchmarking single forward pass latency for new approach model...")
    for seq_len in seq_lengths:
        print(f"  Running sequence length {seq_len}...")
        new_lats = benchmark_forward(new_model, args.device, seq_len, args.num_warmup, args.num_runs)
        new_mean, new_med, new_p95 = get_stats(new_lats)
        forward_results[seq_len]["new"] = (new_mean * 1000, new_med * 1000, new_p95 * 1000)
    print("Done.\n")

    print("Benchmarking full generation (diffusion decode) for new approach model...")
    new_gen_lats = benchmark_generation(
        new_model, args.device, prompt_ids, args.gen_length, args.steps, 
        args.block_length, args.num_warmup, args.num_runs, is_new=True
    )
    new_gen_mean, new_gen_med, new_gen_p95 = get_stats(new_gen_lats)
    new_tok_per_sec = args.gen_length / new_gen_mean
    new_ms_per_step = (new_gen_mean * 1000) / args.steps
    print("Done.\n")

    print("Unloading new approach model to free memory...")
    del new_model
    gc.collect()
    if "cuda" in args.device:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device=args.device)
    print("Done.\n")

    # Output Results Table
    print("=" * 85)
    print("                           BENCHMARK RESULTS COMPARISON")
    print("=" * 85)
    print(f"| Metric / Test Case                | Baseline (src)        | New (model_update)    | Speedup |")
    print(f"|-----------------------------------|-----------------------|-----------------------|---------|")
    print(f"| Model Loading Time (sec)          | {baseline_load_time:21.2f} | {new_load_time:21.2f} | {baseline_load_time/new_load_time:6.2f}x |")
    if "cuda" in args.device:
        print(f"| Peak GPU Memory Usage (GB)        | {baseline_peak_mem:21.2f} | {new_peak_mem:21.2f} | {baseline_peak_mem/new_peak_mem:6.2f}x |")
    
    print(f"|-----------------------------------|-----------------------|-----------------------|---------|")
    
    # Forward passes
    for seq_len in seq_lengths:
        b_mean, b_med, b_p95 = forward_results[seq_len]["baseline"]
        n_mean, n_med, n_p95 = forward_results[seq_len]["new"]
        print(f"| Fwd Pass Latency ({seq_len:4d} tokens)   |                       |                       |         |")
        print(f"|   - Mean (ms)                     | {b_mean:21.2f} | {n_mean:21.2f} | {b_mean/n_mean:6.2f}x |")
        print(f"|   - Median / p50 (ms)             | {b_med:21.2f} | {n_med:21.2f} | {b_med/n_med:6.2f}x |")
        print(f"|   - p95 (ms)                      | {b_p95:21.2f} | {n_p95:21.2f} | {b_p95/n_p95:6.2f}x |")
        print(f"|-----------------------------------|-----------------------|-----------------------|---------|")

    # Generation metrics
    print(f"| Generation Benchmark              |                       |                       |         |")
    print(f"|   - Total Time (sec)              | {baseline_gen_mean:21.2f} | {new_gen_mean:21.2f} | {baseline_gen_mean/new_gen_mean:6.2f}x |")
    print(f"|   - Time per Step (ms)            | {baseline_ms_per_step:21.2f} | {new_ms_per_step:21.2f} | {baseline_ms_per_step/new_ms_per_step:6.2f}x |")
    print(f"|   - Throughput (tokens/sec)       | {baseline_tok_per_sec:21.2f} | {new_tok_per_sec:21.2f} | {new_tok_per_sec/baseline_tok_per_sec:6.2f}x |")
    print("=" * 85)
    print("\nNote: Speedup > 1.0x indicates that the New approach is faster than the Baseline.")


if __name__ == "__main__":
    main()
