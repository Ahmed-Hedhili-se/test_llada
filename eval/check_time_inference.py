"""
Inference time and latency benchmark comparing our LLaDA-MoE implementation vs Hugging Face reference.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

# Set unbuffered output
os.environ.setdefault("PYTHONUNBUFFERED", "1")

# Add the workspace root to sys.path to import src.model
workspace_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(workspace_root))

MASK_ID = 156895


def load_ours(weight_dir: str, device: str):
    from src.model import LLaDAMoE, load_weights
    
    if "cuda" in device:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    
    # Load model
    model = LLaDAMoE().to(torch.bfloat16).to(device).eval()
    load_weights(model, weight_dir, verbose=False)
    
    if "cuda" in device:
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    
    return model, t1 - t0


def load_ours_kv(weight_dir: str, device: str):
    from src.Model_KVcache import LLaDAMoEKV
    from src.model import load_weights
    
    if "cuda" in device:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    
    # Load model
    model = LLaDAMoEKV().to(torch.bfloat16).to(device).eval()
    load_weights(model, weight_dir, verbose=False)
    
    if "cuda" in device:
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    
    return model, t1 - t0


def load_hf(weight_dir: str, device: str):
    if "cuda" in device:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    
    model = AutoModelForCausalLM.from_pretrained(
        weight_dir,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to(device).eval()
    
    if "cuda" in device:
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    
    return model, t1 - t0


def benchmark_forward(model, device: str, seq_len: int, num_warmup: int, num_runs: int, is_hf: bool):
    x = torch.full((1, seq_len), MASK_ID, dtype=torch.long, device=device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(x).logits if is_hf else model(x)
            
    if "cuda" in device:
        torch.cuda.synchronize()
        
    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = model(x).logits if is_hf else model(x)
            if "cuda" in device:
                torch.cuda.synchronize()
            latencies.append(time.perf_counter() - t0)
            
    return latencies


def diffusion_generate(model, prompt_ids, gen_length=64, steps=64, block_length=32, is_hf=False):
    """Run the masked diffusion decode loop, works for both our model and HF."""
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
                logits = model(x).logits if is_hf else model(x)
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


def benchmark_generation(model, device: str, prompt_ids, gen_length: int, steps: int, block_length: int, num_warmup: int, num_runs: int, is_hf: bool, is_kv: bool = False):
    if is_kv:
        from src.generate_KVcache import generate_cached as generate_kv
        gen_fn = lambda: generate_kv(model, prompt_ids, gen_length, steps, block_length, temperature=0.0)
    else:
        gen_fn = lambda: diffusion_generate(model, prompt_ids, gen_length, steps, block_length, is_hf=is_hf)

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
    print(f" LLaDA-MoE Inference Speed & Latency Benchmark")
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
        print(f"Please run setup.sh or specify the correct weight directory via --weight-dir")
        sys.exit(1)

    print("Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(args.weight_dir, trust_remote_code=True)
    print("Done.\n")

    # 1. Load and Benchmark Custom Model
    print("Benchmarking model loading time...")
    print("  Loading our custom model implementation...")
    ours, ours_load_time = load_ours(args.weight_dir, args.device)
    print(f"  Ours loaded in {ours_load_time:.2f} seconds.")
    
    # Measure memory after loading ours
    ours_peak_mem = 0
    if "cuda" in args.device:
        ours_peak_mem = torch.cuda.max_memory_allocated(device=args.device) / (1024 ** 3)
        torch.cuda.reset_peak_memory_stats(device=args.device)

    # 2. Benchmark Single Forward Pass Latency for Ours
    seq_lengths = [128, 256, 512, 1024]
    forward_results = {seq_len: {} for seq_len in seq_lengths}
    
    print("Benchmarking single forward pass latency for custom model...")
    for seq_len in seq_lengths:
        print(f"  Running sequence length {seq_len}...")
        ours_lats = benchmark_forward(ours, args.device, seq_len, args.num_warmup, args.num_runs, is_hf=False)
        ours_mean, ours_med, ours_p95 = get_stats(ours_lats)
        forward_results[seq_len]["ours"] = (ours_mean * 1000, ours_med * 1000, ours_p95 * 1000)
    print("Done.\n")

    # 3. Benchmark Iterative Generation Latency for Ours
    test_prompt = "The chemical symbol for gold is Au and for silver is"
    prompt_ids = tok(test_prompt, return_tensors="pt")["input_ids"].to(args.device)
    
    print("Benchmarking full generation (diffusion decode) for custom model...")
    print(f"  Prompt: {repr(test_prompt)}")
    print(f"  Config: Gen Length={args.gen_length}, Steps={args.steps}, Block Length={args.block_length}")
    
    ours_gen_lats = benchmark_generation(
        ours, args.device, prompt_ids, args.gen_length, args.steps, 
        args.block_length, args.num_warmup, args.num_runs, is_hf=False
    )
    ours_gen_mean, ours_gen_med, ours_gen_p95 = get_stats(ours_gen_lats)
    ours_tok_per_sec = args.gen_length / ours_gen_mean
    ours_ms_per_step = (ours_gen_mean * 1000) / args.steps
    print("Done.\n")

    # UNLOAD OURS TO FREE MEMORY
    print("Unloading custom model to free memory...")
    del ours
    import gc
    gc.collect()
    if "cuda" in args.device:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device=args.device)
    print("Done.\n")

    # 3.5 Load and Benchmark KV Custom Model
    print("Benchmarking model loading time for KV model...")
    ours_kv, ours_kv_load_time = load_ours_kv(args.weight_dir, args.device)
    print(f"  Ours KV loaded in {ours_kv_load_time:.2f} seconds.")
    
    ours_kv_peak_mem = 0
    if "cuda" in args.device:
        ours_kv_peak_mem = torch.cuda.max_memory_allocated(device=args.device) / (1024 ** 3)
        torch.cuda.reset_peak_memory_stats(device=args.device)

    print("Benchmarking single forward pass latency for KV custom model...")
    for seq_len in seq_lengths:
        print(f"  Running sequence length {seq_len}...")
        ours_kv_lats = benchmark_forward(ours_kv, args.device, seq_len, args.num_warmup, args.num_runs, is_hf=False)
        ours_kv_mean, ours_kv_med, ours_kv_p95 = get_stats(ours_kv_lats)
        forward_results[seq_len]["ours_kv"] = (ours_kv_mean * 1000, ours_kv_med * 1000, ours_kv_p95 * 1000)
    print("Done.\n")

    print("Benchmarking full generation (diffusion decode) for KV custom model...")
    ours_kv_gen_lats = benchmark_generation(
        ours_kv, args.device, prompt_ids, args.gen_length, args.steps, 
        args.block_length, args.num_warmup, args.num_runs, is_hf=False, is_kv=True
    )
    ours_kv_gen_mean, ours_kv_gen_med, ours_kv_gen_p95 = get_stats(ours_kv_gen_lats)
    ours_kv_tok_per_sec = args.gen_length / ours_kv_gen_mean
    ours_kv_ms_per_step = (ours_kv_gen_mean * 1000) / args.steps
    print("Done.\n")

    print("Unloading KV custom model to free memory for Hugging Face model...")
    del ours_kv
    gc.collect()
    if "cuda" in args.device:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device=args.device)
    print("Done.\n")

    # 4. Load and Benchmark HF Model
    print("  Loading Hugging Face model implementation...")
    hf, hf_load_time = load_hf(args.weight_dir, args.device)
    print(f"  HF loaded in {hf_load_time:.2f} seconds.")
    
    # Measure memory after loading HF
    hf_peak_mem = 0
    if "cuda" in args.device:
        hf_peak_mem = torch.cuda.max_memory_allocated(device=args.device) / (1024 ** 3)
        torch.cuda.reset_peak_memory_stats(device=args.device)

    print("Benchmarking single forward pass latency for HF model...")
    for seq_len in seq_lengths:
        print(f"  Running sequence length {seq_len}...")
        hf_lats = benchmark_forward(hf, args.device, seq_len, args.num_warmup, args.num_runs, is_hf=True)
        hf_mean, hf_med, hf_p95 = get_stats(hf_lats)
        forward_results[seq_len]["hf"] = (hf_mean * 1000, hf_med * 1000, hf_p95 * 1000)
    print("Done.\n")

    print("Benchmarking full generation (diffusion decode) for HF model...")
    hf_gen_lats = benchmark_generation(
        hf, args.device, prompt_ids, args.gen_length, args.steps, 
        args.block_length, args.num_warmup, args.num_runs, is_hf=True
    )
    hf_gen_mean, hf_gen_med, hf_gen_p95 = get_stats(hf_gen_lats)
    hf_tok_per_sec = args.gen_length / hf_gen_mean
    hf_ms_per_step = (hf_gen_mean * 1000) / args.steps
    print("Done.\n")

    # Output Results Table
    print("=" * 110)
    print("                           BENCHMARK RESULTS COMPARISON")
    print("=" * 110)
    print(f"| Metric / Test Case                | Custom Implementation | Custom w/ KV Cache | Hugging Face Ref  | Speedup (KV vs HF)|")
    print(f"|-----------------------------------|-----------------------|--------------------|-------------------|-------------------|")
    print(f"| Model Loading Time (sec)          | {ours_load_time:21.2f} | {ours_kv_load_time:18.2f} | {hf_load_time:17.2f} | {hf_load_time/ours_kv_load_time:16.2f}x |")
    if "cuda" in args.device:
        print(f"| Peak GPU Memory Usage (GB)        | {ours_peak_mem:21.2f} | {ours_kv_peak_mem:18.2f} | {hf_peak_mem:17.2f} | {hf_peak_mem/ours_kv_peak_mem:16.2f}x |")
    
    print(f"|-----------------------------------|-----------------------|--------------------|-------------------|-------------------|")
    
    # Forward passes
    for seq_len in seq_lengths:
        ours_mean, ours_med, ours_p95 = forward_results[seq_len]["ours"]
        ours_kv_mean, ours_kv_med, ours_kv_p95 = forward_results[seq_len]["ours_kv"]
        hf_mean, hf_med, hf_p95 = forward_results[seq_len]["hf"]
        print(f"| Fwd Pass Latency ({seq_len:4d} tokens)   |                       |                    |                   |                   |")
        print(f"|   - Mean (ms)                     | {ours_mean:21.2f} | {ours_kv_mean:18.2f} | {hf_mean:17.2f} | {hf_mean/ours_kv_mean:16.2f}x |")
        print(f"|   - Median / p50 (ms)             | {ours_med:21.2f} | {ours_kv_med:18.2f} | {hf_med:17.2f} | {hf_med/ours_kv_med:16.2f}x |")
        print(f"|   - p95 (ms)                      | {ours_p95:21.2f} | {ours_kv_p95:18.2f} | {hf_p95:17.2f} | {hf_p95/ours_kv_p95:16.2f}x |")
        print(f"|-----------------------------------|-----------------------|--------------------|-------------------|-------------------|")

    # Generation metrics
    print(f"| Generation Benchmark              |                       |                    |                   |                   |")
    print(f"|   - Total Time (sec)              | {ours_gen_mean:21.2f} | {ours_kv_gen_mean:18.2f} | {hf_gen_mean:17.2f} | {hf_gen_mean/ours_kv_gen_mean:16.2f}x |")
    print(f"|   - Time per Step (ms)            | {ours_ms_per_step:21.2f} | {ours_kv_ms_per_step:18.2f} | {hf_ms_per_step:17.2f} | {hf_ms_per_step/ours_kv_ms_per_step:16.2f}x |")
    print(f"|   - Throughput (tokens/sec)       | {ours_tok_per_sec:21.2f} | {ours_kv_tok_per_sec:18.2f} | {hf_tok_per_sec:17.2f} | {ours_kv_tok_per_sec/hf_tok_per_sec:16.2f}x |")
    print("=" * 110)
    print("\nNote: Speedup > 1.0x indicates that the Custom KV implementation is faster than HF.")


if __name__ == "__main__":
    main()
