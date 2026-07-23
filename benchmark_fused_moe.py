"""
Benchmark script to measure token/s speedup from Fused MoE (vLLM Triton kernel)
vs Unfused MoE (PyTorch expert loop) in LLaDA-MoE.

Usage:
    python benchmark_fused_moe.py
    python benchmark_fused_moe.py --gen-len 128 --steps 64 --block-len 32
    python benchmark_fused_moe.py --config small --runs 5
    python benchmark_fused_moe.py --layer-only
"""

import argparse
import time
import sys
from pathlib import Path

import torch
import torch.nn as nn

# Ensure local workspace paths can be imported
sys.path.insert(0, str(Path(__file__).parent))

from model_update.model import LLaDAMoEKV, MoEBlock, VLLMFusedMoEBlock, FULL_CFG, SMALL_CFG
from model_update.generate import generate_cached


def init_moe_weights(module: nn.Module):
    """Initialize random weights for benchmark stability."""
    with torch.no_grad():
        for p in module.parameters():
            if p.dim() > 1:
                nn.init.normal_(p, std=0.02)
            else:
                nn.init.zeros_(p)


def benchmark_layer_moe(cfg, device, dtype, batch_size, seq_len, warmup=5, runs=20):
    """Benchmark isolated MoE block: MoEBlock (unfused) vs VLLMFusedMoEBlock (fused)."""
    print("\n" + "=" * 60)
    print(f" 1. Isolated MoE Layer Benchmark (Batch={batch_size}, Tokens={seq_len})")
    print("=" * 60)

    # 1. Unfused MoE
    unfused_block = MoEBlock(cfg).to(device=device, dtype=dtype).eval()
    init_moe_weights(unfused_block)

    # 2. Fused MoE
    try:
        fused_block = VLLMFusedMoEBlock(cfg).to(device=device, dtype=dtype).eval()
        init_moe_weights(fused_block)
        # Load weights from unfused to ensure matching shapes & initialized values
        fused_block.load_from_unfused(unfused_block.experts)
        fused_available = True
    except Exception as e:
        print(f"[WARN] Fused MoE initialization failed: {e}")
        fused_available = False

    x = torch.randn(batch_size, seq_len, cfg.H, device=device, dtype=dtype)

    # Warmup Unfused
    for _ in range(warmup):
        with torch.no_grad():
            _ = unfused_block(x)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Measure Unfused
    start_evt = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
    end_evt = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None

    if device.type == "cuda":
        start_evt.record()
        for _ in range(runs):
            with torch.no_grad():
                _ = unfused_block(x)
        end_evt.record()
        torch.cuda.synchronize()
        unfused_ms = start_evt.elapsed_time(end_evt) / runs
    else:
        t0 = time.perf_counter()
        for _ in range(runs):
            with torch.no_grad():
                _ = unfused_block(x)
        t1 = time.perf_counter()
        unfused_ms = ((t1 - t0) / runs) * 1000.0

    unfused_tok_per_sec = (batch_size * seq_len) / (unfused_ms / 1000.0)

    print(f"  Unfused MoE Block latency: {unfused_ms:.3f} ms | Throughput: {unfused_tok_per_sec:,.1f} tok/s")

    if fused_available:
        # Warmup Fused
        for _ in range(warmup):
            with torch.no_grad():
                _ = fused_block(x)
        if device.type == "cuda":
            torch.cuda.synchronize()

        # Measure Fused
        if device.type == "cuda":
            start_evt.record()
            for _ in range(runs):
                with torch.no_grad():
                    _ = fused_block(x)
            end_evt.record()
            torch.cuda.synchronize()
            fused_ms = start_evt.elapsed_time(end_evt) / runs
        else:
            t0 = time.perf_counter()
            for _ in range(runs):
                with torch.no_grad():
                    _ = fused_block(x)
            t1 = time.perf_counter()
            fused_ms = ((t1 - t0) / runs) * 1000.0

        fused_tok_per_sec = (batch_size * seq_len) / (fused_ms / 1000.0)
        speedup = unfused_ms / fused_ms if fused_ms > 0 else 0
        pct_increase = ((fused_tok_per_sec - unfused_tok_per_sec) / unfused_tok_per_sec) * 100.0

        print(f"  Fused MoE Block latency:   {fused_ms:.3f} ms | Throughput: {fused_tok_per_sec:,.1f} tok/s")
        print(f"  >>> Isolated MoE Speedup: {speedup:.2f}x ({pct_increase:+.1f}% throughput increase)")
    else:
        print("  [SKIP] Fused MoE could not be benchmarked on this environment.")


def benchmark_end_to_end_generation(
    cfg,
    device,
    dtype,
    prompt_len=128,
    gen_len=128,
    steps=64,
    block_len=32,
    warmup=2,
    runs=5,
):
    """Benchmark end-to-end diffusion generation (tokens/sec) Unfused vs Fused."""
    print("\n" + "=" * 60)
    print(f" 2. End-to-End Generation Benchmark")
    print(f"    (Prompt={prompt_len}, GenLen={gen_len}, Steps={steps}, BlockLen={block_len})")
    print("=" * 60)

    prompt_ids = torch.randint(0, cfg.VS, (1, prompt_len), device=device, dtype=torch.long)

    # 1. Unfused Full Model
    print("\n[+] Instantiating Unfused Model (use_fused_moe=False)...")
    model_unfused = LLaDAMoEKV(cfg=cfg, use_fused_moe=False).to(device=device, dtype=dtype).eval()
    init_moe_weights(model_unfused)

    # Warmup Unfused
    print(f"    Warming up ({warmup} runs)...")
    for _ in range(warmup):
        with torch.no_grad():
            _ = generate_cached(
                model_unfused,
                prompt_ids,
                gen_length=gen_len,
                steps=steps,
                block_length=block_len,
            )
    if device.type == "cuda":
        torch.cuda.synchronize()

    print(f"    Running benchmark ({runs} runs)...")
    unfused_times = []
    for r in range(runs):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = generate_cached(
                model_unfused,
                prompt_ids,
                gen_length=gen_len,
                steps=steps,
                block_length=block_len,
            )
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        unfused_times.append(t1 - t0)

    avg_unfused_sec = sum(unfused_times) / len(unfused_times)
    unfused_tok_per_sec = gen_len / avg_unfused_sec
    print(f"  --> Unfused Model: {avg_unfused_sec:.3f} s total ({gen_len} tokens) | {unfused_tok_per_sec:.2f} token/s")

    del model_unfused
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # 2. Fused Full Model
    print("\n[+] Instantiating Fused Model (use_fused_moe=True)...")
    try:
        model_fused = LLaDAMoEKV(cfg=cfg, use_fused_moe=True).to(device=device, dtype=dtype).eval()
        init_moe_weights(model_fused)

        print(f"    Warming up ({warmup} runs)...")
        for _ in range(warmup):
            with torch.no_grad():
                _ = generate_cached(
                    model_fused,
                    prompt_ids,
                    gen_length=gen_len,
                    steps=steps,
                    block_length=block_len,
                )
        if device.type == "cuda":
            torch.cuda.synchronize()

        print(f"    Running benchmark ({runs} runs)...")
        fused_times = []
        for r in range(runs):
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = generate_cached(
                    model_fused,
                    prompt_ids,
                    gen_length=gen_len,
                    steps=steps,
                    block_length=block_len,
                )
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            fused_times.append(t1 - t0)

        avg_fused_sec = sum(fused_times) / len(fused_times)
        fused_tok_per_sec = gen_len / avg_fused_sec
        speedup = avg_unfused_sec / avg_fused_sec
        pct_increase = ((fused_tok_per_sec - unfused_tok_per_sec) / unfused_tok_per_sec) * 100.0

        print(f"  --> Fused Model:   {avg_fused_sec:.3f} s total ({gen_len} tokens) | {fused_tok_per_sec:.2f} token/s")

        print("\n" + "=" * 60)
        print(" SUMMARY RESULTS:")
        print(f"  - Unfused MoE Throughput : {unfused_tok_per_sec:.2f} token/s")
        print(f"  - Fused MoE Throughput   : {fused_tok_per_sec:.2f} token/s")
        print(f"  - Speedup Factor         : {speedup:.2f}x")
        print(f"  - Token/s Increase       : {pct_increase:+.2f}%")
        print("=" * 60)

    except Exception as e:
        print(f"\n[ERROR] Fused MoE Generation failed: {e}")
        print("Ensure vLLM / Triton CUDA dependencies are correctly installed.")


def main():
    parser = argparse.ArgumentParser(description="Benchmark Fused MoE vs Unfused MoE in LLaDA")
    parser.add_argument("--config", type=str, choices=["full", "small"], default="full", help="Model configuration scale")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--dtype", type=str, choices=["bfloat16", "float16", "float32"], default="bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float32")
    parser.add_argument("--prompt-len", type=int, default=128, help="Prompt sequence length")
    parser.add_argument("--gen-len", type=int, default=128, help="Generated sequence length")
    parser.add_argument("--steps", type=int, default=64, help="Total diffusion steps")
    parser.add_argument("--block-len", type=int, default=32, help="Block length for cached generation")
    parser.add_argument("--runs", type=int, default=5, help="Number of benchmark runs")
    parser.add_argument("--warmup", type=int, default=2, help="Number of warmup runs")
    parser.add_argument("--layer-only", action="store_true", help="Only run isolated layer benchmark")
    args = parser.parse_args()

    cfg = FULL_CFG if args.config == "full" else SMALL_CFG
    device = torch.device(args.device)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    print("============================================================")
    print(" LLaDA-MoE Fused MoE Speedup Benchmark")
    print("============================================================")
    print(f" Device     : {device}")
    print(f" Dtype      : {dtype}")
    print(f" Config     : {args.config.upper()} (Hidden={cfg.H}, Layers={cfg.NL}, Experts={cfg.NE}, TopK={cfg.TOPK})")

    # 1. Benchmark isolated MoE Layer
    benchmark_layer_moe(cfg, device, dtype, batch_size=1, seq_len=args.block_len, warmup=args.warmup * 2, runs=args.runs * 4)

    # 2. Benchmark Full Generation
    if not args.layer_only:
        benchmark_end_to_end_generation(
            cfg=cfg,
            device=device,
            dtype=dtype,
            prompt_len=args.prompt_len,
            gen_len=args.gen_len,
            steps=args.steps,
            block_len=args.block_len,
            warmup=args.warmup,
            runs=args.runs,
        )


if __name__ == "__main__":
    main()
