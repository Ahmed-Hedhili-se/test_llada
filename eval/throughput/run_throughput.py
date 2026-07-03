"""Throughput benchmark for LLaDA-MoE inference server.

Sends concurrent requests and measures tokens/second.

Usage:
    python3 -m eval.throughput.run_throughput --base-url http://localhost:8000
"""

import argparse
import asyncio
import json
import os
import sys
import time

import aiohttp

os.environ.setdefault("PYTHONUNBUFFERED", "1")
sys.stdout.reconfigure(line_buffering=True)

PROMPTS = [
    "Explain the concept of entropy in thermodynamics.",
    "Write a Python function to compute the nth Fibonacci number efficiently.",
    "What are the main differences between supervised and unsupervised learning?",
    "Solve: If a train travels at 80 km/h for 2.5 hours, how far does it go?",
    "Describe the process of photosynthesis step by step.",
    "What is the time complexity of merge sort and why?",
    "Explain how the TCP handshake works.",
    "Convert 98.6°F to Celsius and explain the formula.",
]


async def send_request(session: aiohttp.ClientSession, base_url: str, prompt: str,
                        max_tokens: int, steps: int, block_length: int) -> dict:
    payload = {
        "model": "inclusionAI/LLaDA-MoE-7B-A1B-Instruct",
        "messages": [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "steps": steps,
        "block_length": block_length,
    }
    t0 = time.perf_counter()
    async with session.post(f"{base_url}/v1/chat/completions", json=payload) as resp:
        data = await resp.json()
    elapsed = time.perf_counter() - t0
    usage = data.get("usage", {})
    return {
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "elapsed": elapsed,
        "ok": resp.status == 200,
    }


async def run_benchmark(base_url: str, concurrency: int, max_tokens: int,
                         steps: int, block_length: int, n_requests: int):
    prompts = (PROMPTS * ((n_requests // len(PROMPTS)) + 1))[:n_requests]
    connector = aiohttp.TCPConnector(limit=concurrency)

    async with aiohttp.ClientSession(connector=connector) as session:
        # Warm up
        print("Warming up...", flush=True)
        await send_request(session, base_url, prompts[0], max_tokens, steps, block_length)

        print(f"Running {n_requests} requests (concurrency={concurrency})...", flush=True)
        t_start = time.perf_counter()

        sem = asyncio.Semaphore(concurrency)
        async def bounded(p):
            async with sem:
                return await send_request(session, base_url, p, max_tokens, steps, block_length)

        results = await asyncio.gather(*[bounded(p) for p in prompts])
        t_total = time.perf_counter() - t_start

    ok = [r for r in results if r["ok"]]
    total_out = sum(r["completion_tokens"] for r in ok)
    total_in  = sum(r["prompt_tokens"] for r in ok)
    latencies = [r["elapsed"] for r in ok]

    print(f"\n{'='*60}")
    print(f"  LLaDA-MoE Throughput Benchmark")
    print(f"{'='*60}")
    print(f"  Requests:          {len(ok)}/{n_requests} succeeded")
    print(f"  Concurrency:       {concurrency}")
    print(f"  max_tokens:        {max_tokens}  steps={steps}  block={block_length}")
    print(f"  Total wall time:   {t_total:.1f}s")
    print(f"  Prompt tokens:     {total_in}")
    print(f"  Output tokens:     {total_out}")
    print(f"  Output tok/s:      {total_out / t_total:.1f}")
    print(f"  Req/s:             {len(ok) / t_total:.2f}")
    if latencies:
        latencies.sort()
        print(f"  Latency p50:       {latencies[len(latencies)//2]:.2f}s")
        print(f"  Latency p95:       {latencies[int(len(latencies)*0.95)]:.2f}s")
        print(f"  Latency p99:       {latencies[int(len(latencies)*0.99)]:.2f}s")
    print(f"{'='*60}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--block-length", type=int, default=32)
    ap.add_argument("--n-requests", type=int, default=16)
    args = ap.parse_args()

    asyncio.run(run_benchmark(
        args.base_url, args.concurrency, args.max_tokens,
        args.steps, args.block_length, args.n_requests,
    ))


if __name__ == "__main__":
    main()
