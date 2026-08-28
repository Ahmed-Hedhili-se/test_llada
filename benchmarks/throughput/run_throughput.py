"""Throughput benchmark for LLaDA-MoE inference server.

Sends concurrent requests and measures tokens/second.

Usage:
    python3 -m benchmarks.throughput.run_throughput --base-url http://localhost:8000
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


async def run_benchmark(base_urls: list[str], concurrency: int, max_tokens: int,
                         steps: int, block_length: int, n_requests: int,
                         fixed_prompt: bool = False, timeout: float = 1800.0):
    """
    base_urls: one or more independent backend URLs. Prompts are round-robined
    across them and a SINGLE semaphore of size `concurrency` is shared across
    all of them -- so "N backends at --concurrency C" and "1 backend at
    --concurrency C" both offer the same total in-flight load, making them a
    fair apples-to-apples comparison (e.g. one TP+EP instance vs. len(base_urls)
    independent data-parallel replicas, each replica a separate single-GPU
    server process -- see benchmarks/throughput/README or the DP launch commands
    in the project history for how to start replicas on distinct ports/GPUs).

    fixed_prompt: send PROMPTS[0] for every request instead of cycling
    through PROMPTS. dminfr/serving/server.py's request batching only groups requests
    with an IDENTICAL tokenized prompt length (see _PendingRequest.key) --
    varied prompts would mostly miss each other and never batch, so this
    flag is needed to actually exercise batching in this benchmark.
    """
    prompts = (
        [PROMPTS[0]] * n_requests if fixed_prompt
        else (PROMPTS * ((n_requests // len(PROMPTS)) + 1))[:n_requests]
    )
    connector = aiohttp.TCPConnector(limit=concurrency)

    async with aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(total=timeout, sock_connect=15)) as session:
        # Warm up every backend once, not just the first.
        print(f"Warming up {len(base_urls)} backend(s)...", flush=True)
        await asyncio.gather(*[
            send_request(session, url, prompts[0], max_tokens, steps, block_length)
            for url in base_urls
        ])

        print(f"Running {n_requests} requests across {len(base_urls)} backend(s) "
              f"(total concurrency={concurrency})...", flush=True)
        t_start = time.perf_counter()

        sem = asyncio.Semaphore(concurrency)
        async def bounded(p, url):
            async with sem:
                return await send_request(session, url, p, max_tokens, steps, block_length)

        # Round-robin assignment across backends.
        results = await asyncio.gather(*[
            bounded(p, base_urls[i % len(base_urls)]) for i, p in enumerate(prompts)
        ])
        t_total = time.perf_counter() - t_start

    ok = [r for r in results if r["ok"]]
    total_out = sum(r["completion_tokens"] for r in ok)
    total_in  = sum(r["prompt_tokens"] for r in ok)
    latencies = [r["elapsed"] for r in ok]

    print(f"\n{'='*60}")
    print(f"  LLaDA-MoE Throughput Benchmark")
    print(f"{'='*60}")
    print(f"  Backends:          {len(base_urls)} ({', '.join(base_urls)})")
    print(f"  Requests:          {len(ok)}/{n_requests} succeeded")
    print(f"  Concurrency:       {concurrency} (total, shared across backends)")
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
    ap.add_argument("--base-url", default="http://localhost:8000",
                    help="Single backend URL (default mode: one TP+EP instance).")
    ap.add_argument(
        "--base-urls", default=None,
        help="Comma-separated list of independent backend URLs to round-robin "
             "across, e.g. for comparing N data-parallel replicas against a "
             "single TP+EP instance at the same total --concurrency "
             "(http://localhost:8000,http://localhost:8001). Overrides --base-url.",
    )
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--block-length", type=int, default=32)
    ap.add_argument("--n-requests", type=int, default=16)
    ap.add_argument("--timeout", type=float, default=1800.0,
                    help="Per-request ceiling in seconds. aiohttp defaults to 300, "
                         "which silently aborts healthy requests against a slow or "
                         "serialized backend -- the src/ baseline takes ~30s per "
                         "request, so 32 queued requests blow past it. Raise for "
                         "long generations.")
    ap.add_argument(
        "--fixed-prompt", action="store_true",
        help="Send the same prompt for every request instead of cycling "
             "through PROMPTS. Needed to exercise server-side batching, "
             "which only groups requests with an identical tokenized "
             "prompt length (see dminfr/serving/server.py's _PendingRequest.key).",
    )
    args = ap.parse_args()

    base_urls = args.base_urls.split(",") if args.base_urls else [args.base_url]

    asyncio.run(run_benchmark(
        base_urls, args.concurrency, args.max_tokens,
        args.steps, args.block_length, args.n_requests,
        fixed_prompt=args.fixed_prompt,
        timeout=args.timeout,
    ))


if __name__ == "__main__":
    main()
