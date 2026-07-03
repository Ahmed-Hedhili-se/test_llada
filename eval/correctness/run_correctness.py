"""Correctness evaluation using GSM8K-CoT via lm-evaluation-harness.

Runs 200 GSM8K chain-of-thought problems against an OpenAI-compatible
chat completions endpoint and reports exact-match accuracy.

Usage:
    python3 -m eval.correctness.run_correctness --base-url http://localhost:8000
"""

import argparse
import json
import os
import random
import subprocess
import sys

os.environ.setdefault("PYTHONUNBUFFERED", "1")
sys.stdout.reconfigure(line_buffering=True)

TASK  = "gsm8k_cot"
LIMIT = 200


def run_eval(base_url: str, output_dir: str, num_concurrent: int, limit: int, seed: int) -> dict:
    base_url = base_url.rstrip("/")
    model_args = (
        f"model=inclusionAI/LLaDA-MoE-7B-A1B-Instruct,"
        f"base_url={base_url}/v1/chat/completions,"
        f"num_concurrent={num_concurrent},"
        f"tokenizer_backend=huggingface,"
        f"timeout=600"
    )
    cmd = [
        sys.executable, "-u", "-m", "lm_eval",
        "--model", "local-chat-completions",
        "--model_args", model_args,
        "--tasks", TASK,
        "--limit", str(limit),
        "--apply_chat_template",
        "--gen_kwargs", "temperature=0,top_p=1.0",
        "--seed", str(seed),
        "--output_path", output_dir,
        "--log_samples",
    ]
    print(f"Running: {' '.join(cmd)}\n", flush=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    result = subprocess.run(cmd, capture_output=False, env=env)
    if result.returncode != 0:
        print(f"\nlm-eval exited with code {result.returncode}")
        sys.exit(1)

    for root, _, files in os.walk(output_dir):
        for fn in files:
            if fn == "results.json":
                with open(os.path.join(root, fn)) as f:
                    return json.load(f)
    return {}


def print_results(results: dict, baseline_path: str | None = None):
    if not results:
        print("No results found.")
        return
    task_results = results.get("results", {}).get(TASK, {})
    flexible = task_results.get("exact_match,flexible-extract")
    strict   = task_results.get("exact_match,strict-match")

    print(f"\n{'='*60}")
    print(f"  GSM8K-CoT Results ({LIMIT} problems)")
    print(f"{'='*60}")
    if flexible is not None:
        print(f"  Exact match (flexible): {flexible:.4f} ({flexible*100:.1f}%)")
    if strict is not None:
        print(f"  Exact match (strict):   {strict:.4f} ({strict*100:.1f}%)")

    if baseline_path and os.path.exists(baseline_path):
        with open(baseline_path) as f:
            bl = json.load(f)
        bl_acc = bl.get("accuracy")
        if bl_acc is not None:
            diff = (flexible or strict or 0) - bl_acc
            print(f"  Baseline accuracy:      {bl_acc:.4f} ({bl_acc*100:.1f}%)")
            print(f"  Difference:             {diff:+.4f} ({diff*100:+.1f}%)")
    print(f"{'='*60}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--limit", type=int, default=LIMIT)
    ap.add_argument("--num-concurrent", type=int, default=1)
    ap.add_argument("--output-dir", default="results/correctness")
    ap.add_argument("--output", default=None)
    ap.add_argument("--baseline", default="baseline/results/correctness_baseline.json")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    seed = args.seed if args.seed is not None else random.randint(0, 999999)
    print(f"Target: {args.base_url}")
    print(f"Task: {TASK} ({args.limit} problems)")
    print(f"Concurrent: {args.num_concurrent}")
    print(f"Seed: {seed}\n")

    results = run_eval(args.base_url, args.output_dir, args.num_concurrent, args.limit, seed)
    print_results(results, args.baseline)

    if args.output and results:
        task_results = results.get("results", {}).get(TASK, {})
        accuracy = next((v for k, v in task_results.items() if "exact_match" in k and "stderr" not in k), None)
        summary = {"task": TASK, "limit": args.limit, "seed": seed, "accuracy": accuracy, "full_results": task_results}
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved to {args.output}")


if __name__ == "__main__":
    main()
