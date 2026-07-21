"""
Correctness evaluation using GSM8K-CoT via lm-evaluation-harness.

Runs 200 GSM8K chain-of-thought problems against an OpenAI-compatible
chat completions endpoint and reports exact-match accuracy.

NEW: Supports comparing multiple configurations against a saved baseline.

Usage:
    # 1. Run baseline and save results
    python3 -m eval.correctness.run_correctness \
        --base-url http://localhost:8000 \
        --output baseline_results.json

    # 2. Run with optimized config and compare to baseline
    python3 -m eval.correctness.run_correctness \
        --base-url http://localhost:8000 \
        --baseline baseline_results.json \
        --output optimized_results.json

    # 3. Run all configs in sequence (if server supports config switching)
    python3 -m eval.correctness.run_correctness \
        --base-url http://localhost:8000 \
        --compare-all \
        --output-dir results/comparison
"""

import argparse
import json
import os
import random
import subprocess
import sys
import time

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


def extract_accuracy(results: dict) -> float | None:
    """Extract the best accuracy metric from results."""
    if not results:
        return None
    task_results = results.get("results", {}).get(TASK, {})
    # Prefer flexible extract, fall back to strict
    flexible = task_results.get("exact_match,flexible-extract")
    strict = task_results.get("exact_match,strict-match")
    return flexible if flexible is not None else strict


def print_single_results(results: dict, label: str = "Results"):
    if not results:
        print("No results found.")
        return
    task_results = results.get("results", {}).get(TASK, {})
    flexible = task_results.get("exact_match,flexible-extract")
    strict = task_results.get("exact_match,strict-match")

    print(f"\n{'='*60}")
    print(f"  {label} - GSM8K-CoT ({LIMIT} problems)")
    print(f"{'='*60}")
    if flexible is not None:
        print(f"  Exact match (flexible): {flexible:.4f} ({flexible*100:.1f}%)")
    if strict is not None:
        print(f"  Exact match (strict):   {strict:.4f} ({strict*100:.1f}%)")
    print(f"{'='*60}\n")


def print_comparison(results: dict, baseline_path: str | None = None, label: str = "Optimized"):
    """Print results with comparison to baseline."""
    acc = extract_accuracy(results)
    if acc is None:
        print("No results to compare.")
        return

    print_single_results(results, label)

    if baseline_path and os.path.exists(baseline_path):
        with open(baseline_path) as f:
            baseline = json.load(f)
        bl_acc = baseline.get("accuracy")
        if bl_acc is not None:
            diff = acc - bl_acc
            print(f"  Baseline accuracy:      {bl_acc:.4f} ({bl_acc*100:.1f}%)")
            print(f"  {label} accuracy:       {acc:.4f} ({acc*100:.1f}%)")
            print(f"  Difference:             {diff:+.4f} ({diff*100:+.1f}%)")
            if abs(diff) <= 0.01:
                print(f"  ✅ PASS: Within 1% of baseline")
            elif abs(diff) <= 0.02:
                print(f"  ⚠️  WARNING: Within 2% of baseline (acceptable)")
            else:
                print(f"  ❌ FAIL: Degradation >2% from baseline")
            print(f"{'='*60}\n")


def save_summary(results: dict, output_path: str, seed: int, config_name: str = ""):
    """Save a summary JSON with accuracy and metadata."""
    acc = extract_accuracy(results)
    task_results = results.get("results", {}).get(TASK, {}) if results else {}
    summary = {
        "task": TASK,
        "limit": LIMIT,
        "seed": seed,
        "accuracy": acc,
        "config": config_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "full_results": task_results,
    }
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {output_path}")


def main():
    ap = argparse.ArgumentParser(description="GSM8K-CoT correctness evaluation")
    ap.add_argument("--base-url", default="http://localhost:8000", help="OpenAI-compatible endpoint URL")
    ap.add_argument("--limit", type=int, default=LIMIT, help="Number of problems to evaluate")
    ap.add_argument("--num-concurrent", type=int, default=1, help="Concurrent requests")
    ap.add_argument("--output-dir", default="results/correctness", help="lm-eval output directory")
    ap.add_argument("--output", default=None, help="Path to save summary JSON")
    ap.add_argument("--baseline", default=None, help="Path to baseline summary JSON for comparison")
    ap.add_argument("--seed", type=int, default=None, help="Random seed (auto-generated if not set)")
    ap.add_argument("--config-name", default="", help="Name of config being tested (for summary)")
    ap.add_argument("--compare-all", action="store_true", help="Run all configs and compare (requires server support)")
    args = ap.parse_args()

    seed = args.seed if args.seed is not None else random.randint(0, 999999)
    print(f"Target: {args.base_url}")
    print(f"Task: {TASK} ({args.limit} problems)")
    print(f"Concurrent: {args.num_concurrent}")
    print(f"Seed: {seed}\n")

    if args.compare_all:
        # Run all configurations in sequence
        # NOTE: This requires the server to support switching configs
        # or you must restart the server between runs
        configs = [
            ("baseline", {"config": "baseline"}),
            ("cache_only", {"config": "cache_only"}),
            ("dynamic_experts", {"config": "dynamic_experts"}),
            ("fast_dense_cached", {"config": "fast_dense_cached"}),
        ]
        all_results = {}
        for name, _ in configs:
            print(f"\n{'='*60}")
            print(f"  Testing config: {name}")
            print(f"{'='*60}")
            # TODO: Signal server to switch config if supported
            # Or document that server must be restarted with new config
            results = run_eval(args.base_url, f"{args.output_dir}/{name}", args.num_concurrent, args.limit, seed)
            acc = extract_accuracy(results)
            all_results[name] = acc
            print_single_results(results, name)

        # Print comparison table
        print(f"\n{'='*60}")
        print(f"  COMPARISON TABLE")
        print(f"{'='*60}")
        baseline_acc = all_results.get("baseline", 0)
        for name, acc in all_results.items():
            if acc is not None:
                diff = acc - baseline_acc if name != "baseline" else 0
                marker = "✅" if abs(diff) <= 0.01 else "⚠️" if abs(diff) <= 0.02 else "❌"
                print(f"  {marker} {name:<20} {acc:.4f} ({acc*100:.1f}%)  {diff:+.4f}")
        print(f"{'='*60}\n")
    else:
        # Single run
        results = run_eval(args.base_url, args.output_dir, args.num_concurrent, args.limit, seed)

        if args.baseline:
            print_comparison(results, args.baseline, args.config_name or "Current")
        else:
            print_single_results(results, "Results")

        if args.output:
            save_summary(results, args.output, seed, args.config_name)


if __name__ == "__main__":
    main()