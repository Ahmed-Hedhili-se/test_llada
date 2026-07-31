"""
Correctness evaluation for LLaDA-MoE-7B-A1B-Instruct via lm-evaluation-harness.

WHY NOT GSM8K?
--------------
LLaDA-MoE is a *masked diffusion* language model, not an autoregressive one.
GSM8K-CoT requires left-to-right chain-of-thought generation, which is not how
diffusion LMs work — the model fills in masked tokens globally, so multi-step
CoT is unreliable.  GSM8K-CoT is therefore NOT a valid benchmark here.

WHY NOT HUMANEVAL / MBPP VIA THIS SCRIPT?
------------------------------------------
HumanEval and MBPP require code *execution* (pass@k sandbox) which lm-eval
cannot do when the model is accessed through a remote chat API.  Use the
dedicated script for a manual pass@1 check instead:

    python eval/correctness/diagnose_humaneval.py

TASKS USED HERE
---------------
MMLU (default) — 57 subjects, multiple-choice, short answer.
This is one of the tasks evaluated in the official LLaDA-MoE paper and maps
well to the diffusion generation paradigm (the model picks the best token,
not a long reasoning chain).

Other suitable tasks (pass with --task):
  arc_challenge, arc_easy, hellaswag, winogrande, piqa, boolq
  Any mmlu_* sub-category (e.g. mmlu_abstract_algebra)

Usage:
    # 1. Run MMLU baseline and save results
    python -m eval.correctness.run_correctness \\
        --base-url http://localhost:8000 \\
        --output results/baseline_mmlu.json

    # 2. Compare optimised config to saved baseline
    python -m eval.correctness.run_correctness \\
        --base-url http://localhost:8000 \\
        --baseline results/baseline_mmlu.json \\
        --output results/optimised_mmlu.json

    # 3. Run a faster MMLU sub-category
    python -m eval.correctness.run_correctness \\
        --task mmlu_abstract_algebra \\
        --limit 50 \\
        --output results/algebra.json

    # 4. Run ARC-Challenge instead
    python -m eval.correctness.run_correctness \\
        --task arc_challenge \\
        --output results/arc.json
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

# ── Defaults ────────────────────────────────────────────────────────────────────
DEFAULT_TASK  = "mmlu"
DEFAULT_LIMIT = 200          # set to None to run the full dataset

# Tasks that MUST NOT be run through this script (require code execution sandbox)
_CODE_TASKS = {"humaneval", "mbpp", "mbpp_plus", "humaneval_plus"}


def _guard_task(task: str):
    if task in _CODE_TASKS:
        print(
            f"\n[ERROR] Task '{task}' requires code execution and cannot be "
            f"evaluated through the chat API.\n"
            f"Use the dedicated script for a manual pass@1 check:\n"
            f"  python eval/correctness/diagnose_humaneval.py\n"
        )
        sys.exit(1)


# ── Core eval runner ─────────────────────────────────────────────────────────────

def run_eval(
    task: str,
    base_url: str,
    output_dir: str,
    num_concurrent: int,
    limit: int | None,
    seed: int,
) -> dict:
    _guard_task(task)

    base_url = base_url.rstrip("/")
    model_args = (
        f"model=inclusionAI/LLaDA-MoE-7B-A1B-Instruct,"
        f"base_url={base_url}/v1/chat/completions,"
        f"num_concurrent={num_concurrent},"
        f"tokenizer_backend=huggingface,"
        f"timeout=300"
    )

    # Multiple-choice / short-answer: greedy decoding, short output
    gen_kwargs = "temperature=0,top_p=1.0,max_tokens=64,steps=64"

    cmd = [
        sys.executable, "-u", "-m", "lm_eval",
        "--model",        "local-chat-completions",
        "--model_args",   model_args,
        "--tasks",        task,
        "--apply_chat_template",
        "--gen_kwargs",   gen_kwargs,
        "--seed",         str(seed),
        "--output_path",  output_dir,
        "--log_samples",
    ]
    if limit is not None:
        cmd += ["--limit", str(limit)]

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


# ── Metric extraction ────────────────────────────────────────────────────────────

def extract_accuracy(results: dict, task: str) -> float | None:
    """Extract accuracy from lm-eval results, handling MMLU sub-task aggregation."""
    if not results:
        return None
    task_results = results.get("results", {})

    # Try exact task key first
    tr = task_results.get(task, {})
    if tr:
        for key in ("acc,none", "acc_norm,none", "exact_match,flexible-extract", "exact_match,strict-match"):
            if key in tr:
                return tr[key]

    # Aggregate across sub-tasks (e.g. "mmlu" umbrella key is missing, sub-results present)
    accs = []
    for v in task_results.values():
        if not isinstance(v, dict):
            continue
        for key in ("acc,none", "acc_norm,none", "exact_match,flexible-extract"):
            if key in v:
                accs.append(v[key])
                break
    return sum(accs) / len(accs) if accs else None


# ── Printing ─────────────────────────────────────────────────────────────────────

def print_single_results(results: dict, task: str, label: str = "Results"):
    if not results:
        print("No results found.")
        return
    acc = extract_accuracy(results, task)
    print(f"\n{'='*60}")
    print(f"  {label} — {task}")
    print(f"{'='*60}")
    if acc is not None:
        print(f"  Accuracy: {acc:.4f}  ({acc*100:.1f}%)")
    else:
        # Fallback: dump all numeric metrics
        for t, metrics in results.get("results", {}).items():
            if isinstance(metrics, dict):
                for k, v in metrics.items():
                    if isinstance(v, float):
                        print(f"  {t} / {k}: {v:.4f}")
    print(f"{'='*60}\n")


def print_comparison(results: dict, task: str, baseline_path: str | None, label: str = "Optimised"):
    acc = extract_accuracy(results, task)
    if acc is None:
        print("No results to compare.")
        return

    print_single_results(results, task, label)

    if baseline_path and os.path.exists(baseline_path):
        with open(baseline_path) as f:
            baseline = json.load(f)
        bl_acc = baseline.get("accuracy")
        if bl_acc is not None:
            diff = acc - bl_acc
            print(f"  Baseline accuracy : {bl_acc:.4f} ({bl_acc*100:.1f}%)")
            print(f"  {label} accuracy  : {acc:.4f} ({acc*100:.1f}%)")
            print(f"  Difference        : {diff:+.4f} ({diff*100:+.1f}%)")
            if abs(diff) <= 0.01:
                print("  ✅ PASS: within 1 % of baseline")
            elif abs(diff) <= 0.02:
                print("  ⚠️  WARNING: within 2 % of baseline (acceptable)")
            else:
                print("  ❌ FAIL: degradation > 2 % from baseline")
            print(f"{'='*60}\n")


def save_summary(results: dict, task: str, output_path: str, seed: int, config_name: str = ""):
    acc = extract_accuracy(results, task)
    task_results = results.get("results", {}) if results else {}
    summary = {
        "task":         task,
        "accuracy":     acc,
        "config":       config_name,
        "seed":         seed,
        "timestamp":    time.strftime("%Y-%m-%d %H:%M:%S"),
        "full_results": task_results,
    }
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {output_path}")


# ── Entry point ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=(
            "Correctness eval for LLaDA-MoE-7B-A1B-Instruct via lm-eval. "
            "Default task: MMLU.  "
            "HumanEval/MBPP are NOT supported — use diagnose_humaneval.py."
        )
    )
    ap.add_argument(
        "--task", default=DEFAULT_TASK,
        help=(
            "lm-eval task name.  Recommended: mmlu (default), arc_challenge, "
            "arc_easy, hellaswag, winogrande, piqa, boolq, mmlu_<subject>."
        ),
    )
    ap.add_argument("--base-url",       default="http://localhost:8000")
    ap.add_argument("--limit",          type=int, default=DEFAULT_LIMIT,
                    help="Max questions to evaluate (omit for full dataset).")
    ap.add_argument("--num-concurrent", type=int, default=1)
    ap.add_argument("--output-dir",     default="results/correctness")
    ap.add_argument("--output",         default=None,
                    help="Path to save a compact summary JSON.")
    ap.add_argument("--baseline",       default=None,
                    help="Path to a previous summary JSON for delta comparison.")
    ap.add_argument("--seed",           type=int, default=None)
    ap.add_argument("--config-name",    default="",
                    help="Label stored in the summary JSON (e.g. 'fast_dense').")
    args = ap.parse_args()

    _guard_task(args.task)

    seed  = args.seed if args.seed is not None else random.randint(0, 999_999)
    limit = args.limit

    print(f"\nTarget     : {args.base_url}")
    print(f"Task       : {args.task}")
    print(f"Limit      : {limit if limit else 'full dataset'}")
    print(f"Concurrent : {args.num_concurrent}")
    print(f"Seed       : {seed}\n")

    results = run_eval(
        task=args.task,
        base_url=args.base_url,
        output_dir=args.output_dir,
        num_concurrent=args.num_concurrent,
        limit=limit,
        seed=seed,
    )

    if args.baseline:
        print_comparison(results, args.task, args.baseline, args.config_name or "Current")
    else:
        print_single_results(results, args.task, "Results")

    if args.output:
        save_summary(results, args.task, args.output, seed, args.config_name)


if __name__ == "__main__":
    main()