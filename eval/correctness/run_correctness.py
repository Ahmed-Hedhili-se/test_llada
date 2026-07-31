"""
Correctness evaluation for LLaDA-MoE-7B-A1B-Instruct.

WHY NOT lm-eval FOR MMLU?
--------------------------
MMLU in lm-evaluation-harness uses *loglikelihood* scoring: it measures the
log-probability of each answer choice ("A", "B", "C", "D") and picks the
highest.  This requires direct access to model logits.

The `local-chat-completions` backend of lm-eval does NOT support loglikelihood
(it only receives generated text back from the API), so it raises:

    NotImplementedError: Loglikelihood is not supported for chat completions.

SOLUTION: We bypass lm-eval entirely.  This script:
  1. Loads MMLU (or ARC-Challenge) directly from HuggingFace datasets.
  2. Formats each question as a chat prompt that asks for a single letter answer.
  3. Sends it to the server via the OpenAI-compatible chat API.
  4. Parses the first A/B/C/D token from the model response.
  5. Computes accuracy against the ground-truth labels.

This approach works correctly for masked diffusion LMs like LLaDA-MoE because:
  - The model only needs to generate a short answer ("A", "B", "C", or "D").
  - No logit access required.
  - No chain-of-thought required.

SUPPORTED TASKS
---------------
  mmlu            : all 57 MMLU subjects (test split, ~14 000 questions)
  mmlu_<subject>  : single MMLU subject (e.g. mmlu_anatomy)
  arc_challenge   : ARC-Challenge (1172 test questions)
  arc_easy        : ARC-Easy (2376 test questions)

HUMANEVAL / MBPP
----------------
Still not supported here — use the dedicated script:
    python eval/correctness/diagnose_humaneval.py

Usage:
    # Run 200 MMLU questions (default)
    python -m eval.correctness.run_correctness \\
        --output results/baseline_mmlu.json

    # Full MMLU
    python -m eval.correctness.run_correctness --limit 0 \\
        --output results/baseline_mmlu_full.json

    # ARC-Challenge
    python -m eval.correctness.run_correctness --task arc_challenge \\
        --output results/arc.json

    # Compare to a saved baseline
    python -m eval.correctness.run_correctness \\
        --baseline results/baseline_mmlu.json \\
        --output results/optimised_mmlu.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import requests

# ── Defaults ────────────────────────────────────────────────────────────────────
DEFAULT_TASK    = "mmlu"
DEFAULT_LIMIT   = 200
DEFAULT_URL     = "http://localhost:8000"
DEFAULT_TIMEOUT = 120   # seconds per request

# Tasks that MUST NOT run through this script
_CODE_TASKS = {"humaneval", "mbpp", "mbpp_plus", "humaneval_plus"}

# Answer choices
CHOICES = ["A", "B", "C", "D"]

SYSTEM_PROMPT = (
    "You are a helpful, precise assistant. "
    "Answer multiple-choice questions by responding with ONLY the letter "
    "of the correct answer (A, B, C, or D). "
    "Do not explain your answer."
)


# ── Dataset loading ──────────────────────────────────────────────────────────────

def _load_mmlu(subject: Optional[str], limit: int) -> list[dict]:
    """
    Load MMLU test questions.
    Each returned dict: {question, choices:[A,B,C,D], answer_idx: int, subject}
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("[ERROR] 'datasets' package is required.  Run: pip install datasets")
        sys.exit(1)

    if subject:
        ds = load_dataset("cais/mmlu", subject, trust_remote_code=True)
    else:
        ds = load_dataset("cais/mmlu", "all", trust_remote_code=True)

    split = ds["test"]
    items = []
    for row in split:
        items.append({
            "question":   row["question"],
            "choices":    row["choices"],   # list of 4 strings
            "answer_idx": int(row["answer"]),  # 0-indexed
            "subject":    row.get("subject", subject or "mmlu"),
        })

    if limit and len(items) > limit:
        random.shuffle(items)
        items = items[:limit]
    return items


def _load_arc(variant: str, limit: int) -> list[dict]:
    """
    Load ARC-Challenge or ARC-Easy test questions.
    Each returned dict: {question, choices:[...], answer_idx: int}
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("[ERROR] 'datasets' package is required.  Run: pip install datasets")
        sys.exit(1)

    name = "ARC-Challenge" if variant == "arc_challenge" else "ARC-Easy"
    ds = load_dataset("allenai/ai2_arc", name, trust_remote_code=True)
    split = ds["test"]

    items = []
    for row in split:
        labels  = row["choices"]["label"]   # ["A","B","C","D"] or ["1","2","3","4"]
        texts   = row["choices"]["text"]
        answer  = row["answerKey"]          # "A"/"B"/"C"/"D" or "1"/"2"...

        # Normalise labels to A/B/C/D
        label_map = {l: i for i, l in enumerate(labels)}
        if answer not in label_map:
            continue
        answer_idx = label_map[answer]
        # Re-map labels to A/B/C/D
        mapped_choices = texts[:4]          # at most 4 options
        items.append({
            "question":   row["question"],
            "choices":    mapped_choices,
            "answer_idx": answer_idx,
            "subject":    variant,
        })

    if limit and len(items) > limit:
        random.shuffle(items)
        items = items[:limit]
    return items


def load_dataset_for_task(task: str, limit: int) -> list[dict]:
    if task in _CODE_TASKS:
        print(
            f"\n[ERROR] Task '{task}' requires code execution and cannot be run "
            f"through the chat API.\n"
            f"Use: python eval/correctness/diagnose_humaneval.py\n"
        )
        sys.exit(1)

    if task == "mmlu":
        return _load_mmlu(None, limit)
    elif task.startswith("mmlu_"):
        subject = task[len("mmlu_"):]
        return _load_mmlu(subject, limit)
    elif task in ("arc_challenge", "arc_easy"):
        return _load_arc(task, limit)
    else:
        print(f"[ERROR] Unsupported task: '{task}'")
        print("Supported: mmlu, mmlu_<subject>, arc_challenge, arc_easy")
        sys.exit(1)


# ── Prompt construction ──────────────────────────────────────────────────────────

def build_prompt(item: dict) -> str:
    q = item["question"].strip()
    lines = [q, ""]
    for letter, text in zip(CHOICES, item["choices"]):
        lines.append(f"{letter}) {text}")
    lines.append("\nAnswer with only the letter A, B, C, or D.")
    return "\n".join(lines)


# ── Server call ──────────────────────────────────────────────────────────────────

def call_server(base_url: str, prompt: str, timeout: int) -> str:
    payload = {
        "model":       "inclusionAI/LLaDA-MoE-7B-A1B-Instruct",
        "messages":    [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        "temperature": 0.0,
        "top_p":       1.0,
        "max_tokens":  16,
        "steps":       32,
    }
    resp = requests.post(
        f"{base_url}/v1/chat/completions",
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def parse_answer(text: str) -> Optional[str]:
    """Extract the first A/B/C/D letter from model output."""
    text = text.strip()
    # Exact single letter
    if text.upper() in CHOICES:
        return text.upper()
    # First letter that is A/B/C/D (ignore punctuation / wrapping)
    m = re.search(r"\b([ABCD])\b", text.upper())
    if m:
        return m.group(1)
    # Last resort: first A/B/C/D character anywhere
    for ch in text.upper():
        if ch in CHOICES:
            return ch
    return None


# ── Evaluation loop ───────────────────────────────────────────────────────────────

def evaluate(
    items: list[dict],
    base_url: str,
    num_concurrent: int,
    timeout: int,
) -> dict:
    correct = 0
    total   = len(items)
    errors  = 0
    per_subject: dict[str, list[bool]] = {}

    def _eval_one(idx: int, item: dict):
        prompt    = build_prompt(item)
        try:
            raw       = call_server(base_url, prompt, timeout)
            predicted = parse_answer(raw)
        except Exception as e:
            return idx, item, None, str(e)
        expected  = CHOICES[item["answer_idx"]]
        is_correct = (predicted == expected)
        return idx, item, is_correct, raw

    futures = {}
    results_list = [None] * total
    with ThreadPoolExecutor(max_workers=num_concurrent) as ex:
        for i, item in enumerate(items):
            futures[ex.submit(_eval_one, i, item)] = i

        done = 0
        for future in as_completed(futures):
            idx, item, is_correct, raw = future.result()
            results_list[idx] = (item, is_correct, raw)
            done += 1

            subj = item.get("subject", "unknown")
            per_subject.setdefault(subj, [])

            if is_correct is None:
                errors += 1
                per_subject[subj].append(False)
                print(f"  [{done:4d}/{total}] ERROR: {raw}", flush=True)
            else:
                if is_correct:
                    correct += 1
                per_subject[subj].append(is_correct)
                mark = "✅" if is_correct else "❌"
                exp  = CHOICES[item["answer_idx"]]
                pred = parse_answer(raw) or "?"
                print(
                    f"  [{done:4d}/{total}] {mark}  "
                    f"Expected={exp} Got={pred}  "
                    f"({subj})",
                    flush=True,
                )

    accuracy = correct / total if total else 0.0
    per_subject_acc = {
        s: sum(v) / len(v) for s, v in per_subject.items() if v
    }

    return {
        "accuracy":         accuracy,
        "correct":          correct,
        "total":            total,
        "errors":           errors,
        "per_subject":      per_subject_acc,
    }


# ── Printing ─────────────────────────────────────────────────────────────────────

def print_results(stats: dict, task: str, label: str = "Results"):
    print(f"\n{'='*60}")
    print(f"  {label} — {task}")
    print(f"{'='*60}")
    print(f"  Accuracy : {stats['accuracy']:.4f}  ({stats['accuracy']*100:.1f}%)")
    print(f"  Correct  : {stats['correct']} / {stats['total']}")
    if stats['errors']:
        print(f"  Errors   : {stats['errors']}  (server timeouts / parse failures)")
    # Show per-subject breakdown if multiple subjects
    if len(stats.get("per_subject", {})) > 1:
        print(f"\n  Per-subject breakdown:")
        for subj, acc in sorted(stats["per_subject"].items(), key=lambda x: -x[1]):
            print(f"    {subj:<40} {acc:.3f}  ({acc*100:.1f}%)")
    print(f"{'='*60}\n")


def print_comparison(stats: dict, baseline_path: str, task: str, label: str = "Current"):
    print_results(stats, task, label)
    if not (baseline_path and os.path.exists(baseline_path)):
        return
    with open(baseline_path) as f:
        baseline = json.load(f)
    bl_acc = baseline.get("accuracy")
    if bl_acc is None:
        return
    acc  = stats["accuracy"]
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


def save_summary(stats: dict, task: str, output_path: str, seed: int, config_name: str = ""):
    summary = {
        "task":        task,
        "accuracy":    stats["accuracy"],
        "correct":     stats["correct"],
        "total":       stats["total"],
        "errors":      stats["errors"],
        "config":      config_name,
        "seed":        seed,
        "timestamp":   time.strftime("%Y-%m-%d %H:%M:%S"),
        "per_subject": stats.get("per_subject", {}),
    }
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {output_path}")


# ── Entry point ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=(
            "Direct correctness evaluation for LLaDA-MoE-7B-A1B-Instruct. "
            "Loads MMLU/ARC from HuggingFace and sends questions to the chat API. "
            "No lm-eval loglikelihood needed."
        )
    )
    ap.add_argument(
        "--task", default=DEFAULT_TASK,
        help="Task: mmlu, mmlu_<subject>, arc_challenge, arc_easy. (default: mmlu)",
    )
    ap.add_argument("--base-url",       default=DEFAULT_URL)
    ap.add_argument("--limit",          type=int, default=DEFAULT_LIMIT,
                    help="Max questions to evaluate (0 = full dataset).")
    ap.add_argument("--num-concurrent", type=int, default=1,
                    help="Concurrent requests to the server.")
    ap.add_argument("--timeout",        type=int, default=DEFAULT_TIMEOUT,
                    help="Request timeout in seconds.")
    ap.add_argument("--output",         default=None,
                    help="Path to save the summary JSON.")
    ap.add_argument("--baseline",       default=None,
                    help="Path to a previous summary JSON for delta comparison.")
    ap.add_argument("--seed",           type=int, default=None)
    ap.add_argument("--config-name",    default="",
                    help="Label stored in the summary JSON.")
    args = ap.parse_args()

    if args.task in _CODE_TASKS:
        print(
            f"\n[ERROR] Task '{args.task}' requires code execution.\n"
            f"Use: python eval/correctness/diagnose_humaneval.py\n"
        )
        sys.exit(1)

    seed = args.seed if args.seed is not None else random.randint(0, 999_999)
    random.seed(seed)
    limit = args.limit if args.limit > 0 else 0  # 0 = no limit

    print(f"\nTarget     : {args.base_url}")
    print(f"Task       : {args.task}")
    print(f"Limit      : {limit if limit else 'full dataset'}")
    print(f"Concurrent : {args.num_concurrent}")
    print(f"Seed       : {seed}\n")

    print("Loading dataset...", flush=True)
    items = load_dataset_for_task(args.task, limit)
    print(f"Loaded {len(items)} questions.\n")

    t0    = time.time()
    stats = evaluate(items, args.base_url, args.num_concurrent, args.timeout)
    elapsed = time.time() - t0

    print(f"\nTotal time : {elapsed:.1f}s  ({elapsed/max(len(items),1):.1f}s/question)")

    if args.baseline:
        print_comparison(stats, args.baseline, args.task, args.config_name or "Current")
    else:
        print_results(stats, args.task, "Results")

    if args.output:
        save_summary(stats, args.task, args.output, seed, args.config_name)


if __name__ == "__main__":
    main()