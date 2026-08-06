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
  - No logit access required — only generated text.
  - By default it uses chain-of-thought prompting (model reasons, then ends
    with "Final Answer: <letter>"), since MMLU-Pro was specifically built to
    punish models forced to answer without reasoning — bare "answer with only
    the letter" prompting reliably scores well below published numbers on it.
    Pass --direct-answer to go back to the old short bare-letter mode.

SUPPORTED TASKS
---------------
  mmlu            : all 57 MMLU subjects (test split, ~14 000 questions)
  mmlu_<subject>  : single MMLU subject (e.g. mmlu_anatomy)
  mmlu_pro        : MMLU-Pro (all subjects)
  mmlu_pro_<subj> : single MMLU-Pro subject
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

# Answer choices (up to J for MMLU-Pro)
CHOICES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

SYSTEM_PROMPT_DIRECT = (
    "You are a helpful, precise assistant. "
    "Answer multiple-choice questions by responding with ONLY the letter "
    "of the correct answer. "
    "Do not explain your answer."
)

SYSTEM_PROMPT_COT = (
    "You are a helpful, precise assistant. "
    "Think through multiple-choice questions step by step, then end your "
    "response with a new line in exactly this format:\n"
    "Final Answer: <letter>"
)

# CoT-friendly generation budget (bare-letter mode uses a much smaller one — see main()).
DEFAULT_COT_MAX_TOKENS   = 256
DEFAULT_COT_STEPS        = 128
DEFAULT_COT_BLOCK_LENGTH = 32


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
        ds = load_dataset("cais/mmlu", subject)
    else:
        ds = load_dataset("cais/mmlu", "all")

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


def _load_mmlu_pro(subject: Optional[str], limit: int) -> list[dict]:
    """
    Load MMLU-Pro test questions.
    Each returned dict: {question, choices:[A..J], answer_idx: int, subject}
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("[ERROR] 'datasets' package is required.  Run: pip install datasets")
        sys.exit(1)

    ds = load_dataset("TIGER-Lab/MMLU-Pro")
    split = ds["test"]
    items = []
    for row in split:
        subj = row.get("category", "mmlu_pro")
        if subject and subj != subject:
            continue
            
        # answer is provided as "A", "B", etc.
        ans_letter = row["answer"]
        if ans_letter not in CHOICES:
            continue
            
        items.append({
            "question":   row["question"],
            "choices":    row["options"],
            "answer_idx": CHOICES.index(ans_letter),
            "subject":    subj,
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
    ds = load_dataset("allenai/ai2_arc", name)
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
    elif task.startswith("mmlu_pro"):
        subject = task[len("mmlu_pro_"):] if task.startswith("mmlu_pro_") else None
        return _load_mmlu_pro(subject, limit)
    elif task.startswith("mmlu_"):
        subject = task[len("mmlu_"):]
        return _load_mmlu(subject, limit)
    elif task in ("arc_challenge", "arc_easy"):
        return _load_arc(task, limit)
    else:
        print(f"[ERROR] Unsupported task: '{task}'")
        print("Supported: mmlu, mmlu_pro, mmlu_<subject>, arc_challenge, arc_easy")
        sys.exit(1)


# ── Prompt construction ──────────────────────────────────────────────────────────

def build_prompt(item: dict, cot: bool = True) -> str:
    q = item["question"].strip()
    lines = [q, ""]
    valid_letters = CHOICES[:len(item["choices"])]
    for letter, text in zip(valid_letters, item["choices"]):
        lines.append(f"{letter}) {text}")

    letters_str = ", ".join(valid_letters[:-1]) + f", or {valid_letters[-1]}" if len(valid_letters) > 1 else valid_letters[0]
    if cot:
        lines.append(
            f"\nThink step by step, then end with a new line reading exactly "
            f"'Final Answer: <letter>', where <letter> is one of {letters_str}."
        )
    else:
        lines.append(f"\nAnswer with only the letter {letters_str}.")
    return "\n".join(lines)


# ── Server call ──────────────────────────────────────────────────────────────────

def call_server(
    base_url: str, prompt: str, timeout: int,
    gen_config: Optional[dict] = None,
    system_prompt: str = SYSTEM_PROMPT_COT,
    max_tokens: int = DEFAULT_COT_MAX_TOKENS,
    steps: int = DEFAULT_COT_STEPS,
) -> str:
    payload = {
        "model":       "inclusionAI/LLaDA-MoE-7B-A1B-Instruct",
        "messages":    [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": prompt},
        ],
        "temperature": 0.0,
        "top_p":       1.0,
        "max_tokens":  max_tokens,
        "steps":       steps,
    }
    if gen_config:
        payload.update(gen_config)
    resp = requests.post(
        f"{base_url}/v1/chat/completions",
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def parse_answer(text: str) -> Optional[str]:
    """Extract the model's answer letter, preferring an explicit
    'Final Answer: <letter>' / 'Answer: <letter>' marker (chain-of-thought
    mode) over positional heuristics (bare-letter mode)."""
    text = text.strip()
    # Explicit "Final Answer: X" / "Answer: X" marker, anywhere in the text
    # (CoT responses may mention several letters while reasoning — the
    # explicit marker is the actual decision, so check it first).
    m = re.search(r"(?:final\s*answer|answer)\s*[:\-]?\s*\(?([A-J])\)?\b", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # Exact single letter (bare-letter mode's common case)
    if text.upper() in CHOICES:
        return text.upper()
    # No marker found — fall back to the LAST standalone A-J letter mentioned,
    # since CoT reasoning tends to walk through options before concluding.
    matches = re.findall(r"\b([A-J])\b", text.upper())
    if matches:
        return matches[-1]
    # Last resort: first A-J character anywhere
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
    gen_config: Optional[dict] = None,
    cot: bool = True,
    max_tokens: int = DEFAULT_COT_MAX_TOKENS,
    steps: int = DEFAULT_COT_STEPS,
    transcript_path: Optional[str] = None,
) -> dict:
    correct = 0
    total   = len(items)
    errors  = 0
    per_subject: dict[str, list[bool]] = {}
    system_prompt = SYSTEM_PROMPT_COT if cot else SYSTEM_PROMPT_DIRECT

    def _eval_one(idx: int, item: dict):
        prompt    = build_prompt(item, cot=cot)
        try:
            raw       = call_server(
                base_url, prompt, timeout, gen_config=gen_config,
                system_prompt=system_prompt, max_tokens=max_tokens, steps=steps,
            )
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

    if transcript_path:
        records = []
        for idx, entry in enumerate(results_list):
            if entry is None:
                continue
            item, is_correct, raw = entry
            records.append({
                "idx":       idx,
                "subject":   item.get("subject", "unknown"),
                "question":  item["question"],
                "choices":   item["choices"],
                "expected":  CHOICES[item["answer_idx"]],
                "predicted": parse_answer(raw) if is_correct is not None else None,
                "correct":   is_correct,
                "raw_response": raw,
            })
        os.makedirs(os.path.dirname(transcript_path) if os.path.dirname(transcript_path) else ".", exist_ok=True)
        with open(transcript_path, "w") as f:
            json.dump(records, f, indent=2)
        print(f"Transcripts saved to {transcript_path}")

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
    ap.add_argument(
        "--direct-answer", action="store_true",
        help=(
            "Use the old bare-letter prompt ('answer with only the letter') "
            "instead of the default chain-of-thought prompt. MMLU-Pro was built "
            "to punish non-reasoning answers, so this will score notably lower "
            "on that task — mainly useful to reproduce old results."
        ),
    )
    ap.add_argument("--max-tokens", type=int, default=None,
                    help=f"Generation budget. Default: {DEFAULT_COT_MAX_TOKENS} "
                         f"(CoT) or 16 (--direct-answer).")
    ap.add_argument("--steps", type=int, default=None,
                    help=f"Diffusion steps. Default: {DEFAULT_COT_STEPS} "
                         f"(CoT) or 32 (--direct-answer).")
    ap.add_argument("--block-length", type=int, default=DEFAULT_COT_BLOCK_LENGTH,
                    help="Server-side block_length for generation.")
    ap.add_argument("--confidence-threshold", type=float, default=None,
                    help="Opt-in hierarchical threshold-based decoding (see "
                         "model_update/generate.py's generate_cached, ported from dInfer's "
                         "HierarchyDecoder). Default None is the server's normal "
                         "fixed-step-schedule behavior. This is the accuracy-impact check for "
                         "that feature -- run once without this flag and once with it, same "
                         "--task/--seed/--limit, and compare the two summary scores.")
    ap.add_argument("--low-confidence-threshold", type=float, default=0.4,
                    help="Floor applied to hierarchy decoding's per-segment picks. Ignored "
                         "unless --confidence-threshold is set.")
    ap.add_argument("--remask-threshold", type=float, default=None,
                    help="Opt-in, ported from dInfer's get_transfer_index_hierarchy_remask -- "
                         "lets an already-revealed position be reverted to MASK and "
                         "reconsidered if its confidence drops on a later step. Requires "
                         "--confidence-threshold to also be set. Fixes a real degenerate-"
                         "repetition failure mode found on long generations (see README).")
    ap.add_argument(
        "--save-transcripts", default=None,
        help=(
            "Path to save full per-question transcripts (question, choices, "
            "expected, predicted, and the raw model response) as JSON — for "
            "diagnosing whether wrong answers are reasoning failures or "
            "answer-extraction failures. Not saved by default."
        ),
    )
    args = ap.parse_args()

    cot = not args.direct_answer
    max_tokens = args.max_tokens if args.max_tokens is not None else (DEFAULT_COT_MAX_TOKENS if cot else 16)
    steps      = args.steps if args.steps is not None else (DEFAULT_COT_STEPS if cot else 32)

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
    print(f"Prompting  : {'chain-of-thought' if cot else 'direct (bare letter)'}")
    print(f"Max tokens : {max_tokens}  |  Steps: {steps}  |  Block length: {args.block_length}")
    print(f"Confidence threshold: {args.confidence_threshold} (low={args.low_confidence_threshold}, "
          f"remask={args.remask_threshold})")
    print(f"Seed       : {seed}\n")

    print("Loading dataset...", flush=True)
    items = load_dataset_for_task(args.task, limit)
    print(f"Loaded {len(items)} questions.\n")

    gen_config = {"block_length": args.block_length}
    if args.confidence_threshold is not None:
        gen_config["confidence_threshold"] = args.confidence_threshold
        gen_config["low_confidence_threshold"] = args.low_confidence_threshold
        if args.remask_threshold is not None:
            gen_config["remask_threshold"] = args.remask_threshold

    t0    = time.time()
    stats = evaluate(
        items, args.base_url, args.num_concurrent, args.timeout,
        gen_config=gen_config, cot=cot, max_tokens=max_tokens, steps=steps,
        transcript_path=args.save_transcripts,
    )
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