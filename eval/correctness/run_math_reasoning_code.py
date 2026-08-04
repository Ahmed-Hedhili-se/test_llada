"""
Math / Reasoning / Coding correctness evaluation for LLaDA-MoE-7B-A1B-Instruct.

Extends eval/correctness/run_correctness.py's MMLU/MMLU-Pro/ARC harness with
three benchmarks that need free-form answer grading instead of single-letter
matching. Kept as a separate file rather than bolted onto run_correctness.py
because the *grading model* is fundamentally different (numeric match / exact
string match vs. "does this equal CHOICES[answer_idx]") -- but it reuses
run_correctness's generic, non-MCQ-specific pieces directly: call_server()
for the HTTP round-trip, and print_results()/print_comparison()/save_summary()
for reporting, since those only operate on a generic {accuracy, correct,
total, errors, per_subject} stats dict.

DATASET SELECTION -- based on the_paper.pdf
---------------------------------------------
the_paper.pdf is the LLaDA-MoE technical report (arXiv:2509.24389, "LLaDA-MoE:
A Sparse MoE Diffusion Language Model" -- the same model this repo serves).
Section 4 ("Experiments") states the model is evaluated on benchmarks spanning
"mathematics (Cobbe et al., 2021; ...), ... reasoning (Suzgun et al., 2022;
...), coding (Gu et al., 2024; ...)" -- i.e. GSM8K, BBH, and CRUXEval are the
paper's own citations for these three categories (Tables 1/3 report LLaDA-MoE
scores directly on GSM8K and CRUX-O for the Instruct model, and on BBH for the
Base model). One dataset per category was picked from that list on two
practical grounds beyond just "the paper uses it":

  - open-source and API-loadable: all three load via a single
    `datasets.load_dataset(...)` call, same mechanism run_correctness.py
    already uses for MMLU/ARC -- no manual download, no gated license.
  - gradable without code execution: like MMLU's letter-matching, all three
    are graded by comparing extracted model text to a fixed ground-truth
    string/number. This is *why* CRUX-O was picked over e.g. HumanEval/MBPP
    for the "coding" slot -- CRUX-O asks the model to predict a function's
    output for a given input, and the dataset ships the correct output
    already computed by its authors, so grading never needs to execute
    model-generated code (unlike HumanEval/MBPP, which run_correctness.py
    already excludes for exactly this reason -- see diagnose_humaneval.py).

  gsm8k    : Math.      Grade-school math word problems (Cobbe et al., 2021).
             Free-form numeric answer, ground truth after "#### " in the
             dataset's reference solution.
  bbh      : Reasoning. BIG-Bench-Hard, 27 diverse reasoning tasks (Suzgun
             et al., 2022). Free-form short answer (word/phrase/number/
             "(A)"-style letter depending on subtask). NOTE: this runs
             zero-shot CoT, not the original paper's few-shot exemplar
             format (see _load_bbh docstring) -- absolute scores will not
             match published BBH numbers, but relative before/after
             comparisons on this repo's own model changes remain valid.
  cruxeval : Coding.     CRUX-O: given a Python function and an input,
             predict its return value (Gu et al., 2024). Ground truth is
             precomputed by the dataset authors.

Usage:
    python eval/correctness/run_math_reasoning_code.py --task gsm8k    --limit 400 --output results/gsm8k.json
    python eval/correctness/run_math_reasoning_code.py --task bbh      --limit 400 --output results/bbh.json
    python eval/correctness/run_math_reasoning_code.py --task cruxeval --limit 400 --output results/cruxeval.json
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
from dataclasses import dataclass
from typing import Callable, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from eval.correctness.run_correctness import (
    call_server, print_results, print_comparison, save_summary,
    DEFAULT_URL, DEFAULT_TIMEOUT, DEFAULT_COT_MAX_TOKENS, DEFAULT_COT_STEPS,
    DEFAULT_COT_BLOCK_LENGTH,
)

DEFAULT_TASK = "gsm8k"

# All 27 canonical BIG-Bench-Hard tasks (Suzgun et al., 2022).
BBH_SUBTASKS = [
    "boolean_expressions", "causal_judgement", "date_understanding",
    "disambiguation_qa", "dyck_languages", "formal_fallacies",
    "geometric_shapes", "hyperbaton", "logical_deduction_five_objects",
    "logical_deduction_seven_objects", "logical_deduction_three_objects",
    "movie_recommendation", "multistep_arithmetic_two", "navigate",
    "object_counting", "penguins_in_a_table", "reasoning_about_colored_objects",
    "ruin_names", "salient_translation_error_detection", "snarks",
    "sports_understanding", "temporal_sequences",
    "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects",
    "tracking_shuffled_objects_three_objects", "web_of_lies", "word_sorting",
]


@dataclass
class TaskItem:
    prompt: str        # full user-turn prompt text (question already formatted)
    expected: str       # ground-truth string/number used for grading
    subject: str         # per-subject breakdown key (bbh subtask, or task name)


# ── Dataset loading ──────────────────────────────────────────────────────────────

def _load_datasets_or_die():
    try:
        from datasets import load_dataset
        return load_dataset
    except ImportError:
        print("[ERROR] 'datasets' package is required. Run: pip install datasets")
        sys.exit(1)


def _load_gsm8k(limit: int) -> list[TaskItem]:
    load_dataset = _load_datasets_or_die()
    ds = load_dataset("gsm8k", "main")["test"]

    items = []
    for row in ds:
        answer_text = row["answer"]
        if "####" not in answer_text:
            continue
        expected = answer_text.split("####")[-1].strip().replace(",", "")
        items.append(TaskItem(
            prompt=row["question"].strip(),
            expected=expected,
            subject="gsm8k",
        ))

    if limit and len(items) > limit:
        random.shuffle(items)
        items = items[:limit]
    return items


def _load_bbh(subtask: Optional[str], limit: int) -> list[TaskItem]:
    """
    Load BIG-Bench-Hard. The HF mirror's `input` field is the bare question
    for that task -- it does NOT include the original paper's few-shot
    exemplar prefix (those live separately in the BBH GitHub repo's
    cot-prompts/ directory, not in this dataset). Reproducing them here would
    need bundling 27 separate few-shot templates; instead this evaluates
    zero-shot CoT for all subtasks uniformly. This is a real, valid
    evaluation setup (many papers report zero-shot BBH), it is just not the
    same setup as the paper's own BBH number -- treat this task as a
    same-model-before/after comparison tool, not a way to reproduce Table 1.
    """
    load_dataset = _load_datasets_or_die()
    subtasks = [subtask] if subtask else BBH_SUBTASKS

    items = []
    for name in subtasks:
        ds = load_dataset("lukaemon/bbh", name)["test"]
        for row in ds:
            items.append(TaskItem(
                prompt=row["input"].strip(),
                expected=row["target"].strip(),
                subject=name,
            ))

    if limit and len(items) > limit:
        random.shuffle(items)
        items = items[:limit]
    return items


def _load_cruxeval(limit: int) -> list[TaskItem]:
    load_dataset = _load_datasets_or_die()
    ds = load_dataset("cruxeval-org/cruxeval")["test"]

    items = []
    for row in ds:
        prompt = (
            f"Consider the following Python function:\n\n"
            f"```python\n{row['code']}\n```\n\n"
            f"What is the return value of the call: {row['input']}"
        )
        items.append(TaskItem(
            prompt=prompt,
            expected=row["output"].strip(),
            subject="cruxeval",
        ))

    if limit and len(items) > limit:
        random.shuffle(items)
        items = items[:limit]
    return items


def load_task(task: str, limit: int) -> tuple[list[TaskItem], str]:
    """Returns (items, system_prompt)."""
    if task == "gsm8k":
        return _load_gsm8k(limit), SYSTEM_PROMPT_GSM8K
    elif task == "bbh":
        return _load_bbh(None, limit), SYSTEM_PROMPT_BBH
    elif task.startswith("bbh_"):
        subtask = task[len("bbh_"):]
        if subtask not in BBH_SUBTASKS:
            print(f"[ERROR] Unknown BBH subtask: '{subtask}'")
            print(f"Supported: {', '.join(BBH_SUBTASKS)}")
            sys.exit(1)
        return _load_bbh(subtask, limit), SYSTEM_PROMPT_BBH
    elif task == "cruxeval":
        return _load_cruxeval(limit), SYSTEM_PROMPT_CRUXEVAL
    else:
        print(f"[ERROR] Unsupported task: '{task}'")
        print("Supported: gsm8k, bbh, bbh_<subtask>, cruxeval")
        sys.exit(1)


# ── Prompts ──────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT_GSM8K = (
    "You are a helpful, precise assistant skilled at solving math word "
    "problems. Think through the problem step by step, then end your "
    "response with a new line in exactly this format:\n"
    "Final Answer: <number>"
)

SYSTEM_PROMPT_BBH = (
    "You are a helpful, precise assistant skilled at multi-step logical "
    "reasoning. Think through the problem step by step, then end your "
    "response with a new line in exactly this format:\n"
    "Final Answer: <answer>\n"
    "The <answer> must be the shortest possible correct answer (a single "
    "word, phrase, number, or option letter) with no extra explanation on "
    "that line."
)

SYSTEM_PROMPT_CRUXEVAL = (
    "You are a helpful, precise assistant skilled at tracing Python code "
    "execution. Given a function and an input, work out exactly what the "
    "function returns. Think step by step, then end your response with a "
    "new line in exactly this format:\n"
    "Final Answer: <output>\n"
    "The <output> must be the exact return value only (as Python's repr "
    "would print it), nothing else."
)


# ── Answer extraction & grading ─────────────────────────────────────────────────

def extract_final_answer(text: str) -> Optional[str]:
    """Text after an explicit 'Final Answer:' marker; falls back to the last
    non-empty line if the model didn't use the marker."""
    text = text.strip()
    m = re.search(r"final\s*answer\s*[:\-]\s*(.+)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip().splitlines()[0].strip()
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return lines[-1] if lines else None


def _grade_gsm8k(predicted: Optional[str], expected: str, raw: str = "") -> bool:
    r"""
    Numeric match, with a fallback for the "Final Answer:" marker line being
    immediately followed by a multi-line LaTeX block (e.g.
    'Final Answer:\n\[\n\boxed{16\n\]' -- note the missing closing brace,
    which real generations produce). extract_final_answer()'s single-line
    regex only sees the '\[' on the first line in that case and finds no
    digits, even though the correct answer is right there one line down.
    When that happens, search the full raw response for the last \boxed{...}
    (robust to a missing closing brace), then finally the last number
    anywhere in the raw text, before giving up.
    """
    if predicted is not None:
        pred_nums = re.findall(r"-?\d+\.?\d*", predicted.replace(",", ""))
        if pred_nums:
            return _numeric_match(pred_nums[-1], expected)

    boxed = re.findall(r"\\boxed\{?\s*(-?\d[\d,]*\.?\d*)", raw)
    if boxed:
        return _numeric_match(boxed[-1].replace(",", ""), expected)

    all_nums = re.findall(r"-?\d+\.?\d*", raw.replace(",", ""))
    if all_nums:
        return _numeric_match(all_nums[-1], expected)

    return False


def _numeric_match(pred_num: str, expected: str) -> bool:
    try:
        return abs(float(pred_num) - float(expected)) < 1e-4
    except ValueError:
        return False


def _normalize_short_answer(s: str) -> str:
    s = s.strip().strip(".").strip()
    s = re.sub(r"^\((.+)\)$", r"\1", s)  # "(A)" -> "A"
    return s.lower()


def _grade_bbh(predicted: Optional[str], expected: str, raw: str = "") -> bool:
    if predicted is None:
        return False
    return _normalize_short_answer(predicted) == _normalize_short_answer(expected)


def _grade_cruxeval(predicted: Optional[str], expected: str, raw: str = "") -> bool:
    if predicted is None:
        return False

    def norm(s: str) -> str:
        s = s.strip()
        if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
            s = s[1:-1]
        return s

    if norm(predicted) == norm(expected):
        return True
    # Looser fallback: ignore whitespace differences inside lists/dicts/etc.
    return re.sub(r"\s+", "", norm(predicted)) == re.sub(r"\s+", "", norm(expected))


GRADERS: dict[str, Callable[[Optional[str], str, str], bool]] = {
    "gsm8k": _grade_gsm8k,
    "bbh": _grade_bbh,
    "cruxeval": _grade_cruxeval,
}


def grader_for(task: str) -> Callable[[Optional[str], str, str], bool]:
    base_task = task.split("_")[0] if task.startswith("bbh_") else task
    return GRADERS[base_task]


# ── Evaluation loop ───────────────────────────────────────────────────────────────

def evaluate(
    items: list[TaskItem],
    grade_fn: Callable[[Optional[str], str, str], bool],
    system_prompt: str,
    base_url: str,
    num_concurrent: int,
    timeout: int,
    gen_config: Optional[dict] = None,
    max_tokens: int = DEFAULT_COT_MAX_TOKENS,
    steps: int = DEFAULT_COT_STEPS,
    transcript_path: Optional[str] = None,
) -> dict:
    correct = 0
    total = len(items)
    errors = 0
    per_subject: dict[str, list[bool]] = {}

    def _eval_one(idx: int, item: TaskItem):
        try:
            raw = call_server(
                base_url, item.prompt, timeout, gen_config=gen_config,
                system_prompt=system_prompt, max_tokens=max_tokens, steps=steps,
            )
            predicted = extract_final_answer(raw)
        except Exception as e:
            return idx, item, None, str(e)
        is_correct = grade_fn(predicted, item.expected, raw)
        return idx, item, is_correct, raw

    results_list = [None] * total
    with ThreadPoolExecutor(max_workers=num_concurrent) as ex:
        futures = {ex.submit(_eval_one, i, item): i for i, item in enumerate(items)}

        done = 0
        for future in as_completed(futures):
            idx, item, is_correct, raw = future.result()
            results_list[idx] = (item, is_correct, raw)
            done += 1

            per_subject.setdefault(item.subject, [])

            if is_correct is None:
                errors += 1
                per_subject[item.subject].append(False)
                print(f"  [{done:4d}/{total}] ERROR: {raw}", flush=True)
            else:
                if is_correct:
                    correct += 1
                per_subject[item.subject].append(is_correct)
                mark = "✅" if is_correct else "❌"
                pred = extract_final_answer(raw) or "?"
                print(
                    f"  [{done:4d}/{total}] {mark}  "
                    f"Expected={item.expected!r} Got={pred!r}  "
                    f"({item.subject})",
                    flush=True,
                )

    accuracy = correct / total if total else 0.0
    per_subject_acc = {s: sum(v) / len(v) for s, v in per_subject.items() if v}

    if transcript_path:
        records = []
        for idx, entry in enumerate(results_list):
            if entry is None:
                continue
            item, is_correct, raw = entry
            records.append({
                "idx": idx,
                "subject": item.subject,
                "prompt": item.prompt,
                "expected": item.expected,
                "predicted": extract_final_answer(raw) if is_correct is not None else None,
                "correct": is_correct,
                "raw_response": raw,
            })
        os.makedirs(os.path.dirname(transcript_path) if os.path.dirname(transcript_path) else ".", exist_ok=True)
        with open(transcript_path, "w") as f:
            json.dump(records, f, indent=2)
        print(f"Transcripts saved to {transcript_path}")

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "errors": errors,
        "per_subject": per_subject_acc,
    }


# ── Entry point ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=(
            "Math/Reasoning/Coding correctness evaluation for "
            "LLaDA-MoE-7B-A1B-Instruct (GSM8K, BBH, CRUX-O), selected from "
            "the_paper.pdf's own evaluation suite. Free-form answer grading "
            "-- see run_correctness.py for MMLU/MMLU-Pro/ARC (multiple choice)."
        )
    )
    ap.add_argument("--task", default=DEFAULT_TASK,
                    help="Task: gsm8k, bbh, bbh_<subtask>, cruxeval. (default: gsm8k)")
    ap.add_argument("--base-url", default=DEFAULT_URL)
    ap.add_argument("--limit", type=int, default=200,
                    help="Max questions to evaluate (0 = full dataset).")
    ap.add_argument("--num-concurrent", type=int, default=1,
                    help="Concurrent requests to the server.")
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT,
                    help="Request timeout in seconds.")
    ap.add_argument("--output", default=None, help="Path to save the summary JSON.")
    ap.add_argument("--baseline", default=None,
                    help="Path to a previous summary JSON for delta comparison.")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--config-name", default="", help="Label stored in the summary JSON.")
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_COT_MAX_TOKENS,
                    help="Generation budget.")
    ap.add_argument("--steps", type=int, default=DEFAULT_COT_STEPS, help="Diffusion steps.")
    ap.add_argument("--block-length", type=int, default=DEFAULT_COT_BLOCK_LENGTH,
                    help="Server-side block_length for generation.")
    ap.add_argument(
        "--save-transcripts", default=None,
        help="Path to save full per-question transcripts as JSON. Not saved by default.",
    )
    args = ap.parse_args()

    seed = args.seed if args.seed is not None else random.randint(0, 999_999)
    random.seed(seed)
    limit = args.limit if args.limit > 0 else 0

    print(f"\nTarget     : {args.base_url}")
    print(f"Task       : {args.task}")
    print(f"Limit      : {limit if limit else 'full dataset'}")
    print(f"Concurrent : {args.num_concurrent}")
    print(f"Max tokens : {args.max_tokens}  |  Steps: {args.steps}  |  Block length: {args.block_length}")
    print(f"Seed       : {seed}\n")

    print("Loading dataset...", flush=True)
    items, system_prompt = load_task(args.task, limit)
    print(f"Loaded {len(items)} questions.\n")

    grade_fn = grader_for(args.task)
    gen_config = {"block_length": args.block_length}

    t0 = time.time()
    stats = evaluate(
        items, grade_fn, system_prompt, args.base_url, args.num_concurrent,
        args.timeout, gen_config=gen_config, max_tokens=args.max_tokens,
        steps=args.steps, transcript_path=args.save_transcripts,
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
