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
             dataset's reference solution. Uses 4-shot prompting by default,
             matching inclusionAI/dInfer's own gsm8k-llada-moe.yaml task
             config verbatim (exemplars, prompt template, generation
             lengths) -- see GSM8K_FEWSHOT_EXAMPLES and GSM8K_GEN_DEFAULTS.
             dInfer is Ant Group's inference framework for LLaDA-MoE/LLaDA2,
             the most authoritative source available for how the_paper.pdf's
             own numbers were likely produced (https://github.com/
             inclusionAI/dInfer). Pass --num-fewshot 0 for a zero-shot
             ablation against this default.
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

PAPER REFERENCE VALUES -- Table 3, not Table 1
------------------------------------------------
The paper has two result tables: Table 1 is the Base checkpoint (pretrain
only, no SFT); Table 3 is the Instruct checkpoint. Every model this repo
serves and every prompt this harness sends (chat-formatted, CoT-instructed)
targets the Instruct model, so **Table 3 is the correct reference, never
Table 1** -- they are different checkpoints, not a stricter/looser version
of the same number.

    Task      Table 3 (Instruct) value
    --------  -------------------------
    gsm8k     82.41
    cruxeval  42.38
    bbh       *no Table 3 entry* -- Table 3's Reasoning Tasks row only
              reports Drop/KorBench. BBH=52.71 exists ONLY in Table 1, for
              the Base checkpoint -- not a valid reference for the Instruct
              model served here. Treat this task's score as a same-model
              before/after comparison tool, not a reproduction target.

Even where a Table 3 number exists, it isn't an exact-reproduction bar
unless generation config matches. gsm8k's defaults (GSM8K_GEN_DEFAULTS:
max_tokens=1024, block_length=64) match both the paper's Section 4
("gen_length of 1024 and a block length of 64") and dInfer's
gsm8k-llada-moe.yaml directly -- the closest alignment available without
inclusionAI publishing exact reproduction code. bbh/cruxeval still use
run_correctness.py's smaller CoT defaults (max_tokens=256, block_length=32),
since no equivalent authoritative config was found for them.

block_length and steps are coupled regardless of task (num_blocks =
gen_length / block_length; steps_per_block = steps / num_blocks, how many
denoising refinement passes each block gets) -- raising block_length
without also raising steps proportionally *reduces* steps_per_block and
can hurt quality rather than help it. dInfer's yaml doesn't specify a step
count for gsm8k (steps is a model_update/-specific KV-cache parameter, not
an lm-eval-harness concept), so GSM8K_GEN_DEFAULTS' steps=256 is *derived*
(preserves this file's steps_per_block=16 at the larger block_length), not
sourced from dInfer -- don't treat it as equally authoritative as
max_tokens/block_length.

Even with all of this, exact reproduction of Table 3's 82.41 isn't
guaranteed: dInfer's config is the most authoritative source found, but
"most authoritative found" is not the same as "confirmed identical to
what produced the paper's number."

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

# LLaDA-MoE-7B-A1B-Instruct scores from the_paper.pdf, Table 3 (Instruct
# checkpoint -- NOT Table 1, which is the Base/pre-SFT checkpoint this repo
# does not serve). "bbh" is deliberately absent: Table 3 never reports a BBH
# score for the Instruct model (Table 1's 52.71 is Base-only, a different
# checkpoint). See the module docstring's "PAPER REFERENCE VALUES" section
# for the generation-config caveat (paper uses gen_length=1024/block=64;
# this script's CoT defaults are smaller).
PAPER_REFERENCE_TABLE3 = {
    "gsm8k": 82.41,
    "cruxeval": 42.38,
}

# gsm8k's generation defaults, aligned with inclusionAI/dInfer's
# evaluations/tasks/gsm8k/gsm8k-llada-moe.yaml (max_tokens/block_length --
# dInfer specifies gen_length=1024, block_length=64, matching the_paper.pdf
# Section 4's stated config) plus a derived `steps`: dInfer doesn't specify
# a diffusion step count (that's a model_update/-specific KV-cache
# generation parameter, not an lm-eval-harness concept), so 256 here is
# chosen to preserve this file's steps_per_block=16 density (steps_per_block
# = steps / (max_tokens / block_length)) at the larger block_length, not
# copied from any source -- see the module docstring's coupling note.
GSM8K_GEN_DEFAULTS = {"max_tokens": 1024, "steps": 256, "block_length": 64}

# BBH and CRUX-O were tried at the paper's 1024/64 generative-benchmark
# protocol (Section 4) and reverted: real evaluation showed it made both
# tasks WORSE than the short CoT default, not more comparable to Table 3
# (see README's "Adaptive Decoding" section) -- likely because our
# free-form-generation + string-extraction harness is a fundamentally
# different scoring methodology than whatever the paper used for these
# tasks, so matching its generation-length knob doesn't make results more
# comparable, and the coarser block_length=64 (twice the tokens revealed
# per step at the same steps_per_block density) hurt the fixed schedule's
# fine-grained iterative self-consistency. Both tasks use the short
# DEFAULT_COT_* config below instead, same as every other CoT task.

# The exact 4-shot GSM8K exemplar set from inclusionAI/dInfer's
# evaluations/tasks/gsm8k/gsm8k-llada-moe.yaml -- dInfer is Ant Group's own
# inference framework for diffusion LMs, built specifically for LLaDA-MoE
# and LLaDA2, and this task config is the LLaDA-MoE-specific variant
# (https://github.com/inclusionAI/dInfer). This is the most authoritative
# source available for how the_paper.pdf's Table 3 GSM8K=82.41 was likely
# produced -- far more likely than a guessed convention. Reproduced verbatim
# (these are themselves lm-evaluation-harness's standard default GSM8K
# exemplars: Angelo/Melanie studying, Mark's basketball, Bella's marbles,
# fruit baskets). Answers end in "The answer is <number>", NOT
# "Final Answer:" -- that's their convention, not ours; SYSTEM_PROMPT_GSM8K
# and extract_final_answer() below were updated to match it rather than
# impose our own marker on top of theirs.
GSM8K_FEWSHOT_EXAMPLES = [
    (
        "Angelo and Melanie want to plan how many hours over the next week "
        "they should study together for their test next week. They have 2 "
        "chapters of their textbook to study and 4 worksheets to memorize. "
        "They figure out that they should dedicate 3 hours to each chapter "
        "of their textbook and 1.5 hours for each worksheet. If they plan "
        "to study no more than 4 hours each day, how many days should they "
        "plan to study total over the next week if they take a 10-minute "
        "break every hour, include 3 10-minute snack breaks each day, and "
        "30 minutes for lunch each day?",
        "Angelo and Melanie think they should dedicate 3 hours to each of "
        "the 2 chapters, 3 hours x 2 chapters = 6 hours total.\n"
        "For the worksheets they plan to dedicate 1.5 hours for each "
        "worksheet, 1.5 hours x 4 worksheets = 6 hours total.\n"
        "Angelo and Melanie need to start with planning 12 hours to study, "
        "at 4 hours a day, 12 / 4 = 3 days.\n"
        "However, they need to include time for breaks and lunch. Every "
        "hour they want to include a 10-minute break, so 12 total hours x "
        "10 minutes = 120 extra minutes for breaks.\n"
        "They also want to include 3 10-minute snack breaks, 3 x 10 "
        "minutes = 30 minutes.\n"
        "And they want to include 30 minutes for lunch each day, so 120 "
        "minutes for breaks + 30 minutes for snack breaks + 30 minutes for "
        "lunch = 180 minutes, or 180 / 60 minutes per hour = 3 extra "
        "hours.\n"
        "So Angelo and Melanie want to plan 12 hours to study + 3 hours of "
        "breaks = 15 hours total.\n"
        "They want to study no more than 4 hours each day, 15 hours / 4 "
        "hours each day = 3.75\n"
        "They will need to plan to study 4 days to allow for all the time "
        "they need.\nThe answer is 4",
    ),
    (
        "Mark's basketball team scores 25 2 pointers, 8 3 pointers and 10 "
        "free throws.  Their opponents score double the 2 pointers but "
        "half the 3 pointers and free throws.  What's the total number of "
        "points scored by both teams added together?",
        "Mark's team scores 25 2 pointers, meaning they scored 25*2= 50 "
        "points in 2 pointers.\n"
        "His team also scores 6 3 pointers, meaning they scored 8*3= 24 "
        "points in 3 pointers\n"
        "They scored 10 free throws, and free throws count as one point "
        "so they scored 10*1=10 points in free throws.\n"
        "All together his team scored 50+24+10= 84 points\n"
        "Mark's opponents scored double his team's number of 2 pointers, "
        "meaning they scored 50*2=100 points in 2 pointers.\n"
        "His opponents scored half his team's number of 3 pointers, "
        "meaning they scored 24/2= 12 points in 3 pointers.\n"
        "They also scored half Mark's team's points in free throws, "
        "meaning they scored 10/2=5 points in free throws.\n"
        "All together Mark's opponents scored 100+12+5=117 points\n"
        "The total score for the game is both team's scores added "
        "together, so it is 84+117=201 points\nThe answer is 201",
    ),
    (
        "Bella has two times as many marbles as frisbees. She also has 20 "
        "more frisbees than deck cards. If she buys 2/5 times more of "
        "each item, what would be the total number of the items she will "
        "have if she currently has 60 marbles?",
        "When Bella buys 2/5 times more marbles, she'll have increased "
        "the number of marbles by 2/560 = 24\n"
        "The total number of marbles she'll have is 60+24 = 84\n"
        "If Bella currently has 60 marbles, and she has two times as many "
        "marbles as frisbees, she has 60/2 = 30 frisbees.\n"
        "If Bella buys 2/5 times more frisbees, she'll have 2/530 = 12 "
        "more frisbees.\n"
        "The total number of frisbees she'll have will increase to "
        "30+12 = 42\n"
        "Bella also has 20 more frisbees than deck cards, meaning she has "
        "30-20 = 10 deck cards\n"
        "If she buys 2/5 times more deck cards, she'll have 2/5*10 = 4 "
        "more deck cards.\nThe total number of deck cards she'll have is "
        "10+4 = 14\n"
        "Together, Bella will have a total of 14+42+84 = 140 items\n"
        "The answer is 140",
    ),
    (
        "A group of 4 fruit baskets contains 9 apples, 15 oranges, and 14 "
        "bananas in the first three baskets and 2 less of each fruit in "
        "the fourth basket. How many fruits are there?",
        "For the first three baskets, the number of apples and oranges in "
        "one basket is 9+15=24\n"
        "In total, together with bananas, the number of fruits in one "
        "basket is 24+14=38 for the first three baskets.\n"
        "Since there are three baskets each having 38 fruits, there are "
        "3*38=114 fruits in the first three baskets.\n"
        "The number of apples in the fourth basket is 9-2=7\n"
        "There are also 15-2=13 oranges in the fourth basket\n"
        "The combined number of oranges and apples in the fourth basket "
        "is 13+7=20\n"
        "The fourth basket also contains 14-2=12 bananas.\n"
        "In total, the fourth basket has 20+12=32 fruits.\n"
        "The four baskets together have 32+114=146 fruits.\n"
        "The answer is 146",
    ),
]

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


def _build_gsm8k_fewshot_prefix(num_fewshot: int) -> str:
    """Renders exemplars with the same 'Question/Please reason step by
    step/Answer' template as the real question (matches dInfer's doc_to_text,
    which lm-eval-harness applies uniformly to fewshot examples and the
    actual query)."""
    if num_fewshot <= 0:
        return ""
    examples = GSM8K_FEWSHOT_EXAMPLES[:num_fewshot]
    blocks = [
        f"Question: {q}\nPlease reason step by step\nAnswer: {a}"
        for q, a in examples
    ]
    return "\n\n".join(blocks) + "\n\n"


def _load_gsm8k(limit: int, num_fewshot: int = 4) -> list[TaskItem]:
    load_dataset = _load_datasets_or_die()
    ds = load_dataset("gsm8k", "main")["test"]
    prefix = _build_gsm8k_fewshot_prefix(num_fewshot)

    items = []
    for row in ds:
        answer_text = row["answer"]
        if "####" not in answer_text:
            continue
        expected = answer_text.split("####")[-1].strip().replace(",", "")
        items.append(TaskItem(
            prompt=f"{prefix}Question: {row['question'].strip()}\n"
                   f"Please reason step by step\nAnswer:",
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


def load_task(task: str, limit: int, num_fewshot: int = 4) -> tuple[list[TaskItem], str]:
    """Returns (items, system_prompt). num_fewshot only affects gsm8k."""
    if task == "gsm8k":
        return _load_gsm8k(limit, num_fewshot), SYSTEM_PROMPT_GSM8K
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
    "You are a helpful assistant. Solve the math word problem, reasoning "
    "step by step in the same style as the worked examples above, then "
    "state your final numeric answer the same way they do."
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
    """Text after an explicit 'Final Answer:' marker (bbh/cruxeval's
    convention) or 'The answer is ...' (gsm8k's dInfer-aligned convention,
    since its fewshot exemplars demonstrate that phrasing, not ours) --
    whichever appears. Uses the FIRST match of either pattern, not the
    last: with a fewshot completion-style prompt the model can drift into
    hallucinating a new "Question: ..." after its real answer (the server
    has no stop-sequence support to cut it off -- see ChatRequest in
    src/server.py), and the real answer is always the first occurrence,
    before any such drift. Falls back to the last non-empty line if
    neither marker appears."""
    text = text.strip()
    m = re.search(r"final\s*answer\s*[:\-]\s*(.+)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip().splitlines()[0].strip()
    m = re.search(r"the\s+answer\s+is\s*[:\-]?\s*(.+)", text, re.IGNORECASE)
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
    ap.add_argument("--max-tokens", type=int, default=None,
                    help=f"Generation budget. Default: {GSM8K_GEN_DEFAULTS['max_tokens']} "
                         f"for gsm8k (matches dInfer's gsm8k-llada-moe.yaml / the paper's "
                         f"gen_length=1024), {DEFAULT_COT_MAX_TOKENS} otherwise.")
    ap.add_argument("--steps", type=int, default=None,
                    help=f"Diffusion steps. Default: {GSM8K_GEN_DEFAULTS['steps']} for "
                         f"gsm8k (preserves this file's steps_per_block density at "
                         f"block_length=64 -- not from dInfer, which doesn't specify "
                         f"this), {DEFAULT_COT_STEPS} otherwise.")
    ap.add_argument("--block-length", type=int, default=None,
                    help=f"Server-side block_length. Default: "
                         f"{GSM8K_GEN_DEFAULTS['block_length']} for gsm8k (matches "
                         f"dInfer / the paper), {DEFAULT_COT_BLOCK_LENGTH} otherwise.")
    ap.add_argument(
        "--save-transcripts", default=None,
        help="Path to save full per-question transcripts as JSON. Not saved by default.",
    )
    ap.add_argument(
        "--num-fewshot", type=int, default=4,
        help="Number of worked examples to prepend for gsm8k (default 4, matching "
             "dInfer's gsm8k-llada-moe.yaml). Ignored by bbh/cruxeval.",
    )
    ap.add_argument("--confidence-threshold", type=float, default=None,
                    help="Opt-in hierarchical threshold-based decoding (see "
                         "model_update/generate.py's generate_cached, ported from dInfer's "
                         "HierarchyDecoder). Default None is the server's normal "
                         "fixed-step-schedule behavior. Validated on MMLU-Pro (see README); "
                         "this is the corroborating check on a free-form-answer task with a "
                         "different block_length profile (64 vs MMLU-Pro's 32).")
    ap.add_argument("--low-confidence-threshold", type=float, default=0.4,
                    help="Floor applied to hierarchy decoding's per-segment picks. Ignored "
                         "unless --confidence-threshold is set.")
    args = ap.parse_args()

    seed = args.seed if args.seed is not None else random.randint(0, 999_999)
    random.seed(seed)
    limit = args.limit if args.limit > 0 else 0

    if args.task == "gsm8k":
        max_tokens = args.max_tokens if args.max_tokens is not None else GSM8K_GEN_DEFAULTS["max_tokens"]
        steps = args.steps if args.steps is not None else GSM8K_GEN_DEFAULTS["steps"]
        block_length = args.block_length if args.block_length is not None else GSM8K_GEN_DEFAULTS["block_length"]
    else:
        max_tokens = args.max_tokens if args.max_tokens is not None else DEFAULT_COT_MAX_TOKENS
        steps = args.steps if args.steps is not None else DEFAULT_COT_STEPS
        block_length = args.block_length if args.block_length is not None else DEFAULT_COT_BLOCK_LENGTH

    print(f"\nTarget     : {args.base_url}")
    print(f"Task       : {args.task}")
    print(f"Limit      : {limit if limit else 'full dataset'}")
    print(f"Concurrent : {args.num_concurrent}")
    print(f"Max tokens : {max_tokens}  |  Steps: {steps}  |  Block length: {block_length}")
    if args.task == "gsm8k":
        print(f"Few-shot   : {args.num_fewshot}")
    print(f"Confidence threshold: {args.confidence_threshold} (low={args.low_confidence_threshold})")
    print(f"Seed       : {seed}\n")

    print("Loading dataset...", flush=True)
    items, system_prompt = load_task(args.task, limit, args.num_fewshot)
    print(f"Loaded {len(items)} questions.\n")

    grade_fn = grader_for(args.task)
    gen_config = {"block_length": block_length}
    if args.confidence_threshold is not None:
        gen_config["confidence_threshold"] = args.confidence_threshold
        gen_config["low_confidence_threshold"] = args.low_confidence_threshold

    t0 = time.time()
    stats = evaluate(
        items, grade_fn, system_prompt, args.base_url, args.num_concurrent,
        args.timeout, gen_config=gen_config, max_tokens=max_tokens,
        steps=steps, transcript_path=args.save_transcripts,
    )
    elapsed = time.time() - t0

    print(f"\nTotal time : {elapsed:.1f}s  ({elapsed/max(len(items),1):.1f}s/question)")

    if args.baseline:
        print_comparison(stats, args.baseline, args.task, args.config_name or "Current")
    else:
        print_results(stats, args.task, "Results")

    print_paper_reference(stats, args.task, max_tokens, block_length)

    if args.output:
        save_summary(stats, args.task, args.output, seed, args.config_name)


def print_paper_reference(stats: dict, task: str, max_tokens: int, block_length: int) -> None:
    """the_paper.pdf Table 3 (LLaDA-MoE-7B-A1B-Instruct) comparison -- see
    PAPER_REFERENCE_TABLE3 and the module docstring for why Table 3, not
    Table 1, and why BBH has no entry."""
    base_task = task.split("_")[0] if task.startswith("bbh_") else task

    if base_task == "bbh":
        print(
            "  Paper reference (Table 3) : none -- the Instruct model's "
            "Reasoning Tasks row only reports Drop/KorBench. Table 1's "
            "BBH=52.71 is the Base (pre-SFT) checkpoint, not a valid "
            "reference here. Use this score for before/after comparisons "
            "on this repo's own changes, not paper reproduction.\n"
        )
        return

    ref = PAPER_REFERENCE_TABLE3.get(base_task)
    if ref is None:
        return

    acc_pct = stats["accuracy"] * 100
    diff = acc_pct - ref
    print(f"  Paper reference (Table 3, Instruct) : {ref:.2f}%")
    print(f"  This run                            : {acc_pct:.2f}%  ({diff:+.2f} pts)")
    if (max_tokens, block_length) != (1024, 64):
        print(
            "  Note: paper's own eval used gen_length=1024, block_length=64 "
            f"(Section 4); this run used max_tokens={max_tokens}, "
            f"block_length={block_length} -- not an exact-reproduction "
            "config, treat the delta above as directional.\n"
        )
    else:
        print()


if __name__ == "__main__":
    main()
