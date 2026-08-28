"""
Diagnostic: isolate whether dminfr.engine's KV-cache buffer is the cause of
the "collapses to bare 'Final Answer: X' with zero reasoning" bug found when
comparing dminfr.engine's CoT MMLU/MMLU-Pro accuracy against HF (44-72% of
dminfr.engine's responses are <100 chars vs 0% for HF, at gen_length=256,
steps=128, block_length=32 -> 8 blocks).

Compares TWO generation paths using the SAME dminfr.engine model class and
weights, so any difference is attributable ONLY to caching:
  1. Cached   - dminfr.engine.generate.generate_cached (the production path)
  2. Dense    - full sequence recomputed from scratch every denoising step,
                via LLaDAMoEKV.forward with cache_buffer=None, past_kv=None
                (mirrors dminfr.reference.generate.generate's algorithm exactly, but
                through dminfr.engine's own model class/weights)

If the collapse only happens in (1), the bug is in the KV-cache/block logic.
If it happens in both, it's elsewhere in dminfr.engine's model code and
unrelated to caching.

Run on a specific known-bad MMLU-Pro question (single GPU is fine -- this is
a correctness check, not a perf test):
    python archive/investigations/diagnose_cache_vs_dense.py --weight-dir ./weights --item-idx 4
    python archive/investigations/diagnose_cache_vs_dense.py --weight-dir ./weights --item-idx 12
    python archive/investigations/diagnose_cache_vs_dense.py --weight-dir ./weights --item-idx 17

(--item-idx is 0-indexed into the same --seed 42 --task mmlu_pro sample used
by run_correctness.py; the ones above are 1-indexed #5, #13, #18 from that
run, which collapsed to "Final Answer: G"/"F"/"B" respectively.)
"""

import argparse
import sys
from pathlib import Path

import torch

workspace_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(workspace_root))

from dminfr.engine.distributed import init_distributed
from dminfr.engine.model import LLaDAMoEKV, FULL_CFG
from dminfr.engine.generate import generate_cached, generate_dense
from benchmarks.correctness.run_correctness import (
    build_prompt, load_dataset_for_task, SYSTEM_PROMPT_COT, CHOICES,
    DEFAULT_COT_MAX_TOKENS, DEFAULT_COT_STEPS, DEFAULT_COT_BLOCK_LENGTH,
)

MASK_ID = 156895

# generate_dense (the no-KV-cache comparison path) now lives in
# dminfr/engine/generate.py -- also used by benchmarks/check_time_inference.py's
# --no-cache flag for timing. Was a local duplicate here (with a slower,
# Python-loop per-row topk selection unsuitable for timing use); switched
# to the shared, vectorized implementation so both call sites always test
# the exact same "dense" semantics.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weight-dir", default="weights")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--task", default="mmlu_pro")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--item-idx", type=int, default=4, help="0-indexed item within the sampled set.")
    parser.add_argument("--gen-length", type=int, default=DEFAULT_COT_MAX_TOKENS)
    parser.add_argument("--steps", type=int, default=DEFAULT_COT_STEPS)
    parser.add_argument("--block-length", type=int, default=DEFAULT_COT_BLOCK_LENGTH)
    args = parser.parse_args()

    init_distributed()
    if torch.cuda.is_available():
        torch.cuda.set_device(args.device)

    import random
    random.seed(args.seed)
    items = load_dataset_for_task(args.task, args.limit)
    item = items[args.item_idx]
    print(f"Question idx={args.item_idx} subject={item['subject']} expected={CHOICES[item['answer_idx']]}")

    from transformers import AutoTokenizer
    from dminfr.engine.distributed import load_weights_tp
    tok = AutoTokenizer.from_pretrained(args.weight_dir, trust_remote_code=True)

    print("Loading model (eager MoE, single GPU)...")
    model = LLaDAMoEKV(FULL_CFG, use_fused_moe=False).to(torch.bfloat16).eval()
    load_weights_tp(model, args.weight_dir, verbose=False)
    model = model.to(args.device)

    user_prompt = build_prompt(item, cot=True)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_COT},
        {"role": "user", "content": user_prompt},
    ]
    prompt_text = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    prompt_ids = tok(prompt_text, return_tensors="pt")["input_ids"].to(args.device)

    gen_kwargs = dict(gen_length=args.gen_length, steps=args.steps, block_length=args.block_length)

    print("\n=== CACHED (generate_cached, production path) ===")
    out_cached = generate_cached(model, prompt_ids, **gen_kwargs)
    text_cached = tok.decode(out_cached[0], skip_special_tokens=True)
    print(f"len={len(text_cached)}")
    print(text_cached[:1000])

    print("\n=== DENSE (no cache, full recompute every step, same model/weights) ===")
    out_dense = generate_dense(model, prompt_ids, **gen_kwargs)
    text_dense = tok.decode(out_dense[0], skip_special_tokens=True)
    print(f"len={len(text_dense)}")
    print(text_dense[:1000])

    print("\n=== VERDICT ===")
    cached_short = len(text_cached) < 100
    dense_short = len(text_dense) < 100
    print(f"cached collapsed: {cached_short}   dense collapsed: {dense_short}")
    if cached_short and not dense_short:
        print("-> Bug is in the KV-cache / block-wise generation logic (generate_cached / KVCacheBuffer).")
    elif cached_short and dense_short:
        print("-> Bug reproduces WITHOUT caching too -- not a caching issue, look elsewhere in dminfr.engine's model code.")
    elif not cached_short and not dense_short:
        print("-> Neither collapsed on this specific question -- try a different --item-idx known to have collapsed.")
    else:
        print("-> Unexpected: dense collapsed but cached didn't. Worth another look.")


if __name__ == "__main__":
    main()
