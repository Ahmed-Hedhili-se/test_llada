"""
Diagnostic: isolate whether model_update's KV-cache buffer is the cause of
the "collapses to bare 'Final Answer: X' with zero reasoning" bug found when
comparing model_update's CoT MMLU/MMLU-Pro accuracy against HF (44-72% of
model_update's responses are <100 chars vs 0% for HF, at gen_length=256,
steps=128, block_length=32 -> 8 blocks).

Compares TWO generation paths using the SAME model_update model class and
weights, so any difference is attributable ONLY to caching:
  1. Cached   - model_update.generate.generate_cached (the production path)
  2. Dense    - full sequence recomputed from scratch every denoising step,
                via LLaDAMoEKV.forward with cache_buffer=None, past_kv=None
                (mirrors src.generate.generate's algorithm exactly, but
                through model_update's own model class/weights)

If the collapse only happens in (1), the bug is in the KV-cache/block logic.
If it happens in both, it's elsewhere in model_update's model code and
unrelated to caching.

Run on a specific known-bad MMLU-Pro question (single GPU is fine -- this is
a correctness check, not a perf test):
    python eval/diagnose_cache_vs_dense.py --weight-dir ./weights --item-idx 4
    python eval/diagnose_cache_vs_dense.py --weight-dir ./weights --item-idx 12
    python eval/diagnose_cache_vs_dense.py --weight-dir ./weights --item-idx 17

(--item-idx is 0-indexed into the same --seed 42 --task mmlu_pro sample used
by run_correctness.py; the ones above are 1-indexed #5, #13, #18 from that
run, which collapsed to "Final Answer: G"/"F"/"B" respectively.)
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

workspace_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(workspace_root))

from model_update.distributed import init_distributed
from model_update.model import LLaDAMoEKV, FULL_CFG
from model_update.generate import generate_cached, add_gumbel_noise, get_num_transfer_tokens
from eval.correctness.run_correctness import (
    build_prompt, load_dataset_for_task, SYSTEM_PROMPT_COT, CHOICES,
    DEFAULT_COT_MAX_TOKENS, DEFAULT_COT_STEPS, DEFAULT_COT_BLOCK_LENGTH,
)

MASK_ID = 156895


@torch.no_grad()
def generate_dense(model, prompt_ids, gen_length, steps, block_length, temperature=0.0, remasking="low_confidence"):
    """Same denoising algorithm as generate_cached, but recomputes the FULL
    sequence from scratch every step (no KV cache at all) -- mirrors
    src.generate.generate's algorithm, through model_update's own model
    class/weights, to isolate caching as the only variable."""
    device = prompt_ids.device
    P = prompt_ids.shape[1]
    total_len = P + gen_length
    x = torch.full((1, total_len), MASK_ID, dtype=torch.long, device=device)
    x[:, :P] = prompt_ids

    num_blocks = gen_length // block_length
    steps_per_block = steps // num_blocks

    for block_idx in range(num_blocks):
        block_start = P + block_idx * block_length
        block_end = P + (block_idx + 1) * block_length
        block_mask_index = (x[:, block_start:block_end] == MASK_ID)
        num_transfer = get_num_transfer_tokens(block_mask_index, steps_per_block)

        for step in range(steps_per_block):
            active_ids = x[:, block_start:block_end]
            mask_index = (active_ids == MASK_ID)

            full_logits, _ = model(x, position_offset=0)
            logits = full_logits[:, block_start:block_end]

            logits_with_noise = add_gumbel_noise(logits, temperature)
            x0 = logits_with_noise.argmax(dim=-1)

            if remasking == "low_confidence":
                p = F.softmax(logits.float(), dim=-1)
                x0_p = p.gather(-1, x0.unsqueeze(-1)).squeeze(-1)
            else:
                x0_p = torch.rand(x0.shape, device=device)

            x0 = torch.where(mask_index, x0, active_ids)
            confidence = torch.where(mask_index, x0_p, torch.full_like(x0_p, -torch.inf))

            transfer_index = torch.zeros_like(x0, dtype=torch.bool)
            for j in range(confidence.shape[0]):
                k = num_transfer[j, step].item()
                if k > 0:
                    _, sel = torch.topk(confidence[j], k=int(k))
                    transfer_index[j, sel] = True

            active_ids = active_ids.clone()
            active_ids[transfer_index] = x0[transfer_index]
            x[:, block_start:block_end] = active_ids

    return x[:, P:]


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
    from model_update.distributed import load_weights_tp
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
        print("-> Bug reproduces WITHOUT caching too -- not a caching issue, look elsewhere in model_update's model code.")
    elif not cached_short and not dense_short:
        print("-> Neither collapsed on this specific question -- try a different --item-idx known to have collapsed.")
    else:
        print("-> Unexpected: dense collapsed but cached didn't. Worth another look.")


if __name__ == "__main__":
    main()
