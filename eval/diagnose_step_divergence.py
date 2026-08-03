"""
Follow-up to diagnose_cache_vs_dense.py: that script proved the "collapses to
bare Final Answer: X" bug is specific to generate_cached (KV-cache path) --
the dense (no-cache) path on the SAME model/weights/greedy-decoding never
collapses. This script catches the divergence in the act: runs BOTH paths
through only the FIRST block, step by step, printing the partially-decoded
block content after every single step side by side, so we can see the exact
step where cached and dense first disagree instead of only comparing final
output.

Run (single GPU, plain python -- point at an unused port if a server is
already running on 29500):
    MASTER_ADDR=localhost MASTER_PORT=29501 \
    python eval/diagnose_step_divergence.py --weight-dir ./weights --item-idx 4
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

workspace_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(workspace_root))

from model_update.distributed import init_distributed
from model_update.model import LLaDAMoEKV, FULL_CFG, KVCacheBuffer
from model_update.generate import add_gumbel_noise, get_num_transfer_tokens
from eval.correctness.run_correctness import (
    build_prompt, load_dataset_for_task, SYSTEM_PROMPT_COT, CHOICES,
    DEFAULT_COT_MAX_TOKENS, DEFAULT_COT_STEPS, DEFAULT_COT_BLOCK_LENGTH,
)

MASK_ID = 156895


def run_one_step(logits, active_ids, mask_index, num_transfer, step, device, temperature=0.0):
    """Shared step logic (mirrors _generate_block_cached exactly)."""
    logits_with_noise = add_gumbel_noise(logits, temperature)
    x0 = logits_with_noise.argmax(dim=-1)
    p = F.softmax(logits.float(), dim=-1)
    x0_p = p.gather(-1, x0.unsqueeze(-1)).squeeze(-1)

    x0 = torch.where(mask_index, x0, active_ids)
    confidence = torch.where(mask_index, x0_p, torch.full_like(x0_p, -torch.inf))

    transfer_index = torch.zeros_like(x0, dtype=torch.bool)
    for j in range(confidence.shape[0]):
        k = num_transfer[j, step].item()
        if k > 0:
            _, sel = torch.topk(confidence[j], k=int(k))
            transfer_index[j, sel] = True

    new_active = active_ids.clone()
    new_active[transfer_index] = x0[transfer_index]
    return new_active


@torch.no_grad()
def trace_block0_cached(model, prompt_ids, gen_length, steps, block_length, tok):
    device = prompt_ids.device
    P = prompt_ids.shape[1]
    total_len = P + gen_length
    x = torch.full((1, total_len), MASK_ID, dtype=torch.long, device=device)
    x[:, :P] = prompt_ids

    cache_buffer = KVCacheBuffer(
        num_layers=model.cfg.NL, batch_size=1,
        kvh_local=model.layers[0].self_attn.KVH_local,
        max_len=total_len, head_dim=model.cfg.HD,
        dtype=next(model.parameters()).dtype, device=device,
    )
    model(prompt_ids, position_offset=0, cache_buffer=cache_buffer, write_pos=0)
    cache_buffer.commit(P)

    num_blocks = gen_length // block_length
    steps_per_block = steps // num_blocks
    block_start, block_end = P, P + block_length
    block_mask_index = (x[:, block_start:block_end] == MASK_ID)
    num_transfer = get_num_transfer_tokens(block_mask_index, steps_per_block)

    print(f"\n--- CACHED trace, block 0 ({steps_per_block} steps) ---")
    for step in range(steps_per_block):
        suffix_ids = x[:, block_start:]
        active_ids = x[:, block_start:block_end]
        mask_index = (active_ids == MASK_ID)

        suffix_logits, _ = model(suffix_ids, position_offset=block_start, cache_buffer=cache_buffer, write_pos=block_start)
        logits = suffix_logits[:, :block_length]

        active_ids = run_one_step(logits, active_ids, mask_index, num_transfer, step, device)
        x[:, block_start:block_end] = active_ids
        partial = tok.decode(active_ids[0], skip_special_tokens=False)
        print(f"  step {step:2d}: {partial!r}")

    return x[:, block_start:block_end]


@torch.no_grad()
def trace_block0_dense(model, prompt_ids, gen_length, steps, block_length, tok):
    device = prompt_ids.device
    P = prompt_ids.shape[1]
    total_len = P + gen_length
    x = torch.full((1, total_len), MASK_ID, dtype=torch.long, device=device)
    x[:, :P] = prompt_ids

    num_blocks = gen_length // block_length
    steps_per_block = steps // num_blocks
    block_start, block_end = P, P + block_length
    block_mask_index = (x[:, block_start:block_end] == MASK_ID)
    num_transfer = get_num_transfer_tokens(block_mask_index, steps_per_block)

    print(f"\n--- DENSE trace, block 0 ({steps_per_block} steps) ---")
    for step in range(steps_per_block):
        active_ids = x[:, block_start:block_end]
        mask_index = (active_ids == MASK_ID)

        full_logits, _ = model(x, position_offset=0)
        logits = full_logits[:, block_start:block_end]

        active_ids = run_one_step(logits, active_ids, mask_index, num_transfer, step, device)
        x[:, block_start:block_end] = active_ids
        partial = tok.decode(active_ids[0], skip_special_tokens=False)
        print(f"  step {step:2d}: {partial!r}")

    return x[:, block_start:block_end]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weight-dir", default="weights")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--task", default="mmlu_pro")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--item-idx", type=int, default=4)
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

    gen_kwargs = dict(gen_length=args.gen_length, steps=args.steps, block_length=args.block_length, tok=tok)
    trace_block0_cached(model, prompt_ids, **gen_kwargs)
    trace_block0_dense(model, prompt_ids, **gen_kwargs)


if __name__ == "__main__":
    main()
