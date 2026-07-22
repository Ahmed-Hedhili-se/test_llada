import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

workspace_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(workspace_root))

from model_update.generate import generate_cached


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_diagnostic():
    parser = argparse.ArgumentParser(description="Token-level correctness diagnostic for Dynamic Experts")
    parser.add_argument("--weight-dir", type=str, default="weights")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--gen-length", type=int, default=128)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--num-trials", type=int, default=5)
    parser.add_argument("--base-k", type=int, default=8)
    parser.add_argument("--min-k", type=int, default=4)
    parser.add_argument("--expert-threshold", type=float, default=0.03)
    args = parser.parse_args()

    print("================================================================")
    print(" Token-Level Correctness Diagnostic (Dynamic Experts vs Baseline)")
    print("================================================================")
    print(f" Device          : {args.device}")
    print(f" Generation L    : {args.gen_length}")
    print(f" Block Length    : {args.block_length}")
    print(f" Total Steps     : {args.steps}")
    print(f" Dynamic K Ramp  : min_k={args.min_k} -> base_k={args.base_k}")
    print(f" Expert Threshold: {args.expert_threshold}")
    print(f" Number of Trials: {args.num_trials}")
    print("================================================================\n")

    # Try loading real model/tokenizer if weights exist, else fallback to SMALL_CFG
    if os.path.exists(args.weight_dir) and os.path.isdir(args.weight_dir):
        print(f"Loading weights from {args.weight_dir}...")
        from transformers import AutoTokenizer
        from model_update.model import LLaDAMoEKV, FULL_CFG
        from src.model import load_weights

        tok = AutoTokenizer.from_pretrained(args.weight_dir, trust_remote_code=True)
        model = LLaDAMoEKV(FULL_CFG).to(torch.bfloat16).to(args.device).eval()
        try:
            load_weights(model, args.weight_dir, verbose=False)
        except Exception as e:
            print(f"Warning: Failed to load weights ({e}), using random weights.")
        vocab_size = FULL_CFG.VS
        prompts = [
            "The chemical symbol for gold is Au and for silver is",
            "What is the derivative of x^2 with respect to x?",
            "Write a python function to compute fibonacci numbers:",
            "Solve the math problem: 15 * 12 + 45 =",
            "The capital of France is Paris, while the capital of Germany is",
        ]
    else:
        print("No weight directory found. Using SMALL_CFG model for local diagnostic...")
        from model_update.model import LLaDAMoEKV, SMALL_CFG
        model = LLaDAMoEKV(SMALL_CFG).to(args.device).eval()
        vocab_size = SMALL_CFG.VS
        prompts = None

    num_blocks = args.gen_length // args.block_length
    trial_divergence_rates = []

    for trial in range(args.num_trials):
        seed = 42 + trial * 100
        if prompts and trial < len(prompts):
            prompt_text = prompts[trial]
            prompt_ids = tok(prompt_text, return_tensors="pt")["input_ids"].to(args.device)
        else:
            prompt_ids = torch.randint(0, vocab_size, (1, 16), device=args.device)

        # 1. Baseline (use_dynamic_experts=False)
        set_seed(seed)
        out_dense = generate_cached(
            model=model,
            prompt_ids=prompt_ids,
            gen_length=args.gen_length,
            steps=args.steps,
            block_length=args.block_length,
            use_dynamic_experts=False,
        )

        # 2. Dynamic Experts (use_dynamic_experts=True)
        set_seed(seed)
        out_dyn = generate_cached(
            model=model,
            prompt_ids=prompt_ids,
            gen_length=args.gen_length,
            steps=args.steps,
            block_length=args.block_length,
            use_dynamic_experts=True,
            base_k=args.base_k,
            min_k=args.min_k,
            expert_threshold=args.expert_threshold,
        )

        dense_tokens = out_dense[0].cpu()
        dyn_tokens = out_dyn[0].cpu()

        diff_mask = (dense_tokens != dyn_tokens)
        num_diff = diff_mask.sum().item()
        total_toks = len(dense_tokens)
        pct_diff = (num_diff / total_toks) * 100.0
        trial_divergence_rates.append(pct_diff)

        # Breakdown by block
        block_diffs = []
        for b in range(num_blocks):
            b_start = b * args.block_length
            b_end = (b + 1) * args.block_length
            b_diff = diff_mask[b_start:b_end].sum().item()
            block_diffs.append(b_diff)

        print(f"Trial {trial+1}/{args.num_trials} (Seed {seed}):")
        print(f"  Token Divergence : {num_diff}/{total_toks} ({pct_diff:.2f}%)")
        print(f"  Diffs per Block  : {block_diffs} (Block length = {args.block_length})")
        print("----------------------------------------------------------------")

    avg_div = sum(trial_divergence_rates) / len(trial_divergence_rates)
    max_div = max(trial_divergence_rates)
    print("\n================ DIAGNOSTIC SUMMARY ================")
    print(f" Average Token Divergence Rate : {avg_div:.2f}%")
    print(f" Maximum Token Divergence Rate : {max_div:.2f}%")
    print("====================================================")


if __name__ == "__main__":
    run_diagnostic()
