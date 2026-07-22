import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

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
    parser = argparse.ArgumentParser(description="Token-level correctness & real activation diagnostic for Dynamic Experts")
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
    print(" Diagnostic: Token Divergence & Real Activation Pruning")
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
            print("Weights loaded successfully.")
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

    # Real activation pruning statistics hooks
    stats_per_layer = {}

    def make_moe_hook(layer_idx):
        def hook(module, input, output):
            x_in = input[0]
            B_in, T_in, _ = x_in.shape
            x_flat = x_in.view(B_in * T_in, module.cfg.H)
            
            rw = F.softmax(module.gate(x_flat), dim=-1, dtype=torch.float32)
            k = module.cfg.TOPK
            rw_top, sel = torch.topk(rw, k, dim=-1)
            
            keep = rw_top > args.expert_threshold
            one_hot = F.one_hot(sel, num_classes=module.cfg.NE) * keep.unsqueeze(-1)
            expert_mask = one_hot.permute(2, 1, 0)
            
            experts_per_token = expert_mask.sum(dim=(0, 1))
            zero_expert_tokens = (experts_per_token == 0).sum().item()
            avg_experts = experts_per_token.float().mean().item()
            total_tokens = B_in * T_in
            
            if layer_idx not in stats_per_layer:
                stats_per_layer[layer_idx] = []
            stats_per_layer[layer_idx].append({
                "total_tokens": total_tokens,
                "zero_expert_tokens": zero_expert_tokens,
                "avg_experts": avg_experts,
            })
        return hook

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

        # 2. Dynamic Experts (use_dynamic_experts=True) with activation hooks
        hooks = [layer.mlp.register_forward_hook(make_moe_hook(i)) for i, layer in enumerate(model.layers)]
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
        for h in hooks:
            h.remove()

        dense_tokens = out_dense[0].cpu()
        dyn_tokens = out_dyn[0].cpu()

        diff_mask = (dense_tokens != dyn_tokens)
        num_diff = diff_mask.sum().item()
        total_toks = len(dense_tokens)
        pct_diff = (num_diff / total_toks) * 100.0
        trial_divergence_rates.append(pct_diff)

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

    print("\n================ DIAGNOSTIC 1: TOKEN DIVERGENCE SUMMARY ================")
    print(f" Average Token Divergence Rate : {avg_div:.2f}%")
    print(f" Maximum Token Divergence Rate : {max_div:.2f}%")
    print("=========================================================================")

    if stats_per_layer:
        print("\n================ DIAGNOSTIC 2: REAL ACTIVATION PRUNING REPORT ===========")
        all_zero_tokens = 0
        all_total_tokens = 0
        all_avg_experts = []

        for l_idx in sorted(stats_per_layer.keys()):
            records = stats_per_layer[l_idx]
            layer_zero = sum(r["zero_expert_tokens"] for r in records)
            layer_toks = sum(r["total_tokens"] for r in records)
            layer_avg_e = sum(r["avg_experts"] for r in records) / len(records)
            
            all_zero_tokens += layer_zero
            all_total_tokens += layer_toks
            all_avg_experts.append(layer_avg_e)
            
            print(f"Layer {l_idx:02d}: Avg Experts/Token = {layer_avg_e:.2f} | Zero-Expert Tokens = {layer_zero}/{layer_toks} ({(layer_zero/max(layer_toks,1))*100:.4f}%)")

        overall_avg_experts = sum(all_avg_experts) / len(all_avg_experts)
        overall_zero_pct = (all_zero_tokens / max(all_total_tokens, 1)) * 100

        print("-------------------------------------------------------------------------")
        print(f"OVERALL MEAN EXPERTS PER TOKEN : {overall_avg_experts:.2f} / {args.base_k}")
        print(f"TOTAL ZERO-EXPERT TOKENS       : {all_zero_tokens}/{all_total_tokens} ({overall_zero_pct:.4f}%)")
        print("=========================================================================\n")


if __name__ == "__main__":
    run_diagnostic()
