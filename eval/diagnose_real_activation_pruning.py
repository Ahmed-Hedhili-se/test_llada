import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

workspace_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(workspace_root))

from model_update.generate import generate_cached


def run_real_activation_diagnostic():
    parser = argparse.ArgumentParser(description="Real Activation Expert Pruning Diagnostic")
    parser.add_argument("--weight-dir", type=str, default="weights")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--gen-length", type=int, default=64)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--base-k", type=int, default=8)
    parser.add_argument("--min-k", type=int, default=4)
    parser.add_argument("--expert-threshold", type=float, default=0.03)
    parser.add_argument("--max-threshold", type=float, default=0.05)
    args = parser.parse_args()

    print("================================================================")
    print(" Real-Activation Expert Pruning & Zero-Expert Diagnostic")
    print("================================================================")
    print(f" Device          : {args.device}")
    print(f" Weight Dir      : {args.weight_dir}")
    print(f" Dynamic K Ramp  : min_k={args.min_k} -> base_k={args.base_k}")
    print(f" Threshold Ramp  : max_thresh={args.max_threshold} -> min_thresh={args.expert_threshold}")
    print("================================================================\n")

    if not os.path.exists(args.weight_dir) or not os.path.isdir(args.weight_dir):
        print(f"Error: Weight directory '{args.weight_dir}' not found.")
        sys.exit(1)

    print("Loading model and tokenizer...")
    tok = AutoTokenizer.from_pretrained(args.weight_dir, trust_remote_code=True)
    from model_update.model import LLaDAMoEKV, FULL_CFG
    from src.model import load_weights

    model = LLaDAMoEKV(FULL_CFG).to(torch.bfloat16).to(args.device).eval()
    try:
        load_weights(model, args.weight_dir, verbose=False)
        print("Weights loaded successfully.")
    except Exception as e:
        print(f"Warning: Failed to load weights: {e}")

    prompt = "The chemical symbol for gold is Au and for silver is"
    prompt_ids = tok(prompt, return_tensors="pt")["input_ids"].to(args.device)

    # Statistics accumulators
    stats_per_layer = {}

    def make_moe_hook(layer_idx):
        def hook(module, input, output):
            x_in = input[0]
            B_in, T_in, _ = x_in.shape
            x_flat = x_in.view(B_in * T_in, module.cfg.H)
            
            # Retrieve routing weights and top-k selection
            rw = F.softmax(module.gate(x_flat), dim=-1, dtype=torch.float32)
            # Find current dynamic_k from threshold
            # Since module doesn't store step, we inspect actual non-zero counts
            # or calculate from current forward pass
            k = module.cfg.TOPK # Default top-k fallback or inspect active
            rw_top, sel = torch.topk(rw, k, dim=-1)
            
            # Apply thresholding
            keep = rw_top > args.expert_threshold
            one_hot = F.one_hot(sel, num_classes=module.cfg.NE) * keep.unsqueeze(-1)
            expert_mask = one_hot.permute(2, 1, 0)
            
            # Count experts per token
            experts_per_token = expert_mask.sum(dim=(0, 1)) # shape [B*T]
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

    hooks = [layer.mlp.register_forward_hook(make_moe_hook(i)) for i, layer in enumerate(model.layers)]

    print("\nRunning generation with dynamic experts...")
    with torch.no_grad():
        out = generate_cached(
            model=model,
            prompt_ids=prompt_ids,
            gen_length=args.gen_length,
            steps=args.steps,
            block_length=args.block_length,
            use_dynamic_experts=True,
            base_k=args.base_k,
            min_k=args.min_k,
            expert_threshold=args.expert_threshold,
            max_threshold=args.max_threshold,
        )

    for h in hooks:
        h.remove()

    print("\n================ DIAGNOSTIC REPORT ON REAL ACTIVATIONS ================")
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
        
        print(f"Layer {l_idx:02d}: Avg Experts/Token = {layer_avg_e:.2f} | Zero-Expert Tokens = {layer_zero}/{layer_toks} ({(layer_zero/layer_toks)*100:.4f}%)")

    overall_avg_experts = sum(all_avg_experts) / len(all_avg_experts)
    overall_zero_pct = (all_zero_tokens / max(all_total_tokens, 1)) * 100

    print("----------------------------------------------------------------------")
    print(f"OVERALL MEAN EXPERTS PER TOKEN : {overall_avg_experts:.2f} / {args.base_k}")
    print(f"TOTAL ZERO-EXPERT TOKENS       : {all_zero_tokens}/{all_total_tokens} ({overall_zero_pct:.4f}%)")
    print("======================================================================\n")

    decoded = tok.decode(out[0], skip_special_tokens=True)
    print(f"Generated text sample:\n{decoded}\n")

if __name__ == "__main__":
    run_real_activation_diagnostic()
