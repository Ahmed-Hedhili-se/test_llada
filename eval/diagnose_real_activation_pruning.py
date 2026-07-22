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
    parser = argparse.ArgumentParser(description="Real Activation Routing Diagnostic")
    parser.add_argument("--weight-dir", type=str, default="weights")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--gen-length", type=int, default=64)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--base-k", type=int, default=8)
    parser.add_argument("--min-k", type=int, default=5)
    args = parser.parse_args()

    print("================================================================")
    print(" Real-Activation Routing Diagnostic")
    print("================================================================")
    print(f" Device          : {args.device}")
    print(f" Weight Dir      : {args.weight_dir}")
    print(f" Dynamic K Ramp  : min_k={args.min_k} -> base_k={args.base_k}")
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
            k = module.cfg.TOPK
            rw_top, sel = torch.topk(rw, k, dim=-1)
            
            # Measure routing weight distribution (no thresholding)
            avg_top1_weight = rw_top[:, 0].mean().item()
            avg_topk_weight = rw_top.mean().item()
            routing_entropy = -(rw * (rw + 1e-10).log()).sum(dim=-1).mean().item()
            total_tokens = B_in * T_in
            
            if layer_idx not in stats_per_layer:
                stats_per_layer[layer_idx] = []
            stats_per_layer[layer_idx].append({
                "total_tokens": total_tokens,
                "avg_top1_weight": avg_top1_weight,
                "avg_topk_weight": avg_topk_weight,
                "routing_entropy": routing_entropy,
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
        )

    for h in hooks:
        h.remove()

    print("\n================ DIAGNOSTIC REPORT ON REAL ACTIVATIONS ================")
    all_avg_top1 = []
    all_avg_topk = []
    all_entropy = []

    for l_idx in sorted(stats_per_layer.keys()):
        records = stats_per_layer[l_idx]
        layer_avg_top1 = sum(r["avg_top1_weight"] for r in records) / len(records)
        layer_avg_topk = sum(r["avg_topk_weight"] for r in records) / len(records)
        layer_entropy = sum(r["routing_entropy"] for r in records) / len(records)
        
        all_avg_top1.append(layer_avg_top1)
        all_avg_topk.append(layer_avg_topk)
        all_entropy.append(layer_entropy)
        
        print(f"Layer {l_idx:02d}: Avg Top-1 Wt = {layer_avg_top1:.4f} | Avg Top-k Wt = {layer_avg_topk:.4f} | Routing Entropy = {layer_entropy:.4f}")

    overall_top1 = sum(all_avg_top1) / len(all_avg_top1)
    overall_topk = sum(all_avg_topk) / len(all_avg_topk)
    overall_entropy = sum(all_entropy) / len(all_entropy)

    print("----------------------------------------------------------------------")
    print(f"OVERALL AVG TOP-1 WEIGHT    : {overall_top1:.4f}")
    print(f"OVERALL AVG TOP-K WEIGHT    : {overall_topk:.4f}")
    print(f"OVERALL ROUTING ENTROPY     : {overall_entropy:.4f}")
    print("======================================================================\n")

    decoded = tok.decode(out[0], skip_special_tokens=True)
    print(f"Generated text sample:\n{decoded}\n")

if __name__ == "__main__":
    run_real_activation_diagnostic()
