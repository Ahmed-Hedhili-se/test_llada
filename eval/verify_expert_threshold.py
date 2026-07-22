import sys
import os
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_update.model import LLaDAMoEKV, SMALL_CFG

def verify_threshold_dispatch():
    print("=== Verifying Option A: Expert Threshold Dispatch Reduction ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LLaDAMoEKV(SMALL_CFG).to(device)
    model.eval()

    # Track total tokens processed across all expert MLPs
    token_counts = {}

    def make_hook(name):
        def hook(module, input, output):
            inp = input[0] # [num_tokens, H]
            token_counts[name] += inp.shape[0]
        return hook

    hooks = []
    for i, layer in enumerate(model.layers):
        for expert_idx, expert in enumerate(layer.mlp.experts):
            name = f"L{i}_E{expert_idx}"
            token_counts[name] = 0
            hooks.append(expert.register_forward_hook(make_hook(name)))

    x = torch.randint(0, SMALL_CFG.VS, (1, 16)).to(device)
    
    # Run 1: expert_threshold = 0.0
    for name in token_counts:
        token_counts[name] = 0
    with torch.no_grad():
        _ = model(x, dynamic_k=4, expert_threshold=0.0)
    total_tokens_t0 = sum(token_counts.values())

    # Run 2: expert_threshold = 0.1
    for name in token_counts:
        token_counts[name] = 0
    with torch.no_grad():
        _ = model(x, dynamic_k=4, expert_threshold=0.1)
    total_tokens_t01 = sum(token_counts.values())

    for h in hooks:
        h.remove()

    print(f"Total Expert Token Evaluations at threshold=0.0: {total_tokens_t0}")
    print(f"Total Expert Token Evaluations at threshold=0.1: {total_tokens_t01}")
    reduction = (1.0 - total_tokens_t01 / max(total_tokens_t0, 1)) * 100
    print(f"Token dispatch reduction: {reduction:.2f}%")
    assert total_tokens_t01 < total_tokens_t0, "Option A failed! Token counts did not decrease with threshold."
    print("SUCCESS: Option A verified! Thresholding now actively skips compute dispatch.")

if __name__ == "__main__":
    verify_threshold_dispatch()
