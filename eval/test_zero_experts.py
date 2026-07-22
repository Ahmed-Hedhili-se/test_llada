import sys
import os
import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_update.model import LLaDAMoEKV, SMALL_CFG

def run_zero_experts_check():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LLaDAMoEKV(SMALL_CFG).to(device)
    model.eval()

    x = torch.randint(0, SMALL_CFG.VS, (1, 16)).to(device)
    
    print("Testing with expert_threshold = 0.05:")
    with torch.no_grad():
        # Pass through layer 0's MoE block directly
        x_flat = model.embed_tokens(x).view(16, SMALL_CFG.H)
        moe = model.layers[0].mlp
        
        routing_weights = F.softmax(moe.gate(x_flat), dim=-1, dtype=torch.float32)
        routing_weights, selected_experts = torch.topk(routing_weights, 4, dim=-1)
        
        # Apply threshold = 0.05
        keep = routing_weights > 0.05
        one_hot = F.one_hot(selected_experts, num_classes=SMALL_CFG.NE) * keep.unsqueeze(-1)
        expert_mask = one_hot.permute(2, 1, 0)

        T = x_flat.shape[0]
        tokens_with_zero_experts = (expert_mask.sum(dim=(0, 1)) == 0).sum().item()
        print(f"{tokens_with_zero_experts}/{T} tokens got NO experts this call")

    print("\nTesting with expert_threshold = 0.15 (higher threshold):")
    with torch.no_grad():
        keep_high = routing_weights > 0.15
        one_hot_high = F.one_hot(selected_experts, num_classes=SMALL_CFG.NE) * keep_high.unsqueeze(-1)
        expert_mask_high = one_hot_high.permute(2, 1, 0)

        tokens_with_zero_experts_high = (expert_mask_high.sum(dim=(0, 1)) == 0).sum().item()
        print(f"{tokens_with_zero_experts_high}/{T} tokens got NO experts this call")

if __name__ == "__main__":
    run_zero_experts_check()
