"""
LLaDA-MoE-Small: scaled-down architecture for fast experimentation.
Same structure as model.py — RMSNorm, RoPE, QK-normed bidirectional MHA,
MoE with top-K routing — but with much smaller dims, ~195M parameters.
Initializes with random weights; no weight loading.

Small config vs full:
  H     512  (was 2048)
  NH    8    (was 16)
  NL    4    (was 16)
  NE    16   (was 64)
  TOPK  4    (was 8)
  EI    256  (was 1024)
  vocab 157184 (same)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Small model constants ──────────────────────────────────────────────────────
H      = 512         # hidden_size
NH     = 8           # num_attention_heads
KVH    = 8           # num_key_value_heads (MHA)
HD     = H // NH     # head_dim = 64
NL     = 4           # num_hidden_layers
NE     = 16          # num_experts
TOPK   = 4           # num_experts_per_tok
EI     = 256         # expert_intermediate_size
VS     = 157184      # vocab_size (same as full model)
EPS    = 1e-5
THETA  = 50000.0
MASK_ID = 156895


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = EPS):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * x).to(dtype)


def build_rope_freqs(max_seq: int, head_dim: int, theta: float, device):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    pos = torch.arange(max_seq, device=device).float()
    freqs = torch.outer(pos, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(H, NH * HD, bias=False)
        self.k_proj = nn.Linear(H, KVH * HD, bias=False)
        self.v_proj = nn.Linear(H, KVH * HD, bias=False)
        self.o_proj = nn.Linear(H, H, bias=False)
        self.q_norm = RMSNorm(HD)
        self.k_norm = RMSNorm(HD)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, NH, HD)
        k = self.k_proj(x).view(B, T, KVH, HD)
        v = self.v_proj(x).view(B, T, KVH, HD)
        q = self.q_norm(q.reshape(-1, HD)).reshape(B, T, NH, HD)
        k = self.k_norm(k.reshape(-1, HD)).reshape(B, T, KVH, HD)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        q, k = apply_rope(q, k, cos, sin)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=False)
        return self.o_proj(out.transpose(1, 2).reshape(B, T, H))


class ExpertMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(H, EI, bias=False)
        self.up_proj   = nn.Linear(H, EI, bias=False)
        self.down_proj = nn.Linear(EI, H, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MoEBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate    = nn.Linear(H, NE, bias=False)
        self.experts = nn.ModuleList([ExpertMLP() for _ in range(NE)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        x_flat = x.view(B * T, H)
        routing_weights = F.softmax(self.gate(x_flat), dim=-1, dtype=torch.float32)
        routing_weights, selected_experts = torch.topk(routing_weights, TOPK, dim=-1)
        routing_weights = routing_weights.to(x.dtype)
        out = torch.zeros_like(x_flat)
        expert_mask = F.one_hot(selected_experts, num_classes=NE).permute(2, 1, 0)
        for expert_idx in range(NE):
            idx, top_x = torch.where(expert_mask[expert_idx])
            if top_x.numel() == 0:
                continue
            h = self.experts[expert_idx](x_flat[top_x]) * routing_weights[top_x, idx, None]
            out.index_add_(0, top_x, h.to(x.dtype))
        return out.view(B, T, H)


class Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layernorm          = RMSNorm(H)
        self.self_attn                = Attention()
        self.post_attention_layernorm = RMSNorm(H)
        self.mlp                      = MoEBlock()

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class LLaDAMoESmall(nn.Module):
    """
    LLaDA-MoE-Small: ~195M parameters, same architecture pattern as the 7B model.
    Initializes with random weights (no loading from HF).
    """
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(VS, H)
        self.layers       = nn.ModuleList([Layer() for _ in range(NL)])
        self.norm         = RMSNorm(H)
        self.lm_head      = nn.Linear(H, VS, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        x = self.embed_tokens(input_ids)
        cos, sin = build_rope_freqs(T, HD, THETA, input_ids.device)
        cos = cos.to(x.dtype)
        sin = sin.to(x.dtype)
        for layer in self.layers:
            x = layer(x, cos, sin)
        return self.lm_head(self.norm(x))


if __name__ == "__main__":
    import sys

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = LLaDAMoESmall().to(torch.bfloat16).to(device)
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total / 1e6:.1f}M")
    print(f"  H={H}, NH={NH}, HD={HD}, NL={NL}, NE={NE}, TOPK={TOPK}, EI={EI}")

    # Forward pass: batch=1, seq=32 tokens with MASK at positions 16-31
    ids = torch.full((1, 32), MASK_ID, dtype=torch.long, device=device)
    ids[0, :16] = torch.randint(0, 1000, (16,))

    with torch.no_grad():
        logits = model(ids)

    assert logits.shape == (1, 32, VS), f"Unexpected shape: {logits.shape}"
    print(f"Forward pass OK — logits shape: {logits.shape}")
    print(f"  Top-1 predicted token at position 16: {logits[0, 16].argmax().item()}")
    print("LLaDA-MoE-Small: all checks passed.")
