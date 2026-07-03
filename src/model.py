"""
Self-contained LLaDA-MoE-7B-A1B implementation in pure PyTorch.

Architecture (from inclusionAI/LLaDA-MoE-7B-A1B-Instruct config.json):
  - 16 transformer layers, all MoE (moe_layer_freq all-1)
  - hidden_size=2048, num_heads=16 (head_dim=128), MHA (no GQA)
  - 64 experts per layer, top-8 routing, softmax scores, no renorm
  - expert_intermediate_size=1024, shared_expert=None
  - QK RMSNorm per head, RoPE(theta=50000), full rotary (factor=1)
  - Bidirectional (non-causal) attention — diffusion LM, not AR
  - vocab_size=157184, mask_id=156895
  - No KV cache; every forward pass sees the full sequence

Weight loading mirrors HF key names exactly so no translation is needed
beyond stripping the top-level "model." prefix.
"""

import json
import math
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open

# ── Model constants (from config.json) ───────────────────────────────────────
H      = 2048        # hidden_size
NH     = 16          # num_attention_heads
KVH    = 16          # num_key_value_heads  (MHA: KVH == NH)
HD     = H // NH     # head_dim = 128
NL     = 16          # num_hidden_layers
NE     = 64          # num_experts
TOPK   = 8           # num_experts_per_tok
EI     = 1024        # expert_intermediate_size
VS     = 157184      # vocab_size
EPS    = 1e-5        # rms_norm_eps
THETA  = 50000.0     # rope_theta
MASK_ID = 156895     # token used for masking in diffusion


# ── RMSNorm ───────────────────────────────────────────────────────────────────
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


# ── RoPE ──────────────────────────────────────────────────────────────────────
def build_rope_freqs(max_seq: int, head_dim: int, theta: float, device) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    pos = torch.arange(max_seq, device=device).float()
    freqs = torch.outer(pos, inv_freq)          # [max_seq, head_dim/2]
    emb = torch.cat([freqs, freqs], dim=-1)     # [max_seq, head_dim]
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    # q/k: [B, NH, T, HD]; cos/sin: [T, HD]
    cos = cos.unsqueeze(0).unsqueeze(0)   # [1,1,T,HD]
    sin = sin.unsqueeze(0).unsqueeze(0)
    q = q * cos + rotate_half(q) * sin
    k = k * cos + rotate_half(k) * sin
    return q, k


# ── Attention ─────────────────────────────────────────────────────────────────
class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(H, NH * HD, bias=False)
        self.k_proj = nn.Linear(H, KVH * HD, bias=False)
        self.v_proj = nn.Linear(H, KVH * HD, bias=False)
        self.o_proj = nn.Linear(H, H, bias=False)
        # Per-head QK RMSNorm (qk_layernorm=True)
        self.q_norm = RMSNorm(HD)
        self.k_norm = RMSNorm(HD)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, NH, HD)
        k = self.k_proj(x).view(B, T, KVH, HD)
        v = self.v_proj(x).view(B, T, KVH, HD)

        # Per-head RMSNorm before RoPE
        q = self.q_norm(q.reshape(-1, HD)).reshape(B, T, NH, HD)
        k = self.k_norm(k.reshape(-1, HD)).reshape(B, T, KVH, HD)

        # [B, NH, T, HD]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q, k = apply_rope(q, k, cos, sin)

        # Full bidirectional attention (is_causal=False, no mask)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=False)
        out = out.transpose(1, 2).reshape(B, T, H)
        return self.o_proj(out)


# ── Expert MLP ────────────────────────────────────────────────────────────────
class ExpertMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(H, EI, bias=False)
        self.up_proj   = nn.Linear(H, EI, bias=False)
        self.down_proj = nn.Linear(EI, H, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ── MoE block ─────────────────────────────────────────────────────────────────
class MoEBlock(nn.Module):
    """
    Top-8 softmax routing over 64 experts. No renorm, no shared expert.
    Mirrors LLaDAMoESparseMoeBlock from the HF implementation exactly.
    """
    def __init__(self):
        super().__init__()
        self.gate    = nn.Linear(H, NE, bias=False)
        self.experts = nn.ModuleList([ExpertMLP() for _ in range(NE)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        x_flat = x.view(B * T, H)                                  # [S, H]

        router_logits  = self.gate(x_flat)                          # [S, NE]
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)  # [S, NE]
        routing_weights, selected_experts = torch.topk(routing_weights, TOPK, dim=-1)  # [S, K]
        routing_weights = routing_weights.to(x.dtype)

        out = torch.zeros_like(x_flat)                              # [S, H]
        # expert_mask: [NE, K, S]
        expert_mask = F.one_hot(selected_experts, num_classes=NE).permute(2, 1, 0)

        for expert_idx in range(NE):
            expert = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])      # which slot, which token
            if top_x.numel() == 0:
                continue
            tokens = x_flat[top_x]                                  # [n, H]
            h = expert(tokens) * routing_weights[top_x, idx, None]  # weighted
            out.index_add_(0, top_x, h.to(x.dtype))

        return out.view(B, T, H)


# ── Transformer layer ─────────────────────────────────────────────────────────
class Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layernorm        = RMSNorm(H)
        self.self_attn              = Attention()
        self.post_attention_layernorm = RMSNorm(H)
        self.mlp                    = MoEBlock()

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


# ── Full model ────────────────────────────────────────────────────────────────
class LLaDAMoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(VS, H)
        self.layers       = nn.ModuleList([Layer() for _ in range(NL)])
        self.norm         = RMSNorm(H)
        self.lm_head      = nn.Linear(H, VS, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: [B, T] long tensor (may contain MASK_ID tokens)
        Returns:
            logits: [B, T, VS]
        """
        B, T = input_ids.shape
        device = input_ids.device

        x = self.embed_tokens(input_ids)        # [B, T, H]

        cos, sin = build_rope_freqs(T, HD, THETA, device)
        cos = cos.to(x.dtype)
        sin = sin.to(x.dtype)

        for layer in self.layers:
            x = layer(x, cos, sin)

        x = self.norm(x)
        return self.lm_head(x)                  # [B, T, VS]


# ── Weight loading ────────────────────────────────────────────────────────────
def _hf_to_our_key(hk: str) -> Optional[str]:
    """Strip HF 'model.' prefix; skip non-backbone keys."""
    if hk.startswith("model."):
        return hk[len("model."):]
    return None


def load_weights(model: LLaDAMoE, weight_dir: str, verbose: bool = True) -> LLaDAMoE:
    index_path = os.path.join(weight_dir, "model.safetensors.index.json")
    with open(index_path) as f:
        wmap = json.load(f)["weight_map"]

    # Group HF keys by shard file
    shards: dict[str, list[str]] = {}
    for hk, shard in wmap.items():
        shards.setdefault(shard, []).append(hk)

    sd = model.state_dict()
    mapped, mismatches = 0, []

    for shard_name in sorted(shards):
        path = os.path.join(weight_dir, shard_name)
        f = safe_open(path, framework="pt", device="cpu")
        for hk in shards[shard_name]:
            mk = _hf_to_our_key(hk)
            if mk is None:
                continue
            if mk not in sd:
                mismatches.append(f"missing in our model: {hk} → {mk}")
                continue
            t = f.get_tensor(hk)
            if t.shape != sd[mk].shape:
                mismatches.append(f"shape mismatch {hk}: hf={t.shape} ours={sd[mk].shape}")
                continue
            sd[mk] = t.to(sd[mk].dtype)
            mapped += 1

    if mismatches:
        print(f"  Issues ({len(mismatches)}):")
        for m in mismatches[:20]:
            print(f"    {m}")
    if verbose:
        total = sum(1 for k in wmap if _hf_to_our_key(k) is not None)
        print(f"  Mapped {mapped}/{total} tensors.")

    model.load_state_dict(sd, strict=False)
    return model
