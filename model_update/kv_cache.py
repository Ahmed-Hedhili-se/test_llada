"""
LLaDA-MoE with block-wise KV caching.

Key idea:
  - "prefix" tokens (prompt + already-finalized blocks) have FIXED content,
    so their K/V (post-RoPE) never change once computed. Cache them.
  - "active" tokens (current block, still being denoised) get fresh Q/K/V
    computed every step, and attend against [cached prefix K/V ; active K/V].
  - MoE/MLP/norms are all per-token, so we only ever need to run them over
    the active block, not the full sequence.
  - Logits are only computed/returned for the active block, since
    generate.py already discards logits past block_end anyway.

Correctness-critical rule: a block's K/V may only be pushed into the
permanent cache AFTER it is fully unmasked. Caching mid-denoising K/V
would permanently bake in stale (masked-token) representations.

Dims are passed via a Config so the exact same code can be correctness
tested at small scale and then run at full 7B-MoE scale.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

KVCache = List[Tuple[torch.Tensor, torch.Tensor]]  # per-layer (k, v), each [B, KVH, T, HD]


@dataclass
class Cfg:
    H: int
    NH: int
    KVH: int
    NL: int
    NE: int
    TOPK: int
    EI: int
    VS: int
    EPS: float = 1e-5
    THETA: float = 50000.0
    MASK_ID: int = 156895

    @property
    def HD(self):
        return self.H // self.NH


FULL_CFG = Cfg(H=2048, NH=16, KVH=16, NL=16, NE=64, TOPK=8, EI=1024, VS=157184)
SMALL_CFG = Cfg(H=512, NH=8, KVH=8, NL=4, NE=16, TOPK=4, EI=256, VS=157184)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
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


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q, k, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


class Attention(nn.Module):
    def __init__(self, cfg: Cfg):
        super().__init__()
        self.cfg = cfg
        H, NH, KVH, HD = cfg.H, cfg.NH, cfg.KVH, cfg.HD
        self.q_proj = nn.Linear(H, NH * HD, bias=False)
        self.k_proj = nn.Linear(H, KVH * HD, bias=False)
        self.v_proj = nn.Linear(H, KVH * HD, bias=False)
        self.o_proj = nn.Linear(H, H, bias=False)
        self.q_norm = RMSNorm(HD, cfg.EPS)
        self.k_norm = RMSNorm(HD, cfg.EPS)

    def forward(self, x, cos, sin, past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        """
        x: [B, Ta, H] active-block hidden states
        cos/sin: [Ta, HD] rope freqs for the ACTIVE positions (absolute offset already applied)
        past_kv: optional (k_prefix, v_prefix) each [B, KVH, Tp, HD], already RoPE'd
        Returns: out [B, Ta, H], (k_active, v_active) each [B, KVH, Ta, HD] (RoPE'd, for caching)
        """
        cfg = self.cfg
        B, Ta, _ = x.shape

        q = self.q_proj(x).view(B, Ta, cfg.NH, cfg.HD)
        k = self.k_proj(x).view(B, Ta, cfg.KVH, cfg.HD)
        v = self.v_proj(x).view(B, Ta, cfg.KVH, cfg.HD)

        q = self.q_norm(q.reshape(-1, cfg.HD)).reshape(B, Ta, cfg.NH, cfg.HD)
        k = self.k_norm(k.reshape(-1, cfg.HD)).reshape(B, Ta, cfg.KVH, cfg.HD)

        q = q.transpose(1, 2)  # [B, NH, Ta, HD]
        k = k.transpose(1, 2)  # [B, KVH, Ta, HD]
        v = v.transpose(1, 2)

        q, k = apply_rope(q, k, cos, sin)  # k here is the fresh active-block k, RoPE'd

        if past_kv is not None:
            k_prefix, v_prefix = past_kv
            k_full = torch.cat([k_prefix, k], dim=2)
            v_full = torch.cat([v_prefix, v], dim=2)
        else:
            k_full, v_full = k, v

        out = F.scaled_dot_product_attention(q, k_full, v_full, attn_mask=None, is_causal=False)
        out = out.transpose(1, 2).reshape(B, Ta, cfg.H)
        return self.o_proj(out), (k, v)


class ExpertMLP(nn.Module):
    def __init__(self, cfg: Cfg):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.H, cfg.EI, bias=False)
        self.up_proj = nn.Linear(cfg.H, cfg.EI, bias=False)
        self.down_proj = nn.Linear(cfg.EI, cfg.H, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MoEBlock(nn.Module):
    def __init__(self, cfg: Cfg):
        super().__init__()
        self.cfg = cfg
        self.gate = nn.Linear(cfg.H, cfg.NE, bias=False)
        self.experts = nn.ModuleList([ExpertMLP(cfg) for _ in range(cfg.NE)])

    def forward(self, x, dynamic_k: Optional[int] = None, expert_threshold: float = 0.0):
        cfg = self.cfg
        B, T, _ = x.shape
        x_flat = x.view(B * T, cfg.H)

        routing_weights = F.softmax(self.gate(x_flat), dim=-1, dtype=torch.float32)
        k = dynamic_k if dynamic_k is not None else cfg.TOPK
        routing_weights, selected_experts = torch.topk(routing_weights, k, dim=-1)
        
        if expert_threshold > 0:
            keep = routing_weights > expert_threshold
            # Hard Safety Floor: Ensure every token retains at least 1 expert (highest weighted)
            zero_expert_tokens = (keep.sum(dim=-1) == 0)
            if zero_expert_tokens.any():
                keep[zero_expert_tokens, 0] = True

            routing_weights = routing_weights * keep
            sum_w = routing_weights.sum(dim=-1, keepdim=True).clamp(min=1e-9)
            routing_weights = routing_weights / sum_w
            one_hot = F.one_hot(selected_experts, num_classes=cfg.NE) * keep.unsqueeze(-1)
            expert_mask = one_hot.permute(2, 1, 0)
        else:
            expert_mask = F.one_hot(selected_experts, num_classes=cfg.NE).permute(2, 1, 0)

        routing_weights = routing_weights.to(x.dtype)

        out = torch.zeros_like(x_flat)

        for expert_idx in range(cfg.NE):
            idx, top_x = torch.where(expert_mask[expert_idx])
            if top_x.numel() == 0:
                continue
            tokens = x_flat[top_x]
            h = self.experts[expert_idx](tokens) * routing_weights[top_x, idx, None]
            out.index_add_(0, top_x, h.to(x.dtype))

        return out.view(B, T, cfg.H)


class Layer(nn.Module):
    def __init__(self, cfg: Cfg):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.H, cfg.EPS)
        self.self_attn = Attention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.H, cfg.EPS)
        self.mlp = MoEBlock(cfg)

    def forward(self, x, cos, sin, past_kv=None, dynamic_k: Optional[int] = None, expert_threshold: float = 0.0):
        attn_out, kv_new = self.self_attn(self.input_layernorm(x), cos, sin, past_kv)
        x = x + attn_out
        x = x + self.mlp(self.post_attention_layernorm(x), dynamic_k=dynamic_k, expert_threshold=expert_threshold)
        return x, kv_new


class LLaDAMoEKV(nn.Module):
    def __init__(self, cfg: Cfg = FULL_CFG):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.VS, cfg.H)
        self.layers = nn.ModuleList([Layer(cfg) for _ in range(cfg.NL)])
        self.norm = RMSNorm(cfg.H, cfg.EPS)
        self.lm_head = nn.Linear(cfg.H, cfg.VS, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_offset: int = 0,
        past_kv: Optional[KVCache] = None,
        dynamic_k: Optional[int] = None,
        expert_threshold: float = 0.0,
    ):
        """
        input_ids: [B, T] — either the full sequence (past_kv=None, e.g. prefix
                   priming) or just the active block (past_kv=cache).
        position_offset: absolute starting position of input_ids in the full
                   sequence, for correct RoPE.
        past_kv: list of (k,v) per layer for the prefix, or None.
        topk_override: override the number of activated experts for this call.
        Returns: logits [B, T, VS] for input_ids positions only, and
                 new_kv: list of (k,v) per layer for input_ids' own tokens
                 (caller decides whether/when to append these to the cache).
        """
        cfg = self.cfg
        B, T = input_ids.shape
        device = input_ids.device

        x = self.embed_tokens(input_ids)

        cos_full, sin_full = build_rope_freqs(position_offset + T, cfg.HD, cfg.THETA, device)
        cos = cos_full[position_offset: position_offset + T].to(x.dtype)
        sin = sin_full[position_offset: position_offset + T].to(x.dtype)

        new_kv: KVCache = []
        for i, layer in enumerate(self.layers):
            layer_past = past_kv[i] if past_kv is not None else None
            x, kv_i = layer(x, cos, sin, layer_past, dynamic_k=dynamic_k, expert_threshold=expert_threshold)
            new_kv.append(kv_i)

        x = self.norm(x)
        logits = self.lm_head(x)
        return logits, new_kv


def concat_kv(cache: Optional[KVCache], new_kv: KVCache) -> KVCache:
    """Append newly-finalized block K/V onto the running prefix cache."""
    if cache is None:
        return new_kv
    out = []
    for (k_old, v_old), (k_new, v_new) in zip(cache, new_kv):
        out.append((torch.cat([k_old, k_new], dim=2), torch.cat([v_old, v_new], dim=2)))
    return out
