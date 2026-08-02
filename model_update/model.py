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
from typing import List, Optional, Tuple, Literal, Union
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib.util
import site
from pathlib import Path
import torch.distributed as dist
from .distributed import get_tp_size, get_tp_rank, get_tp_group
import re



def _load_fused_moe():
    return None

fused_moe = None
KVCache = List[Tuple[torch.Tensor, torch.Tensor]]


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


from .fused_moe_triton import fused_moe

class TritonFusedMoEBlock(nn.Module):
    def __init__(self, cfg: Cfg):
        super().__init__()
        self.cfg = cfg
        self.tp_size = get_tp_size()
        self.tp_rank = get_tp_rank()
        self.num_local_experts = cfg.NE // self.tp_size
        self.local_expert_start = self.tp_rank * self.num_local_experts
        
        self.gate = nn.Linear(cfg.H, cfg.NE, bias=False)
        self.w1 = nn.Parameter(torch.empty(self.num_local_experts, 2 * cfg.EI, cfg.H))
        self.w2 = nn.Parameter(torch.empty(self.num_local_experts, cfg.H, cfg.EI))

    def load_state_dict_from_unfused(self, unfused_block: nn.Module):
        with torch.no_grad():
            self.gate.weight.copy_(unfused_block.gate.weight)
            for i in range(self.num_local_experts):
                expert = unfused_block.experts[i]  # already sliced in MoEBlock
                w_gate = expert.gate_proj.weight
                w_up = expert.up_proj.weight
                self.w1[i].copy_(torch.cat([w_gate, w_up], dim=0))
                self.w2[i].copy_(expert.down_proj.weight)

    def forward(self, x: torch.Tensor, dynamic_k: Optional[int] = None) -> torch.Tensor:
        if x.device.type == "cpu":
            raise NotImplementedError("Triton Fused MoE requires a CUDA GPU.")

        B, T, H = x.shape
        x_flat = x.view(B * T, H)
        router_logits = self.gate(x_flat)
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)

        k = dynamic_k if dynamic_k is not None else self.cfg.TOPK
        topk_weights, topk_ids = torch.topk(routing_weights, k, dim=-1)

        if self.tp_size > 1:
            local_expert_end = self.local_expert_start + self.num_local_experts
            mask = (topk_ids >= self.local_expert_start) & (topk_ids < local_expert_end)
            topk_ids = (topk_ids - self.local_expert_start).clamp(min=0, max=self.num_local_experts - 1)
            topk_weights = topk_weights * mask.float()

        out = fused_moe(
            hidden_states=x_flat,
            w1=self.w1,
            w2=self.w2,
            gating_output=topk_weights.to(x.dtype),
            topk_ids=topk_ids.to(torch.int32),
        )
        
        out = out.view(B, T, H)
        if self.tp_size > 1:
            dist.all_reduce(out, op=dist.ReduceOp.SUM, group=get_tp_group())
            
        return out



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


def build_rope_freqs(start_pos: int, end_pos: int, head_dim: int, theta: float, device):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    pos = torch.arange(start_pos, end_pos, device=device).float()
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
        
        tp_size = get_tp_size()
        tp_rank = get_tp_rank()
        
        assert NH % tp_size == 0, f"NH ({NH}) must be divisible by tp_size ({tp_size})"
        assert KVH % tp_size == 0, f"KVH ({KVH}) must be divisible by tp_size ({tp_size})"
        
        self.NH_local = NH // tp_size
        self.KVH_local = KVH // tp_size
        
        self.q_proj = nn.Linear(H, self.NH_local * HD, bias=False)
        self.k_proj = nn.Linear(H, self.KVH_local * HD, bias=False)
        self.v_proj = nn.Linear(H, self.KVH_local * HD, bias=False)
        self.o_proj = nn.Linear(self.NH_local * HD, H, bias=False)
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

        q = self.q_proj(x).view(B, Ta, self.NH_local, cfg.HD)
        k = self.k_proj(x).view(B, Ta, self.KVH_local, cfg.HD)
        v = self.v_proj(x).view(B, Ta, self.KVH_local, cfg.HD)

        q = self.q_norm(q.reshape(-1, cfg.HD)).reshape(B, Ta, self.NH_local, cfg.HD)
        k = self.k_norm(k.reshape(-1, cfg.HD)).reshape(B, Ta, self.KVH_local, cfg.HD)

        q = q.transpose(1, 2) 
        k = k.transpose(1, 2)  
        v = v.transpose(1, 2)

        q, k = apply_rope(q, k, cos, sin)

        if past_kv is not None:
            k_prefix, v_prefix = past_kv
            k_full = torch.cat([k_prefix, k], dim=2)
            v_full = torch.cat([v_prefix, v], dim=2)
        else:
            k_full, v_full = k, v

        out = F.scaled_dot_product_attention(q, k_full, v_full, attn_mask=None, is_causal=False)
        out = out.transpose(1, 2).reshape(B, Ta, self.NH_local * cfg.HD)
        
        attn_out = self.o_proj(out)
        if get_tp_size() > 1:
            dist.all_reduce(attn_out, op=dist.ReduceOp.SUM, group=get_tp_group())
            
        return attn_out, (k, v)


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
        self.tp_size = get_tp_size()
        self.tp_rank = get_tp_rank()
        assert cfg.NE % self.tp_size == 0
        self.num_local_experts = cfg.NE // self.tp_size
        self.local_expert_start = self.tp_rank * self.num_local_experts
        
        self.gate = nn.Linear(cfg.H, cfg.NE, bias=False)
        self.experts = nn.ModuleList([ExpertMLP(cfg) for _ in range(self.num_local_experts)])

    def forward(self, x, dynamic_k: Optional[int] = None):
        cfg = self.cfg
        B, T, _ = x.shape
        x_flat = x.view(B * T, cfg.H)

        routing_weights = F.softmax(self.gate(x_flat), dim=-1, dtype=torch.float32)
        k = dynamic_k if dynamic_k is not None else cfg.TOPK
        routing_weights, selected_experts = torch.topk(routing_weights, k, dim=-1)
        expert_mask = F.one_hot(selected_experts, num_classes=cfg.NE).permute(2, 1, 0)

        routing_weights = routing_weights.to(x.dtype)

        out = torch.zeros_like(x_flat)

        for local_expert_idx in range(self.num_local_experts):
            global_expert_idx = self.local_expert_start + local_expert_idx
            idx, top_x = torch.where(expert_mask[global_expert_idx])
            if top_x.numel() == 0:
                continue
            tokens = x_flat[top_x]
            h = self.experts[local_expert_idx](tokens) * routing_weights[top_x, idx, None]
            out.index_add_(0, top_x, h.to(x.dtype))

        out = out.view(B, T, cfg.H)
        if self.tp_size > 1:
            dist.all_reduce(out, op=dist.ReduceOp.SUM, group=get_tp_group())
            
        return out


class Layer(nn.Module):
    def __init__(self, cfg: Cfg, use_fused_moe: bool = False):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.H, cfg.EPS)
        self.self_attn = Attention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.H, cfg.EPS)
        self.mlp = TritonFusedMoEBlock(cfg) if use_fused_moe else MoEBlock(cfg)

    def forward(self, x, cos, sin, past_kv=None, dynamic_k: Optional[int] = None):
        attn_out, kv_new = self.self_attn(self.input_layernorm(x), cos, sin, past_kv)
        x = x + attn_out
        x = x + self.mlp(self.post_attention_layernorm(x), dynamic_k=dynamic_k)
        return x, kv_new


class LLaDAMoEKV(nn.Module):
    def __init__(self, cfg: Cfg = FULL_CFG, use_fused_moe: bool = False):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.VS, cfg.H)
        self.layers = nn.ModuleList([Layer(cfg, use_fused_moe=use_fused_moe) for _ in range(cfg.NL)])
        self.norm = RMSNorm(cfg.H, cfg.EPS)
        self.lm_head = nn.Linear(cfg.H, cfg.VS, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_offset: int = 0,
        past_kv: Optional[KVCache] = None,
        dynamic_k: Optional[int] = None,
    ):
        """
        input_ids: [B, T] — either the full sequence (past_kv=None, e.g. prefix
                   priming) or just the active block (past_kv=cache).
        position_offset: absolute starting position of input_ids in the full
                   sequence, for correct RoPE.
        past_kv: list of (k,v) per layer for the prefix, or None.
        dynamic_k: override the number of activated experts for this call.
        Returns: logits [B, T, VS] for input_ids positions only, and
                 new_kv: list of (k,v) per layer for input_ids' own tokens
                 (caller decides whether/when to append these to the cache).
        """
        cfg = self.cfg
        B, T = input_ids.shape
        device = input_ids.device

        x = self.embed_tokens(input_ids)

        cos, sin = build_rope_freqs(position_offset, position_offset + T, cfg.HD, cfg.THETA, device)
        cos = cos.to(x.dtype)
        sin = sin.to(x.dtype)

        new_kv: KVCache = []
        for i, layer in enumerate(self.layers):
            layer_past = past_kv[i] if past_kv is not None else None
            x, kv_i = layer(x, cos, sin, layer_past, dynamic_k=dynamic_k)
            new_kv.append(kv_i)

        x = self.norm(x)
        logits = self.lm_head(x)
        return logits, new_kv

def concat_kv(cache: Optional[KVCache], new_kv: KVCache) -> KVCache:
    if cache is None:
        return new_kv
    out = []
    for (k_old, v_old), (k_new, v_new) in zip(cache, new_kv):
        out.append((torch.cat([k_old, k_new], dim=2), torch.cat([v_old, v_new], dim=2)))
    return out
