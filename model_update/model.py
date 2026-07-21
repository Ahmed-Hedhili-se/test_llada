"""
LLaDA-MoE-7B-A1B with dynamic expert pruning (Option A: conservative, accuracy-safe).

Changes from original:
  - MoEBlock.forward() accepts dynamic_k and expert_threshold
  - Layer and LLaDAMoE pass these through to MoEBlock
  - All other code UNCHANGED (attention, RoPE, weight loading identical)
"""

import json
import math
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open

H = 2048; NH = 16; KVH = 16; HD = H // NH; NL = 16; NE = 64; TOPK = 8; EI = 1024
VS = 157184; EPS = 1e-5; THETA = 50000.0; MASK_ID = 156895


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


def build_rope_freqs(max_seq, head_dim, theta, device):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    pos = torch.arange(max_seq, device=device).float()
    freqs = torch.outer(pos, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q, k, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q = q * cos + rotate_half(q) * sin
    k = k * cos + rotate_half(k) * sin
    return q, k


def rope_cos_sin_at(positions, head_dim, theta, device):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    freqs = torch.outer(positions.float(), inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(H, NH * HD, bias=False)
        self.k_proj = nn.Linear(H, KVH * HD, bias=False)
        self.v_proj = nn.Linear(H, KVH * HD, bias=False)
        self.o_proj = nn.Linear(H, H, bias=False)
        self.q_norm = RMSNorm(HD)
        self.k_norm = RMSNorm(HD)

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, NH, HD)
        k = self.k_proj(x).view(B, T, KVH, HD)
        v = self.v_proj(x).view(B, T, KVH, HD)
        q = self.q_norm(q.reshape(-1, HD)).reshape(B, T, NH, HD)
        k = self.k_norm(k.reshape(-1, HD)).reshape(B, T, KVH, HD)
        q = q.transpose(1, 2); k = k.transpose(1, 2); v = v.transpose(1, 2)
        q, k = apply_rope(q, k, cos, sin)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=False)
        out = out.transpose(1, 2).reshape(B, T, H)
        return self.o_proj(out)

    def forward_ext(self, x, cos, sin, cached_kv=None, sparse_mask=None, need_weights=False):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, NH, HD)
        k = self.k_proj(x).view(B, T, KVH, HD)
        v = self.v_proj(x).view(B, T, KVH, HD)
        q = self.q_norm(q.reshape(-1, HD)).reshape(B, T, NH, HD)
        k = self.k_norm(k.reshape(-1, HD)).reshape(B, T, KVH, HD)
        q = q.transpose(1, 2); k = k.transpose(1, 2); v = v.transpose(1, 2)
        q, k = apply_rope(q, k, cos, sin)
        k_new, v_new = k, v

        if cached_kv is not None:
            k_prefix, v_prefix, _ = cached_kv
            k_full = torch.cat([k_prefix, k], dim=2)
            v_full = torch.cat([v_prefix, v], dim=2)
        else:
            k_full, v_full = k, v

        attn_weights = None
        is_block_mask = sparse_mask is not None and not isinstance(sparse_mask, torch.Tensor)

        if need_weights:
            scale = 1.0 / math.sqrt(HD)
            scores = torch.matmul(q, k_full.transpose(-2, -1)) * scale
            if sparse_mask is not None:
                if is_block_mask:
                    bool_mask = sparse_mask.to_dense().bool()
                    if bool_mask.dim() == 4:
                        scores = scores.masked_fill(~bool_mask, float("-inf"))
                    else:
                        scores = scores.masked_fill(~bool_mask.unsqueeze(0), float("-inf"))
                else:
                    scores = scores.masked_fill(~sparse_mask.unsqueeze(0), float("-inf"))
            attn_weights = torch.softmax(scores.float(), dim=-1).to(q.dtype)
            out = torch.matmul(attn_weights, v_full)
        elif sparse_mask is not None:
            if is_block_mask:
                from torch.nn.attention.flex_attention import flex_attention
                out = flex_attention(q, k_full, v_full, block_mask=sparse_mask)
            else:
                float_mask = torch.zeros(1, NH, T, k_full.size(2), dtype=q.dtype, device=q.device)
                float_mask.masked_fill_(~sparse_mask.unsqueeze(0), float("-inf"))
                out = F.scaled_dot_product_attention(q, k_full, v_full, attn_mask=float_mask, is_causal=False)
        else:
            out = F.scaled_dot_product_attention(q, k_full, v_full, attn_mask=None, is_causal=False)

        out = out.transpose(1, 2).reshape(B, T, H)
        out = self.o_proj(out)
        return out, k_new, v_new, attn_weights


class ExpertMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(H, EI, bias=False)
        self.up_proj = nn.Linear(H, EI, bias=False)
        self.down_proj = nn.Linear(EI, H, bias=False)
    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MoEBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(H, NE, bias=False)
        self.experts = nn.ModuleList([ExpertMLP() for _ in range(NE)])

    def forward(self, x, dynamic_k=None, expert_threshold=0.0):
        B, T, _ = x.shape
        x_flat = x.view(B * T, H)
        x_dtype = x.dtype

        router_logits = self.gate(x_flat)
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)

        k = dynamic_k if dynamic_k is not None else TOPK
        topk_weights, topk_indices = torch.topk(routing_weights, k, dim=-1)

        if expert_threshold > 0:
            mask = topk_weights > expert_threshold
            topk_weights = topk_weights * mask
            sum_w = topk_weights.sum(dim=-1, keepdim=True).clamp(min=1e-9)
            topk_weights = topk_weights / sum_w

        topk_weights = topk_weights.to(x_dtype)

        out = torch.zeros_like(x_flat)
        for expert_idx in range(NE):
            mask = (topk_indices == expert_idx)
            if not mask.any():
                continue
            token_idx, rank_idx = torch.where(mask)
            if token_idx.numel() == 0:
                continue
            weights = topk_weights[token_idx, rank_idx]
            tokens = x_flat[token_idx]
            expert = self.experts[expert_idx]
            h = expert(tokens) * weights.unsqueeze(-1)
            out.index_add_(0, token_idx, h)

        return out.view(B, T, H)


class Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layernorm = RMSNorm(H)
        self.self_attn = Attention()
        self.post_attention_layernorm = RMSNorm(H)
        self.mlp = MoEBlock()

    def forward(self, x, cos, sin, dynamic_k=None, expert_threshold=0.0):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)
        x = x + self.mlp(self.post_attention_layernorm(x), dynamic_k=dynamic_k, expert_threshold=expert_threshold)
        return x

    def forward_ext(self, x, cos, sin, cached_kv=None, sparse_mask=None,
                    need_weights=False, dynamic_k=None, expert_threshold=0.0):
        attn_out, k_new, v_new, aw = self.self_attn.forward_ext(
            self.input_layernorm(x), cos, sin,
            cached_kv=cached_kv, sparse_mask=sparse_mask, need_weights=need_weights,
        )
        x = x + attn_out
        x = x + self.mlp(self.post_attention_layernorm(x), dynamic_k=dynamic_k, expert_threshold=expert_threshold)
        return x, k_new, v_new, aw


class LLaDAMoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(VS, H)
        self.layers = nn.ModuleList([Layer() for _ in range(NL)])
        self.norm = RMSNorm(H)
        self.lm_head = nn.Linear(H, VS, bias=False)

    def forward(self, input_ids, dynamic_k=None, expert_threshold=0.0):
        B, T = input_ids.shape
        device = input_ids.device
        x = self.embed_tokens(input_ids)
        cos, sin = build_rope_freqs(T, HD, THETA, device)
        cos = cos.to(x.dtype); sin = sin.to(x.dtype)
        for layer in self.layers:
            x = layer(x, cos, sin, dynamic_k=dynamic_k, expert_threshold=expert_threshold)
        x = self.norm(x)
        return self.lm_head(x)

    @torch.no_grad()
    def forward_with_attn(self, input_ids, dynamic_k=None, expert_threshold=0.0):
        B, T = input_ids.shape
        device = input_ids.device
        x = self.embed_tokens(input_ids)
        cos, sin = build_rope_freqs(T, HD, THETA, device)
        cos = cos.to(x.dtype); sin = sin.to(x.dtype)
        all_attn = []
        for layer in self.layers:
            x, _, _, aw = layer.forward_ext(x, cos, sin, cached_kv=None, sparse_mask=None,
                                             need_weights=True, dynamic_k=dynamic_k, expert_threshold=expert_threshold)
            all_attn.append(aw)
        x = self.norm(x)
        return self.lm_head(x), all_attn

    @torch.no_grad()
    def forward_active(
        self,
        active_ids,
        prefix_len,
        layer_caches,
        sparse_pattern=None,
        step=0,
        sparse_step_threshold=10 ** 9,
        need_weights=False,
        dynamic_k=None,
        expert_threshold=0.0,
    ):
        B, Ta = active_ids.shape
        device = active_ids.device
        x = self.embed_tokens(active_ids)

        total_len = prefix_len + Ta
        q_positions = torch.arange(prefix_len, total_len, device=device)
        cos, sin = rope_cos_sin_at(q_positions, HD, THETA, device)
        cos = cos.to(x.dtype); sin = sin.to(x.dtype)

        use_sparse = sparse_pattern is not None and step >= sparse_step_threshold
        new_kv = []
        all_attn = []

        for li, layer in enumerate(self.layers):
            cache = layer_caches[li]
            cached = cache.get()

            sparse_mask = None
            if use_sparse:
                k_positions = torch.cat([cached[2], q_positions]) if cached is not None else q_positions
                sparse_mask = sparse_pattern.build_mask(li, q_positions, k_positions, device)

            x, k_new, v_new, aw = layer.forward_ext(
                x, cos, sin, cached_kv=cached, sparse_mask=sparse_mask,
                need_weights=need_weights, dynamic_k=dynamic_k, expert_threshold=expert_threshold,
            )
            new_kv.append((k_new, v_new))
            all_attn.append(aw)

        x = self.norm(x)
        logits = self.lm_head(x)
        return logits, new_kv, q_positions, all_attn


def _hf_to_our_key(hk):
    if hk.startswith("model."):
        return hk[len("model."):]
    if hk == "lm_head.weight":
        return hk
    return None


def load_weights(model, weight_dir, verbose=True):
    index_path = os.path.join(weight_dir, "model.safetensors.index.json")
    with open(index_path) as f:
        wmap = json.load(f)["weight_map"]

    shards = {}
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
                mismatches.append(f"missing: {hk} -> {mk}")
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