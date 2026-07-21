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


def rope_cos_sin_at(positions: torch.Tensor, head_dim: int, theta: float, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Like build_rope_freqs, but for an arbitrary set of absolute positions.
    Needed once we cache K/V: new tokens' RoPE must use their true absolute
    position, not a position relative to the active window."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    freqs = torch.outer(positions.float(), inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


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

    def forward_ext(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cached_kv: Optional[tuple] = None,
        sparse_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ):
        """
        Sparse-dLLM / SparseD attention path used by generate_sparse_cached() and
        by calibration. `forward()` above is untouched and remains the dense,
        no-cache correctness baseline.

        cached_kv   : optional (k_prefix, v_prefix, positions) from a LayerKVCache,
                      representing the frozen prompt + already-finalized blocks.
                      `x` here is only the *active* (not-yet-finalized) suffix.
        sparse_mask : optional [NH, T, Nk] bool mask (True = attend), from a
                      calibrated SparsePattern. None = full dense attention over
                      whatever keys are present (cached prefix + active).
        need_weights: if True (or sparse_mask is given), compute attention with
                      an explicit softmax so weights can be returned/used for
                      Sparse-dLLM saliency updates, instead of the fused SDPA
                      kernel (which doesn't expose weights).

        Returns: (out, k_new, v_new, attn_weights)
          k_new/v_new are this call's *own* tokens' K/V (pre-concatenation),
          i.e. what the caller should later commit to the cache once these
          tokens are finalized. attn_weights is None unless computed.
        """
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
            scores = torch.matmul(q, k_full.transpose(-2, -1)) * scale   # [B,NH,T,Nk]
            if sparse_mask is not None:
                if is_block_mask:
                    # Materialize BlockMask to a boolean tensor for the manual path.
                    # NOTE: BlockMask.to_dense() returns an int32 (0/1) tensor on
                    # PyTorch 2.5.1, not bool — masked_fill requires an actual bool
                    # mask, so cast explicitly (non-zero -> True == "attend").
                    bool_mask = sparse_mask.to_dense().bool()  # [1, NH, Tq, Tk] or similar
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
                # flex_attention: real block-level FLOP savings
                from torch.nn.attention.flex_attention import flex_attention
                out = flex_attention(q, k_full, v_full, block_mask=sparse_mask)
            else:
                # Fallback: use SDPA with an additive float mask from the boolean mask.
                float_mask = torch.zeros(1, NH, T, k_full.size(2), dtype=q.dtype, device=q.device)
                float_mask.masked_fill_(~sparse_mask.unsqueeze(0), float("-inf"))
                out = F.scaled_dot_product_attention(q, k_full, v_full, attn_mask=float_mask, is_causal=False)
        else:
            out = F.scaled_dot_product_attention(q, k_full, v_full, attn_mask=None, is_causal=False)

        out = out.transpose(1, 2).reshape(B, T, H)
        out = self.o_proj(out)
        return out, k_new, v_new, attn_weights


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

    def forward_ext(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cached_kv: Optional[tuple] = None,
        sparse_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ):
        attn_out, k_new, v_new, aw = self.self_attn.forward_ext(
            self.input_layernorm(x), cos, sin,
            cached_kv=cached_kv, sparse_mask=sparse_mask, need_weights=need_weights,
        )
        x = x + attn_out
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, k_new, v_new, aw


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

    @torch.no_grad()
    def forward_with_attn(self, input_ids: torch.Tensor):
        """Dense, no-cache forward that also returns per-layer attention weights.
        Used only for SparseD offline calibration (calibrate_sparse_pattern in
        generate.py). Numerically this uses an explicit softmax rather than the
        fused SDPA kernel used by forward(), since SDPA doesn't expose weights;
        it is not used for actual generation output.
        """
        B, T = input_ids.shape
        device = input_ids.device
        x = self.embed_tokens(input_ids)
        cos, sin = build_rope_freqs(T, HD, THETA, device)
        cos = cos.to(x.dtype)
        sin = sin.to(x.dtype)
        all_attn = []
        for layer in self.layers:
            x, _, _, aw = layer.forward_ext(x, cos, sin, cached_kv=None, sparse_mask=None, need_weights=True)
            all_attn.append(aw)
        x = self.norm(x)
        return self.lm_head(x), all_attn

    @torch.no_grad()
    def forward_active(
        self,
        active_ids: torch.Tensor,
        prefix_len: int,
        layer_caches: list,
        sparse_pattern=None,
        step: int = 0,
        sparse_step_threshold: int = 10 ** 9,
        need_weights: bool = False,
    ):
        """
        Compute logits for the *active* (not-yet-finalized) suffix of the
        sequence, attending to a frozen, evictable cached prefix (prompt +
        finalized blocks, from `layer_caches`, one LayerKVCache per layer)
        plus the active tokens themselves. This is the entry point used by
        generate_sparse_cached() in generate.py; forward() above is untouched
        and remains the fully-dense correctness baseline.

        sparse_pattern         : SparsePattern or None (dense attention only).
        step / sparse_step_threshold: the calibrated pattern is applied only
                                  once `step >= sparse_step_threshold` within
                                  the current block; earlier steps stay dense.

        Returns: (logits, new_kv, q_positions, all_attn)
          new_kv     : list[(k_new, v_new)] per layer for the active tokens —
                       caller commits these to the cache once finalized.
          q_positions: absolute positions of the active tokens (for bookkeeping).
          all_attn   : list of attention-weight tensors per layer if
                       need_weights else list of Nones.
        """
        B, Ta = active_ids.shape
        device = active_ids.device
        x = self.embed_tokens(active_ids)

        total_len = prefix_len + Ta
        q_positions = torch.arange(prefix_len, total_len, device=device)
        cos, sin = rope_cos_sin_at(q_positions, HD, THETA, device)
        cos = cos.to(x.dtype)
        sin = sin.to(x.dtype)

        use_sparse = sparse_pattern is not None and step >= sparse_step_threshold
        new_kv = []
        all_attn = []

        for li, layer in enumerate(self.layers):
            cache = layer_caches[li]
            cached = cache.get()

            sparse_mask = None
            if use_sparse:
                k_positions = torch.cat([cached[2], q_positions]) if cached is not None else q_positions
                try:
                    sparse_mask = sparse_pattern.build_block_mask(li, q_positions, k_positions, device)
                except Exception:
                    sparse_mask = sparse_pattern.build_mask(li, q_positions, k_positions, device)

            x, k_new, v_new, aw = layer.forward_ext(
                x, cos, sin, cached_kv=cached, sparse_mask=sparse_mask, need_weights=need_weights
            )
            new_kv.append((k_new, v_new))
            all_attn.append(aw)

        x = self.norm(x)
        logits = self.lm_head(x)
        return logits, new_kv, q_positions, all_attn


# ── Weight loading ────────────────────────────────────────────────────────────
def _hf_to_our_key(hk: str) -> Optional[str]:
    """Strip HF 'model.' prefix. lm_head lives at top level in HF (no prefix)."""
    if hk.startswith("model."):
        return hk[len("model."):]
    # lm_head.weight is at the top level in HF weights (not under model.)
    if hk == "lm_head.weight":
        return hk
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