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


class KVCacheBuffer:
    """
    Preallocated per-layer K/V buffer for block-wise generation.

    Replaces torch.cat-ing a fresh [prefix; active] tensor on every
    denoising step: profiling (eval/profile_kv_concat.py) showed that cat
    cost 45-68% of combined attend-time, exceeding the actual attention
    math at every context length measured, because it recopies the whole
    (unchanged) prefix just to append a few new tokens.

    Instead, the full [B, KVH, max_len, HD] tensor is allocated once and
    each step writes its K/V in place at `write_pos`; attention reads a
    zero-copy view of the buffer up to that point.
    """

    def __init__(self, num_layers: int, batch_size: int, kvh_local: int,
                 max_len: int, head_dim: int, dtype: torch.dtype, device):
        self.max_len = max_len
        self.committed_len = 0
        self.k = [
            torch.empty(batch_size, kvh_local, max_len, head_dim, dtype=dtype, device=device)
            for _ in range(num_layers)
        ]
        self.v = [
            torch.empty(batch_size, kvh_local, max_len, head_dim, dtype=dtype, device=device)
            for _ in range(num_layers)
        ]

    def write_and_view(self, layer_idx: int, k_new: torch.Tensor, v_new: torch.Tensor, start_pos: int):
        end_pos = start_pos + k_new.shape[2]
        if end_pos > self.max_len:
            raise ValueError(f"KV cache overflow: write to {end_pos} exceeds buffer of {self.max_len}")
        self.k[layer_idx][:, :, start_pos:end_pos, :] = k_new
        self.v[layer_idx][:, :, start_pos:end_pos, :] = v_new
        return self.k[layer_idx][:, :, :end_pos, :], self.v[layer_idx][:, :, :end_pos, :]

    def commit(self, new_committed_len: int):
        """Mark [0:new_committed_len] as permanent (already-finalized) prefix."""
        self.committed_len = new_committed_len


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.device.type == "cpu":
            raise NotImplementedError("Triton Fused MoE requires a CUDA GPU.")

        B, T, H = x.shape
        x_flat = x.view(B * T, H)
        router_logits = self.gate(x_flat)
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)

        topk_weights, topk_ids = torch.topk(routing_weights, self.cfg.TOPK, dim=-1)

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

    def forward(
        self, x, cos, sin,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cache_buffer: Optional["KVCacheBuffer"] = None,
        layer_idx: Optional[int] = None,
        write_pos: Optional[int] = None,
    ):
        """
        x: [B, Ta, H] active-block hidden states
        cos/sin: [Ta, HD] rope freqs for the ACTIVE positions (absolute offset already applied)
        past_kv: optional (k_prefix, v_prefix) each [B, KVH, Tp, HD], already RoPE'd —
                 cat'd with this call's K/V to build the attend-over tensor.
        cache_buffer/layer_idx/write_pos: alternative to past_kv. K/V are written
                 in place into cache_buffer at write_pos and attention reads a
                 zero-copy view of the buffer, instead of cat-ing a fresh tensor.
                 Mutually exclusive with past_kv.
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

        if cache_buffer is not None:
            k_full, v_full = cache_buffer.write_and_view(layer_idx, k, v, write_pos)
        elif past_kv is not None:
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

    def forward(self, x):
        cfg = self.cfg
        B, T, _ = x.shape
        x_flat = x.view(B * T, cfg.H)

        routing_weights_full = F.softmax(self.gate(x_flat), dim=-1, dtype=torch.float32)
        routing_weights, selected_experts = torch.topk(routing_weights_full, cfg.TOPK, dim=-1)

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

    def forward(
        self, x, cos, sin, past_kv=None,
        cache_buffer: Optional["KVCacheBuffer"] = None,
        layer_idx: Optional[int] = None,
        write_pos: Optional[int] = None,
    ):
        attn_out, kv_new = self.self_attn(
            self.input_layernorm(x), cos, sin, past_kv,
            cache_buffer=cache_buffer, layer_idx=layer_idx, write_pos=write_pos,
        )
        x = x + attn_out
        x = x + self.mlp(self.post_attention_layernorm(x))
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
        cache_buffer: Optional[KVCacheBuffer] = None,
        write_pos: Optional[int] = None,
    ):
        """
        input_ids: [B, T] — either the full sequence (past_kv=None, e.g. prefix
                   priming) or just the active block (past_kv=cache).
        position_offset: absolute starting position of input_ids in the full
                   sequence, for correct RoPE.
        past_kv: list of (k,v) per layer for the prefix, or None.
        cache_buffer/write_pos: preallocated in-place KV cache (see
                   KVCacheBuffer). Mutually exclusive with past_kv — when
                   given, K/V for input_ids are written into the buffer at
                   write_pos and attention reads a view of it, instead of
                   cat-ing a fresh tensor from past_kv every call.
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
            x, kv_i = layer(
                x, cos, sin, layer_past,
                cache_buffer=cache_buffer, layer_idx=i, write_pos=write_pos,
            )
            new_kv.append(kv_i)

        x = self.norm(x)
        logits = self.lm_head(x)
        return logits, new_kv


class CUDAGraphRunner:
    """
    Captures LLaDAMoEKV.forward() as a CUDA graph, keyed by the
    (T, position_offset, write_pos) combination that determines every
    kernel's launch shape/arguments for that call, and replays it on later
    calls with the same key -- copying new input data into a static buffer
    instead of re-dispatching every op through Python each time.

    *** OPT-IN, OFF BY DEFAULT, NOT VALIDATED ON REAL HARDWARE AS WRITTEN. ***
    CUDA graph capture is unusually easy to get subtly wrong (stale data,
    mismatched shapes, memory-pool lifetime issues) even for someone
    iterating against a live GPU. This was written and reasoned through
    without one -- every invariant below is justified by inspection of the
    surrounding code, not by a passing test on real hardware, since none of
    that hardware was available while writing it. Run
    eval/test_cuda_graph_forward.py, and eval/diagnose_cache_vs_dense.py
    with use_cuda_graph=True, on an actual GPU before trusting generation
    output from this path -- treat it as substantially less proven than
    everything else in this file.

    Correctness-relevant invariants this relies on:

    1. moe_align_block_size's output shapes must depend only on static
       input shape (num_tokens, top_k, num_experts, block_size), never on
       runtime tensor VALUES -- true as of the rewrite that made it
       allocate a fixed worst-case-sized buffer instead of sizing to the
       exact (data-dependent) padded length (see fused_moe_triton.py).
       Without that, capture would either raise (a host sync inside a
       captured region is invalid) or silently misbehave. This is the
       single hard blocker that made graph capture possible at all; there
       is no runtime assertion for it here, because CUDA graph capture
       doesn't leave a live Python frame to assert from during replay --
       it was verified by code inspection and by
       eval/test_moe_align_block_size.py's CPU-side equivalence tests.

    2. KVCacheBuffer's per-layer k/v tensors are allocated once
       (torch.empty, in KVCacheBuffer.__init__) and never reallocated for
       the lifetime of one generate_cached() call. CUDA graphs record
       fixed memory addresses, not symbolic tensor references -- if the
       cache buffer were ever reallocated (it isn't, by design elsewhere
       in this file) a captured graph would silently read/write the wrong
       memory after replay.

    3. Capture and warmup both run REAL forward passes against the actual
       cache_buffer for this call (a clone of the real input, not
       synthetic/placeholder data). This is safe specifically *because*
       every warmup/capture iteration writes the SAME correct final K/V
       for this position -- write_and_view() overwrites rather than
       accumulates, so redundant identical writes are harmless, and the
       last one (from capture itself) is what ends up cached. This would
       NOT be safe with placeholder/zero/random warmup data, which would
       leave incorrect K/V behind after the warmup writes and before the
       real capture write overwrote them -- fine for warmup here only
       because "real capture write happens last and is correct" holds.

    4. Graphs are cached per CUDAGraphRunner INSTANCE, and generate_cached()
       creates a fresh instance per call (see generate.py) -- matching
       cache_buffer's own per-call lifetime. Graphs are NEVER reused across
       different generate_cached() calls/cache_buffers: a graph captured
       against one cache_buffer's memory addresses would be silently wrong
       (or crash) if replayed against a different one. Amortization is
       therefore within one generation call's repeated steps (capture once
       per block, replay steps_per_block-1 times), not across separate
       requests -- the safe tradeoff given a fresh cache_buffer per call is
       an existing, load-bearing design choice elsewhere in this file.

    5. tp_size must be 1. Attention.forward and the MoE forward both call
       dist.all_reduce() when tp_size > 1 (TP+EP mode) -- capturing a
       collective communication op into a CUDA graph has its own
       NCCL/stream compatibility requirements that were never investigated
       here, on top of everything else in this docstring. Every current
       caller of CUDAGraphRunner only reaches it through the tp_size == 1
       batching path (src/server.py's _run_batch), so this is enforced
       structurally today, not just by convention -- but __init__ still
       asserts it explicitly so a future caller wiring this into the
       TP+EP path fails loudly at construction instead of silently hitting
       unverified collective-capture behavior.
    """

    def __init__(self, model, num_warmup: int = 3):
        from .distributed import get_tp_size
        assert get_tp_size() == 1, (
            "CUDAGraphRunner is only validated (by inspection, not hardware) "
            "for tp_size == 1. dist.all_reduce() calls inside Attention/MoE "
            "forward under TP+EP have unverified CUDA-graph-capture "
            "compatibility -- see invariant 5 in this class's docstring."
        )

        self.model = model
        self.num_warmup = num_warmup
        self._graphs: dict = {}  # key -> (CUDAGraph, static_input, static_logits, static_new_kv)

    def __call__(self, input_ids, position_offset, cache_buffer, write_pos):
        key = (input_ids.shape[1], position_offset, write_pos)
        if key not in self._graphs:
            self._capture(key, input_ids, position_offset, cache_buffer, write_pos)
        return self._replay(key, input_ids)

    def _capture(self, key, input_ids, position_offset, cache_buffer, write_pos):
        static_input = input_ids.clone()

        # Warmup on a side stream (PyTorch's documented CUDA graph pattern):
        # lets one-time lazy CUDA/cuBLAS initialization happen BEFORE
        # capture, so it doesn't get recorded into (or corrupt) the graph.
        # Uses the real cache_buffer with the real input -- see invariant 3
        # above for why that's safe specifically in this case.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(self.num_warmup):
                self.model(
                    static_input, position_offset=position_offset,
                    cache_buffer=cache_buffer, write_pos=write_pos,
                )
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_logits, static_new_kv = self.model(
                static_input, position_offset=position_offset,
                cache_buffer=cache_buffer, write_pos=write_pos,
            )
        self._graphs[key] = (graph, static_input, static_logits, static_new_kv)

    def _replay(self, key, input_ids):
        graph, static_input, static_logits, static_new_kv = self._graphs[key]
        static_input.copy_(input_ids)
        graph.replay()
        return static_logits, static_new_kv
