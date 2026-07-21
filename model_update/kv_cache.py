"""
Shared building blocks for the Sparse-dLLM + SparseD integration.

Two independent mechanisms live here:

  * LayerKVCache  - Sparse-dLLM (arXiv 2508.02558). A per-layer, evictable
    KV cache for the "frozen" part of the sequence (prompt + already
    finalized blocks). After each denoising step we record how much
    attention mass each cached token received ("saliency") and, once the
    cache exceeds a configurable budget, evict the lowest-saliency entries.
    Saliency is tracked with an exponential running average, matching the
    paper's finding that token saliency is stable across denoising steps
    (so a slowly-updated score is a reasonable proxy, not just "this step's
    attention").

  * SparsePattern - SparseD (arXiv 2509.24014). A fixed, per-(layer, head)
    local-window + global-stride attention pattern, calibrated once offline
    (see `calibrate_sparse_pattern` in generate.py) and reused unchanged
    across every denoising step and every future generation call. It is
    only ever applied from a configurable step threshold onward; the first
    steps of each block always use full dense attention (the paper found
    this matters for quality).

Both are wired up as an attention *mask* inside ordinary (non-fused)
softmax attention in model.py / model_small.py. This is a correctness
prototype, not a throughput optimization: a real speed win requires a
custom sparse-attention kernel (and, for the cache, an implementation that
never materializes the evicted K/V rather than concatenating-then-slicing).
That kernel work is explicitly out of scope here (Phase 4).
"""

from __future__ import annotations

import torch


# ─────────────────────────── Sparse-dLLM: evictable cache ──────────────────
class LayerKVCache:
    """Per-transformer-layer cache of *finalized* K/V (prompt + committed blocks).

    Positions stored here are frozen once appended: they are not
    recomputed when later blocks change (the same approximation any
    block-wise dLLM cache makes). What Sparse-dLLM adds on top is that
    entries can also be *evicted* (not just kept forever) once a saliency
    score marks them as unimportant and the cache exceeds `budget`.
    """

    def __init__(self, budget: int | None = None, saliency_decay: float = 0.9):
        self.budget = budget
        self.decay = saliency_decay
        self.k: torch.Tensor | None = None          # [B, NH, N, HD]
        self.v: torch.Tensor | None = None           # [B, NH, N, HD]
        self.positions: torch.Tensor | None = None   # [N] absolute sequence positions
        self.saliency: torch.Tensor | None = None    # [N] running attention-mass score
        self.protected: torch.Tensor | None = None    # [N] bool, never evicted

    def get(self):
        """Returns (k, v, positions) or None if the cache is empty."""
        if self.k is None:
            return None
        return self.k, self.v, self.positions

    @torch.no_grad()
    def append(self, k_new: torch.Tensor, v_new: torch.Tensor,
               positions_new: torch.Tensor, protected: bool = False):
        """Permanently add newly-finalized tokens' K/V to the cache."""
        _, _, T, _ = k_new.shape
        sal_new = torch.zeros(T, device=k_new.device)
        prot_new = torch.full((T,), protected, dtype=torch.bool, device=k_new.device)

        if self.k is None:
            self.k, self.v = k_new, v_new
            self.positions = positions_new.clone()
            self.saliency, self.protected = sal_new, prot_new
        else:
            self.k = torch.cat([self.k, k_new], dim=2)
            self.v = torch.cat([self.v, v_new], dim=2)
            self.positions = torch.cat([self.positions, positions_new])
            self.saliency = torch.cat([self.saliency, sal_new])
            self.protected = torch.cat([self.protected, prot_new])

        if self.budget is not None:
            self.evict()

    @torch.no_grad()
    def update_saliency(self, attn_weights_to_cache: torch.Tensor, key_positions: torch.Tensor):
        """
        attn_weights_to_cache: [B, NH, Tq, Nk] attention weights restricted to the
            columns that correspond to *this cache's* current keys (Nk must equal
            len(key_positions), which should equal len(self.positions)).
        Score per Sparse-dLLM: how much attention mass a token receives, max'd
        over queries/heads/batch this step, folded into a running average since
        saliency is reported to be stable across steps.
        """
        if self.k is None or attn_weights_to_cache is None:
            return
        received = attn_weights_to_cache.amax(dim=(0, 1, 2))  # [Nk]
        n = min(received.shape[0], self.saliency.shape[0])
        self.saliency[:n] = (
            self.decay * self.saliency[:n] + (1 - self.decay) * received[:n].to(self.saliency.device)
        )

    @torch.no_grad()
    def evict(self):
        """Drop lowest-saliency, unprotected prefix/suffix entries down to `budget`."""
        if self.budget is None or self.k is None:
            return
        n = self.k.shape[2]
        n_drop = n - self.budget
        if n_drop <= 0:
            return

        order = torch.argsort(self.saliency)  # ascending: least salient first
        drop = []
        for i in order.tolist():
            if not self.protected[i]:
                drop.append(i)
            if len(drop) == n_drop:
                break
        if not drop:
            return

        keep_mask = torch.ones(n, dtype=torch.bool, device=self.k.device)
        keep_mask[torch.tensor(drop, device=self.k.device)] = False
        self.k = self.k[:, :, keep_mask, :]
        self.v = self.v[:, :, keep_mask, :]
        self.positions = self.positions[keep_mask]
        self.saliency = self.saliency[keep_mask]
        self.protected = self.protected[keep_mask]

    def __len__(self):
        return 0 if self.k is None else self.k.shape[2]


# ─────────────────────────── SparseD: calibrated pattern ───────────────────
class SparsePattern:
    """Fixed per-(layer, head) local-window + global-stride attention pattern.

    A key at absolute position k is attended to by a query at position q for
    head h in layer l iff:
        |q - k| <= window[l, h]      (local)
     OR (stride[l, h] > 0 and k % stride[l, h] == 0)   (strided/global)

    This is deliberately simple (no learned/absolute-index pattern) so it
    generalizes across sequences of different lengths and different prompts,
    unlike a pattern tied to specific calibration-time token indices.
    """

    def __init__(self, num_layers: int, num_heads: int,
                 window: torch.Tensor, stride: torch.Tensor):
        assert window.shape == (num_layers, num_heads)
        assert stride.shape == (num_layers, num_heads)
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.window = window   # LongTensor [NL, NH]
        self.stride = stride   # LongTensor [NL, NH]

    def build_mask(self, layer_idx: int, q_positions: torch.Tensor,
                   k_positions: torch.Tensor, device) -> torch.Tensor:
        """Returns bool mask [NH, Tq, Tk], True = attend."""
        q = q_positions.view(-1, 1).to(device)
        k = k_positions.view(1, -1).to(device)
        dist = (q - k).abs()
        masks = []
        for h in range(self.num_heads):
            w = int(self.window[layer_idx, h].item())
            s = int(self.stride[layer_idx, h].item())
            m = dist <= w
            if s > 0:
                glob = (k_positions.view(1, -1).to(device) % s == 0).expand_as(m)
                m = m | glob
            masks.append(m)
        return torch.stack(masks, dim=0)

    def save(self, path: str):
        torch.save({"window": self.window, "stride": self.stride}, path)

    @classmethod
    def load(cls, path: str) -> "SparsePattern":
        d = torch.load(path)
        nl, nh = d["window"].shape
        return cls(nl, nh, d["window"], d["stride"])


def _candidate_mass(attn_head: torch.Tensor, window: int, stride: int) -> float:
    """Fraction of a [T, T] attention-weight matrix's mass captured by a given
    local-window(+stride) candidate pattern. Used only during calibration."""
    T = attn_head.shape[0]
    device = attn_head.device
    q = torch.arange(T, device=device).view(-1, 1)
    k = torch.arange(T, device=device).view(1, -1)
    dist = (q - k).abs()
    mask = dist <= window
    if stride > 0:
        glob = (k % stride == 0).expand(T, T)
        mask = mask | glob
    total = attn_head.sum()
    if total <= 0:
        return 0.0
    return (attn_head * mask).sum().item() / total.item()
