"""
TP-aware weight loading for LLaDAMoEKV.

Reads the local safetensors directory produced by `download_weights.py`
(via huggingface_hub.snapshot_download) and loads each rank's correct shard.

Usage:
    # 1. Download once (no change to this step)
    python download_weights.py --dest ./weights

    # 2. Launch with torchrun for TP=2
    torchrun --nproc_per_node=2 entrypoint.py

    Inside entrypoint.py:
        from model_update.distributed import init_distributed
        from model_update.model import LLaDAMoEKV, FULL_CFG
        from model_update.load_weights import load_weights

        init_distributed()
        model = LLaDAMoEKV(FULL_CFG).to(device)
        model = load_weights(model, "./weights")
"""

import json
import os
from typing import Optional

import torch
from safetensors import safe_open

from .distributed import get_tp_size, get_tp_rank
from .model import LLaDAMoEKV



def _hf_to_our_key(hk: str) -> Optional[str]:
    if hk.startswith("model."):
        return hk[len("model."):]
    if hk == "lm_head.weight":
        return hk
    return None


_COLWISE_SUFFIXES = (
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
)


_ROWWISE_SUFFIXES = ("self_attn.o_proj.weight",)


def _slice_for_tp(tensor: torch.Tensor, key: str, tp_rank: int, tp_size: int) -> torch.Tensor:
    if any(key.endswith(sfx) for sfx in _COLWISE_SUFFIXES):
        dim_size = tensor.shape[0]
        assert dim_size % tp_size == 0, (
            f"TP slice error: {key} dim0={dim_size} not divisible by tp_size={tp_size}"
        )
        chunk = dim_size // tp_size
        return tensor[tp_rank * chunk : (tp_rank + 1) * chunk].contiguous()

    if any(key.endswith(sfx) for sfx in _ROWWISE_SUFFIXES):
        dim_size = tensor.shape[1]
        assert dim_size % tp_size == 0, (
            f"TP slice error: {key} dim1={dim_size} not divisible by tp_size={tp_size}"
        )
        chunk = dim_size // tp_size
        return tensor[:, tp_rank * chunk : (tp_rank + 1) * chunk].contiguous()

    return tensor 


def load_weights(
    model: LLaDAMoEKV,
    weight_dir: str,
    verbose: bool = True,
) -> LLaDAMoEKV:
    tp_size = get_tp_size()
    tp_rank = get_tp_rank()

    index_path = os.path.join(weight_dir, "model.safetensors.index.json")
    with open(index_path) as f:
        wmap = json.load(f)["weight_map"]

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
            if mk is None or mk not in sd:
                if mk is not None:
                    mismatches.append(f"missing in our model: {hk} → {mk}")
                continue

            t = f.get_tensor(hk)

            t = _slice_for_tp(t, mk, tp_rank, tp_size)

            if t.shape != sd[mk].shape:
                mismatches.append(
                    f"shape mismatch {hk}: "
                    f"checkpoint_sliced={t.shape} model_expects={sd[mk].shape}"
                )
                continue

            sd[mk] = t.to(sd[mk].dtype)
            mapped += 1

    if mismatches:
        print(f"  Issues ({len(mismatches)}):")
        for m in mismatches[:20]:
            print(f"    {m}")

    if verbose:
        total = sum(1 for k in wmap if _hf_to_our_key(k) is not None)
        rank_tag = f"[rank {tp_rank}/{tp_size}] " if tp_size > 1 else ""
        print(f"  {rank_tag}Mapped {mapped}/{total} tensors from {weight_dir}")

    model.load_state_dict(sd, strict=False)
    return model
