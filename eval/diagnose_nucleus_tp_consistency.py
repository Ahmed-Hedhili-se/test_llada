"""
Stage 2 of the adaptive-routing plan (see .claude/plans/polished-crunching-muffin.md):
confirm nucleus routing's per-token keep_mask is bit-identical across TP ranks
BEFORE Stage 3 relies on that property inside the fused Triton kernel.

Reasoning being checked: the attention output `x` feeding into each MoE block is
already all-reduced (summed identically on every rank) before the local
`self.gate(x_flat)` call, so routing_weights/keep_mask computed independently on
each rank from that identical `x` should be bit-identical too — the same
precondition the existing step-based dynamic_k ramp already relies on. This
script verifies it empirically rather than just assuming it holds.

Uses DIFFERENT per-rank seeds for model construction (so local attention/expert
shard weights genuinely differ across ranks, like real sharded pretrained
weights would) but the SAME seed for the input, so both ranks process identical
input through genuinely different local computations up to the all-reduce.

Run with (needs >=2 GPUs):
    torchrun --nproc_per_node=2 eval/diagnose_nucleus_tp_consistency.py
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

workspace_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(workspace_root))

from model_update.distributed import init_distributed, get_tp_rank, get_tp_size
from model_update.model import LLaDAMoEKV, SMALL_CFG, nucleus_mask_weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nucleus-p", type=float, default=0.95)
    parser.add_argument("--seq-len", type=int, default=32)
    args = parser.parse_args()

    init_distributed()
    tp_size = get_tp_size()
    tp_rank = get_tp_rank()

    if tp_size < 2:
        print(
            f"[Rank {tp_rank}] tp_size={tp_size} — this check needs TP>=2 to mean anything.\n"
            f"Run with: torchrun --nproc_per_node=2 eval/diagnose_nucleus_tp_consistency.py"
        )
        return

    device = f"cuda:{tp_rank}"
    torch.cuda.set_device(device)

    # Different seed per rank -> genuinely different local shard weights
    # (attention q/k/v/o shards, local expert slices), like real sharded
    # pretrained weights, not an artificially-identical stand-in.
    torch.manual_seed(1234 + tp_rank)
    model = LLaDAMoEKV(SMALL_CFG, use_fused_moe=False).to(torch.bfloat16).to(device).eval()

    # Same seed across ranks -> identical input on every rank, simulating the
    # real server's broadcast-from-rank0 request without needing comms for it.
    torch.manual_seed(0)
    input_ids = torch.randint(0, SMALL_CFG.VS, (1, args.seq_len), device=device)

    captured = {}

    def make_hook(layer_idx):
        def hook(module, inputs):
            x = inputs[0]
            B, T, H = x.shape
            x_flat = x.reshape(B * T, H)
            routing_weights_full = F.softmax(module.gate(x_flat), dim=-1, dtype=torch.float32)
            vals, ids = torch.topk(routing_weights_full, module.cfg.TOPK, dim=-1)
            masked = nucleus_mask_weights(vals, args.nucleus_p)
            keep_mask = masked != 0
            captured[layer_idx] = (x.clone(), ids.clone(), keep_mask.clone())
        return hook

    hooks = [layer.mlp.register_forward_pre_hook(make_hook(i)) for i, layer in enumerate(model.layers)]
    with torch.no_grad():
        model(input_ids, position_offset=0)
    for h in hooks:
        h.remove()

    all_pass = True
    for layer_idx in sorted(captured.keys()):
        x, ids, keep_mask = captured[layer_idx]

        gathered_x = [torch.zeros_like(x) for _ in range(tp_size)]
        gathered_ids = [torch.zeros_like(ids) for _ in range(tp_size)]
        gathered_mask = [torch.zeros_like(keep_mask) for _ in range(tp_size)]
        dist.all_gather(gathered_x, x)
        dist.all_gather(gathered_ids, ids)
        dist.all_gather(gathered_mask, keep_mask)

        x_match = all(torch.equal(gathered_x[0], g) for g in gathered_x[1:])
        ids_match = all(torch.equal(gathered_ids[0], g) for g in gathered_ids[1:])
        mask_match = all(torch.equal(gathered_mask[0], g) for g in gathered_mask[1:])
        status = "PASS" if (x_match and ids_match and mask_match) else "FAIL"
        all_pass = all_pass and (status == "PASS")

        if tp_rank == 0:
            print(
                f"Layer {layer_idx:02d}: x_match={x_match}  "
                f"selected_experts_match={ids_match}  keep_mask_match={mask_match}  [{status}]"
            )

    if tp_rank == 0:
        print()
        if all_pass:
            print("ALL LAYERS CONSISTENT ACROSS RANKS ✅ — safe to rely on this in Stage 3's fused kernel.")
        else:
            print("MISMATCH DETECTED ❌ — do NOT proceed to Stage 3 until this is understood/fixed.")


if __name__ == "__main__":
    main()
