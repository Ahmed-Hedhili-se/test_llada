"""
MoE expert-routing divergence check: dminfr.engine vs HF, same long GSM8K
prompt as diagnose_layer_divergence.py.

That script found hidden-state cosine similarity between dminfr.engine and
HF collapses sharply and specifically at layer 7 (avg stays ~0.98, but min
crashes from 0.71 at layer 6 to 0.065 at layer 7 and stays broken through
layer 14) -- a signature of a DISCRETE decision flipping for a minority of
positions, not smooth numerical drift affecting everything uniformly.

Leading hypothesis: MoE expert-routing tie-breaking. Router top-8
selection is a hard, non-differentiable decision -- if two experts have
very close routing scores for a token, a tiny bf16 numerical difference
between dminfr.engine's custom Triton fused-MoE kernel and HF's native MoE
implementation could flip which expert wins, producing a completely
different output for that token at that layer, which then propagates
through the residual stream rather than washing out.

This captures the router's actual top-8 EXPERT SELECTION (not just final
hidden states) at every layer for both models, and cross-references
against per-position hidden-state cosine similarity at the layer where
the collapse was found -- if collapsed-cosine positions are exactly the
positions with a different top-8 expert set, that confirms the hypothesis.

Usage:
    python archive/investigations/diagnose_moe_routing_divergence.py --weight-dir ./weights
"""

import argparse
import gc
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

workspace_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(workspace_root))

MASK_ID = 156895
GEN_LEN = 32


def build_long_prompt(tok):
    from benchmarks.correctness.run_math_reasoning_code import _load_gsm8k, SYSTEM_PROMPT_GSM8K

    import random
    random.seed(42)
    items = _load_gsm8k(limit=1, num_fewshot=4)
    item = items[0]
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_GSM8K},
        {"role": "user", "content": item.prompt},
    ]
    prompt_text = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    prompt_ids = tok(prompt_text, return_tensors="pt")["input_ids"]
    return prompt_ids, item


def make_diffusion_input(prompt_ids, gen_length):
    P = prompt_ids.shape[1]
    x = torch.full((1, P + gen_length), MASK_ID, dtype=torch.long, device=prompt_ids.device)
    x[:, :P] = prompt_ids
    return x, P


def run_ours(weight_dir, x, topk):
    from dminfr.engine.model import LLaDAMoEKV, TritonFusedMoEBlock
    from dminfr.reference.model import load_weights

    print("  Loading dminfr.engine...")
    model = LLaDAMoEKV(use_fused_moe=False).to(torch.bfloat16).eval()
    load_weights(model, weight_dir, verbose=False)
    for layer in model.layers:
        fused_mlp = TritonFusedMoEBlock(layer.mlp.cfg).to(torch.bfloat16)
        fused_mlp.load_state_dict_from_unfused(layer.mlp)
        layer.mlp = fused_mlp
    model = model.to("cuda:0")

    router_logits_by_layer, hidden_by_layer = [], []

    def gate_hook(module, inp, out, store=router_logits_by_layer):
        store.append(out.detach().float().cpu())

    def layer_hook(module, inp, out, store=hidden_by_layer):
        store.append(out[0].detach().float().cpu())  # Layer.forward -> (x, kv_new)

    gate_hooks = [layer.mlp.gate.register_forward_hook(gate_hook) for layer in model.layers]
    layer_hooks = [layer.register_forward_hook(layer_hook) for layer in model.layers]

    x = x.to("cuda:0")
    with torch.no_grad():
        model(x)

    for h in gate_hooks + layer_hooks:
        h.remove()
    del model
    gc.collect()
    torch.cuda.empty_cache()

    topk_ids_by_layer = []
    for rl in router_logits_by_layer:
        rw = F.softmax(rl, dim=-1)
        _, ids = torch.topk(rw, topk, dim=-1)
        # TritonFusedMoEBlock flattens x to [B*T, H] before calling self.gate,
        # so captured router logits/ids here are [T, topk] (no batch dim) --
        # normalize to [T, topk] regardless, so both backends' captures have
        # an identical, predictable shape downstream.
        topk_ids_by_layer.append(ids.reshape(-1, topk))

    return topk_ids_by_layer, hidden_by_layer


def _find_decoder_layers(model, expected_len):
    candidates = []
    for name, module in model.named_modules():
        if isinstance(module, nn.ModuleList) and len(module) == expected_len:
            candidates.append((name.count("."), name, module))
    if not candidates:
        raise RuntimeError(f"Could not find a length-{expected_len} ModuleList on the HF model.")
    candidates.sort(key=lambda c: c[0])
    _, name, module_list = candidates[0]
    return [(f"{name}.{i}", layer) for i, layer in enumerate(module_list)]


def _find_router_linear(layer_module, num_experts, hidden_size):
    """Find the router/gate nn.Linear within a single decoder layer by its
    distinctive shape (in=hidden_size, out=num_experts) -- unlikely to
    collide with attention/FFN projections, which use different dims."""
    matches = [
        m for m in layer_module.modules()
        if isinstance(m, nn.Linear) and m.out_features == num_experts and m.in_features == hidden_size
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one router Linear(in={hidden_size}, out={num_experts}) per "
            f"layer, found {len(matches)}. HF architecture may differ from expected."
        )
    return matches[0]


def run_hf(weight_dir, x, topk, num_experts, hidden_size, expected_layers):
    print("  Loading HF reference...")
    model = AutoModelForCausalLM.from_pretrained(
        weight_dir, trust_remote_code=True, torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to("cuda:0").eval()

    layer_modules = _find_decoder_layers(model, expected_layers)
    router_linears = [_find_router_linear(mod, num_experts, hidden_size) for _, mod in layer_modules]
    print(f"  Found {len(layer_modules)} decoder layers and {len(router_linears)} router linears.")

    router_logits_by_layer, hidden_by_layer = [], []

    def gate_hook(module, inp, out, store=router_logits_by_layer):
        store.append(out.detach().float().cpu())

    def layer_hook(module, inp, out, store=hidden_by_layer):
        hs = out[0] if isinstance(out, tuple) else out
        store.append(hs.detach().float().cpu())

    gate_hooks = [lin.register_forward_hook(gate_hook) for lin in router_linears]
    layer_hooks = [mod.register_forward_hook(layer_hook) for _, mod in layer_modules]

    x = x.to("cuda:0")
    with torch.no_grad():
        model(x)

    for h in gate_hooks + layer_hooks:
        h.remove()
    del model
    gc.collect()
    torch.cuda.empty_cache()

    topk_ids_by_layer = []
    for rl in router_logits_by_layer:
        rw = F.softmax(rl, dim=-1)
        _, ids = torch.topk(rw, topk, dim=-1)
        # Normalize to [T, topk] the same way as run_ours, whatever HF's
        # router's actual input shape ([B,T,H] or already-flattened) is.
        topk_ids_by_layer.append(ids.reshape(-1, topk))

    return topk_ids_by_layer, hidden_by_layer


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weight-dir", default="weights")
    ap.add_argument("--target-layer", type=int, default=7,
                     help="Layer to cross-reference expert mismatch against hidden-state "
                          "cosine collapse (default 7, matching diagnose_layer_divergence.py's finding).")
    args = ap.parse_args()

    from dminfr.engine.model import FULL_CFG

    tok = AutoTokenizer.from_pretrained(args.weight_dir, trust_remote_code=True)
    prompt_ids, item = build_long_prompt(tok)
    x, P = make_diffusion_input(prompt_ids, GEN_LEN)
    T = x.shape[1]
    print(f"Long prompt: {P} tokens, total sequence {T} tokens\n")

    print("Running dminfr.engine (ours) with router + layer hooks...")
    our_topk, our_hidden = run_ours(args.weight_dir, x, FULL_CFG.TOPK)

    print("Running HF reference with router + layer hooks...")
    hf_topk, hf_hidden = run_hf(
        args.weight_dir, x, FULL_CFG.TOPK, FULL_CFG.NE, FULL_CFG.H, FULL_CFG.NL,
    )

    n_layers = min(len(our_topk), len(hf_topk))

    print(f"\n{'='*88}")
    print("PER-LAYER EXPERT-SET MISMATCH RATE  (fraction of positions where the two models")
    print("selected a DIFFERENT set of top-8 experts, compared as unordered sets)")
    print(f"{'='*88}")
    print(f"{'layer':>6}  {'mismatch_rate':>15}  {'n_mismatched':>16}")
    mismatch_masks = []
    for i in range(n_layers):
        o_ids = our_topk[i]  # [T, topk]
        h_ids = hf_topk[i]
        o_sets = [set(row.tolist()) for row in o_ids]
        h_sets = [set(row.tolist()) for row in h_ids]
        mism = torch.tensor([1 if o_sets[t] != h_sets[t] else 0 for t in range(T)])
        mismatch_masks.append(mism)
        rate = mism.float().mean().item()
        print(f"{i:>6}  {rate:>15.4f}  {int(mism.sum().item()):>10d}/{T}")

    target_layer = min(args.target_layer, n_layers - 1)
    print(f"\n{'='*88}")
    print(f"CROSS-REFERENCE AT LAYER {target_layer}: expert-set mismatch vs hidden-state cosine")
    print(f"{'='*88}")
    oh = our_hidden[target_layer][0]  # [T, H]
    hh = hf_hidden[target_layer][0]
    cos = F.cosine_similarity(oh, hh, dim=-1)  # [T]
    mism = mismatch_masks[target_layer]

    low_cos_mask = cos < 0.5
    n_low = int(low_cos_mask.sum().item())
    n_mism = int(mism.sum().item())
    both = int((low_cos_mask & (mism == 1)).sum().item())
    print(f"Positions with hidden-state cosine < 0.5 at layer {target_layer}: {n_low}/{T}")
    print(f"Positions with expert-set mismatch at layer {target_layer}: {n_mism}/{T}")
    print(f"Positions with BOTH low cosine AND expert mismatch: {both}")
    if n_low > 0:
        print(f"  -> {both}/{n_low} ({both/n_low*100:.1f}%) of low-cosine positions "
              f"also had an expert-set mismatch at this layer.")
    if n_mism > 0:
        print(f"  -> {both}/{n_mism} ({both/n_mism*100:.1f}%) of expert-mismatch positions "
              f"also had collapsed cosine similarity.")

    print(f"\nFirst 10 low-cosine positions at layer {target_layer} "
          f"(position, cosine, expert_mismatch, ours_top8, hf_top8):")
    shown = 0
    for t in range(T):
        if low_cos_mask[t] and shown < 10:
            print(f"  pos={t:5d}  cos={cos[t].item():.4f}  mismatch={bool(mism[t])}\n"
                  f"    ours={sorted(our_topk[target_layer][t].tolist())}\n"
                  f"    hf  ={sorted(hf_topk[target_layer][t].tolist())}")
            shown += 1
    if shown == 0:
        print("  (none)")


if __name__ == "__main__":
    main()
