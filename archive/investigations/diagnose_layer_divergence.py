"""
Layer-by-layer hidden-state divergence trace: dminfr.engine vs HF reference,
using a REALISTIC LONG PROMPT (GSM8K's actual 4-shot + question, chat-
templated -- ~1500 tokens, matching exactly what the real correctness
harness sends) instead of compare_models.py's short (11-82 token) test
prompts.

Why this script exists: compare_models.py already showed a stable, real
divergence at the single-forward-pass level (avg_cosine=0.9781, top-1
match 91.0% across 256 masked positions, 8 short prompts) -- but every
generation-loop-level hypothesis for the accuracy gap this points to has
been ruled out (KV-cache staleness: tested directly via generate_dense's
caching on/off ablation, no reproducible effect -- see
INVESTIGATION_LOG.md Part 2 SS2.9-2.10). That leaves the forward pass
itself as the likely source, and since the real accuracy gap shows up on
long-prompt tasks (GSM8K's ~1500-token 4-shot prompt) while
compare_models.py only ever tested prompts under 100 tokens, this traces
the SAME kind of divergence at a length actually representative of where
the gap has been observed.

Method: run the identical [long_prompt | MASK*32] input through both
models with a forward hook on every decoder layer, capturing hidden
states after each layer. Compare cosine similarity layer-by-layer,
separately for the prompt region (positions 0..P-1) and the masked
generation region (positions P..P+31) -- if divergence is already large
within the prompt's own self-attention encoding, before any mask
prediction happens, that points at long-context attention/RoPE handling
specifically rather than something about mask prediction itself.

Usage:
    python archive/investigations/diagnose_layer_divergence.py --weight-dir ./weights
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
    """GSM8K's real 4-shot + question format, chat-templated -- exactly
    what the correctness harness sends (~1500 tokens), not an arbitrary
    long string."""
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


def run_ours_with_hooks(weight_dir, x, eager_moe=False):
    from dminfr.engine.model import LLaDAMoEKV, TritonFusedMoEBlock
    from dminfr.reference.model import load_weights

    print(f"  Loading dminfr.engine ({'eager MoEBlock' if eager_moe else 'Triton fused MoE'})...")
    model = LLaDAMoEKV(use_fused_moe=False).to(torch.bfloat16).eval()
    load_weights(model, weight_dir, verbose=False)
    if not eager_moe:
        for layer in model.layers:
            fused_mlp = TritonFusedMoEBlock(layer.mlp.cfg).to(torch.bfloat16)
            fused_mlp.load_state_dict_from_unfused(layer.mlp)
            layer.mlp = fused_mlp
    model = model.to("cuda:0")

    captured = []

    def hook(module, inp, out, captured=captured):
        # Layer.forward returns (x, kv_new)
        captured.append(out[0].detach().float().cpu())

    hooks = [layer.register_forward_hook(hook) for layer in model.layers]

    x = x.to("cuda:0")
    with torch.no_grad():
        logits, _ = model(x)
    logits = logits.detach().float().cpu()

    for h in hooks:
        h.remove()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return logits, captured


def _find_decoder_layers(model, expected_len):
    """Find the ModuleList of per-layer decoder blocks by EXACT length
    match against dminfr.engine's known layer count, preferring the
    shallowest match -- a naive "longest ModuleList" search would instead
    find a single layer's 64-expert MoE list (longer than the 16-layer
    top-level list), since this is a trust_remote_code model class we
    don't have local source for and can't just hardcode an attribute
    path."""
    candidates = []
    for name, module in model.named_modules():
        if isinstance(module, nn.ModuleList) and len(module) == expected_len:
            candidates.append((name.count("."), name, module))
    if not candidates:
        raise RuntimeError(
            f"Could not find a length-{expected_len} ModuleList on the HF model -- "
            f"architecture may differ from expected. Inspect model.named_modules() manually."
        )
    candidates.sort(key=lambda c: c[0])
    depth, name, module_list = candidates[0]
    return [(f"{name}.{i}", layer) for i, layer in enumerate(module_list)]


def run_hf_with_hooks(weight_dir, x, expected_layers, attn_impl="eager"):
    print(f"  Loading HF reference (attn_implementation={attn_impl})...")
    model = AutoModelForCausalLM.from_pretrained(
        weight_dir, trust_remote_code=True, torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
    ).to("cuda:0").eval()

    layer_modules = _find_decoder_layers(model, expected_layers)
    print(f"  Found {len(layer_modules)} HF decoder-layer modules: "
          f"{layer_modules[0][0]} .. {layer_modules[-1][0]}")

    captured = []

    def hook(module, inp, out, captured=captured):
        hs = out[0] if isinstance(out, tuple) else out
        captured.append(hs.detach().float().cpu())

    hooks = [module.register_forward_hook(hook) for _, module in layer_modules]

    x = x.to("cuda:0")
    with torch.no_grad():
        logits = model(x).logits.detach().float().cpu()

    for h in hooks:
        h.remove()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return logits, captured


def report_region(name, our_hidden, hf_hidden, lo, hi, n_layers):
    print(f"\n{'-'*72}")
    print(f"  {name} region (positions {lo}..{hi-1}, n={hi-lo})")
    print(f"{'-'*72}")
    print(f"  {'layer':>6}  {'avg_cosine':>12}  {'min_cosine':>12}")
    for i in range(n_layers):
        oh = our_hidden[i][0, lo:hi]
        hh = hf_hidden[i][0, lo:hi]
        cos = F.cosine_similarity(oh, hh, dim=-1)
        print(f"  {i:>6}  {cos.mean().item():>12.4f}  {cos.min().item():>12.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weight-dir", default="weights")
    ap.add_argument("--eager-moe", action="store_true",
                    help="Use dminfr.engine's eager MoEBlock instead of converting to the "
                         "Triton fused kernel. The eager block is a near-line-for-line match "
                         "of HF's reference loop (sequential expert-ID order, index_add_ into "
                         "a bf16 accumulator, nn.Linear/cuBLAS experts), so comparing this run "
                         "against the default isolates the Triton MoE kernel's own "
                         "contribution to the divergence.")
    ap.add_argument("--hf-attn", choices=["eager", "sdpa"], default="eager",
                    help="HF attn_implementation. dminfr.engine always uses "
                         "F.scaled_dot_product_attention; loading HF with 'sdpa' aligns the "
                         "attention kernel family on both sides, isolating attention's "
                         "contribution to the divergence (the 90%% expert-set mismatch rate "
                         "already present at layer 0 -- BEFORE any MoE computation runs -- "
                         "can only come from attention/norm/RoPE, so this is the leading "
                         "suspect). The checkpoint ships LLaDAMoESdpaAttention, so 'sdpa' is "
                         "supported.")
    args = ap.parse_args()

    from dminfr.engine.model import FULL_CFG

    tok = AutoTokenizer.from_pretrained(args.weight_dir, trust_remote_code=True)
    prompt_ids, item = build_long_prompt(tok)
    x, P = make_diffusion_input(prompt_ids, GEN_LEN)
    print(f"Long prompt: {P} tokens (GSM8K 4-shot + question, chat-templated)")
    print(f"Expected answer: {item.expected}")
    print(f"Input shape: {tuple(x.shape)}  (prompt + {GEN_LEN} masked positions)\n")

    print(f"Config: ours={'eager MoE' if args.eager_moe else 'Triton fused MoE'} + SDPA attention, "
          f"HF=attn_implementation={args.hf_attn}\n")

    print("Running dminfr.engine (ours) with per-layer hooks...")
    our_logits, our_hidden = run_ours_with_hooks(args.weight_dir, x, eager_moe=args.eager_moe)

    print("Running HF reference with per-layer hooks...")
    hf_logits, hf_hidden = run_hf_with_hooks(args.weight_dir, x, FULL_CFG.NL, attn_impl=args.hf_attn)

    n_layers = min(len(our_hidden), len(hf_hidden))
    if len(our_hidden) != len(hf_hidden):
        print(f"WARNING: layer count mismatch (ours={len(our_hidden)}, HF={len(hf_hidden)}) "
              f"-- comparing the first {n_layers} only.")

    print(f"\n{'='*72}")
    print("PER-LAYER HIDDEN-STATE COSINE SIMILARITY")
    print(f"{'='*72}")
    report_region("PROMPT", our_hidden, hf_hidden, 0, P, n_layers)
    report_region("MASKED/GENERATION", our_hidden, hf_hidden, P, P + GEN_LEN, n_layers)

    # Final logits (same metric as compare_models.py, for continuity)
    cos_scores, matches = [], []
    for pos in range(P, P + GEN_LEN):
        ol = our_logits[0, pos]
        hl = hf_logits[0, pos]
        cos_scores.append(F.cosine_similarity(ol.unsqueeze(0), hl.unsqueeze(0)).item())
        matches.append(ol.argmax().item() == hl.argmax().item())

    print(f"\n{'='*72}")
    print("FINAL LOGITS (long prompt, masked positions) -- compare against")
    print("compare_models.py's short-prompt numbers (avg_cosine=0.9781, top1=91.0%)")
    print(f"{'='*72}")
    print(f"avg_cosine={sum(cos_scores)/len(cos_scores):.4f}  "
          f"top1_match={sum(matches)}/{len(matches)} ({sum(matches)/len(matches)*100:.1f}%)")


if __name__ == "__main__":
    main()
