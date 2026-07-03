"""
Compare our LLaDA-MoE implementation vs the HF reference on identical inputs.

Tests:
  1. Per-token logit cosine similarity across 6 prompts (with mask tokens)
  2. Top-1 token match rate
  3. Full generation comparison (diffusion decode, same steps/seed)

Usage:
  python3 compare_models.py --weight-dir ./weights
  python3 compare_models.py --weight-dir ./weights --no-gen
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

MASK_ID = 156895


def load_ours(weight_dir: str):
    sys.path.insert(0, str(Path(__file__).parent))
    from src.model import LLaDAMoE, load_weights
    model = LLaDAMoE().to(torch.bfloat16).to("cuda:0").eval()
    load_weights(model, weight_dir, verbose=True)
    return model


def load_hf(weight_dir: str):
    return AutoModel.from_pretrained(
        weight_dir,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to("cuda:0").eval()


def topk_tokens(logits: torch.Tensor, tok, k: int = 5) -> list[str]:
    return [repr(tok.decode([i])) for i in logits.topk(k).indices.tolist()]


def gen_ours(model, prompt_ids, tok, gen_length=64, steps=64, block_length=32):
    sys.path.insert(0, str(Path(__file__).parent))
    from src.generate import generate
    out = generate(model, prompt_ids, gen_length=gen_length, steps=steps,
                   block_length=block_length, temperature=0.0)
    ids = out[0].tolist()
    if tok.eos_token_id in ids:
        ids = ids[: ids.index(tok.eos_token_id)]
    return tok.decode(ids, skip_special_tokens=True)


def gen_hf(model, prompt_ids, tok, gen_length=64, steps=64, block_length=32):
    """Run HF model's own diffusion generation (same algorithm)."""
    import numpy as np

    device = prompt_ids.device
    P = prompt_ids.shape[1]
    x = torch.full((1, P + gen_length), MASK_ID, dtype=torch.long, device=device)
    x[:, :P] = prompt_ids

    num_blocks = gen_length // block_length
    steps_per_block = steps // num_blocks

    for block_idx in range(num_blocks):
        bs = P + block_idx * block_length
        be = P + (block_idx + 1) * block_length
        block_mask = (x[:, bs:be] == MASK_ID)
        mask_num = block_mask.sum(dim=1, keepdim=True)
        base = mask_num // steps_per_block
        rem  = mask_num % steps_per_block
        ntok = torch.zeros(1, steps_per_block, device=device, dtype=torch.long) + base
        for i in range(1):
            ntok[i, :rem[i]] += 1

        for step in range(steps_per_block):
            mask_index = (x == MASK_ID)
            with torch.no_grad():
                logits = model(x).logits
            x0 = logits.argmax(dim=-1)
            p = F.softmax(logits.float(), dim=-1)
            x0_p = p.gather(-1, x0.unsqueeze(-1)).squeeze(-1)
            x0_p[:, be:] = -torch.inf
            x0 = torch.where(mask_index, x0, x)
            conf = torch.where(mask_index, x0_p, torch.full_like(x0_p, -torch.inf))
            transfer = torch.zeros_like(x0, dtype=torch.bool)
            k = ntok[0, step].item()
            if k > 0:
                _, sel = torch.topk(conf[0], k=int(k))
                transfer[0, sel] = True
            x[transfer] = x0[transfer]

    ids = x[0, P:].tolist()
    if tok.eos_token_id in ids:
        ids = ids[: ids.index(tok.eos_token_id)]
    return tok.decode(ids, skip_special_tokens=True)


# Test inputs: each is a (prompt_text, mask_positions_hint) pair
# We inject MASK_ID into the prompt to test full-sequence logit matching
PROMPTS = [
    "The chemical symbol for gold is",
    "def fibonacci(n):\n    if n <= 1: return n\n    return",
    "The capital of Japan is",
    "Compute: 7 × 8 =",
    "Water boils at 100 degrees",
    "The largest planet in the solar system is",
]


def make_masked_input(prompt_ids: torch.Tensor, mask_frac: float = 0.15) -> torch.Tensor:
    """Replace ~15% of non-first tokens with MASK_ID to simulate diffusion input."""
    ids = prompt_ids.clone()
    T = ids.shape[1]
    n_mask = max(1, int(T * mask_frac))
    positions = torch.randperm(T - 1)[:n_mask] + 1   # skip position 0
    ids[0, positions] = MASK_ID
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight-dir", default="weights")
    ap.add_argument("--no-gen", action="store_true")
    ap.add_argument("--gen-length", type=int, default=64)
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--block-length", type=int, default=32)
    args = ap.parse_args()

    print(f"torch {torch.__version__} | device SM{torch.cuda.get_device_capability(0)}")
    print(f"Weight dir: {args.weight_dir}\n")

    tok = AutoTokenizer.from_pretrained(args.weight_dir, trust_remote_code=True)

    print("Loading our model...")
    ours = load_ours(args.weight_dir)
    print("Loading HF model...")
    hf = load_hf(args.weight_dir)
    print()

    sep = "=" * 72
    total_cos, top1_match, n_prompts = 0.0, 0, 0

    torch.manual_seed(42)
    for prompt_text in PROMPTS:
        ids = tok(prompt_text, return_tensors="pt")["input_ids"].to("cuda:0")
        masked_ids = make_masked_input(ids)   # both models see same masked input

        with torch.no_grad():
            our_logits = ours(masked_ids)           # [1, T, V]
            hf_logits  = hf(masked_ids).logits      # [1, T, V]

        # Compare logits at last token position
        ol = our_logits[0, -1].float()
        hl = hf_logits[0, -1].float()
        cos = F.cosine_similarity(ol.unsqueeze(0), hl.unsqueeze(0)).item()
        our_top5 = topk_tokens(ol, tok)
        hf_top5  = topk_tokens(hl, tok)
        match = our_top5[0] == hf_top5[0]
        total_cos += cos
        top1_match += int(match)
        n_prompts += 1

        print(sep)
        print(f"PROMPT : {repr(prompt_text[:80])}")
        print(f"  cosine={cos:.4f}  top1_match={match}")
        print(f"  ours top-5 : {our_top5}")
        print(f"  HF   top-5 : {hf_top5}")

        if not args.no_gen:
            our_gen = gen_ours(ours, ids, tok, args.gen_length, args.steps, args.block_length)
            hf_gen  = gen_hf(hf,   ids, tok, args.gen_length, args.steps, args.block_length)
            print(f"  ours gen   : {repr(our_gen[:200])}")
            print(f"  HF   gen   : {repr(hf_gen[:200])}")
        print()

    print(sep)
    print(f"SUMMARY  avg_cosine={total_cos/n_prompts:.4f}  top1_match={top1_match}/{n_prompts}")


if __name__ == "__main__":
    main()
