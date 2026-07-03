"""
Compare our LLaDA-MoE implementation vs the HF reference on identical inputs.

LLaDA-MoE is a diffusion LM: meaningful logits come from feeding sequences
that contain MASK tokens. We test by building [prompt | MASK...MASK] inputs
and comparing the logits at the first masked position.

Tests:
  1. Logit cosine similarity at the first masked position (6 prompts)
  2. Top-1 token match rate at that position
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
from transformers import AutoTokenizer, AutoModelForCausalLM

MASK_ID = 156895
GEN_LEN = 32   # short generation suffix appended for logit comparison


def load_ours(weight_dir: str):
    sys.path.insert(0, str(Path(__file__).parent))
    from src.model import LLaDAMoE, load_weights
    model = LLaDAMoE().to(torch.bfloat16).to("cuda:0").eval()
    load_weights(model, weight_dir, verbose=True)
    return model


def load_hf(weight_dir: str):
    """Load HF reference model. Requires transformers==4.53.x (not 5.x)."""
    return AutoModelForCausalLM.from_pretrained(
        weight_dir,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to("cuda:0").eval()


def topk_tokens(logits: torch.Tensor, tok, k: int = 5) -> list[str]:
    return [repr(tok.decode([i])) for i in logits.topk(k).indices.tolist()]


def make_diffusion_input(prompt_ids: torch.Tensor, gen_length: int) -> tuple[torch.Tensor, int]:
    """Build [prompt | MASK * gen_length]. Returns (input_ids, prompt_len)."""
    P = prompt_ids.shape[1]
    x = torch.full((1, P + gen_length), MASK_ID, dtype=torch.long, device=prompt_ids.device)
    x[:, :P] = prompt_ids
    return x, P


def diffusion_generate(model, prompt_ids, gen_length=64, steps=64, block_length=32,
                        temperature=0.0, is_hf=False):
    """Run the masked diffusion decode loop, works for both our model and HF."""
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
                logits = model(x).logits if is_hf else model(x)
            x0 = logits.argmax(dim=-1)
            p = F.softmax(logits.float(), dim=-1)
            x0_p = p.gather(-1, x0.unsqueeze(-1)).squeeze(-1)
            x0_p[:, be:] = -np.inf
            x0 = torch.where(mask_index, x0, x)
            conf = torch.where(mask_index, x0_p, torch.tensor(-np.inf, device=device))
            transfer = torch.zeros_like(x0, dtype=torch.bool)
            k = ntok[0, step].item()
            if k > 0:
                _, sel = torch.topk(conf[0], k=int(k))
                transfer[0, sel] = True
            x[transfer] = x0[transfer]

    return x[0, P:]


# Longer, more demanding prompts that produce non-trivial multi-token completions
PROMPTS = [
    # factual
    "The chemical symbol for gold is Au and for silver is",
    "The capital of Japan is Tokyo and the capital of France is",
    # math reasoning
    "If a train travels at 80 km/h for 2.5 hours, it covers a distance of",
    "The area of a circle with radius 7 is pi times 7 squared, which equals approximately",
    # code completion (longer context)
    "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left  = [x for x in arr if x < pivot]\n    mid   = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return",
    # multi-sentence context
    "Einstein proposed that energy and mass are equivalent, expressed as E=mc². This means that a small amount of mass can be converted into",
    # longer instruction-style
    "Translate the following English sentence to French: 'The sun rises in the east and sets in the west.'",
    "Write a Python function that returns the nth prime number. def nth_prime(n):",
]


def compare_all_masked_positions(our_logits, hf_logits, P, gen_len, tok):
    """
    Compare logits at EVERY masked position (positions P..P+gen_len-1).
    Returns per-position cosine similarity and top-1 match, plus aggregates.
    """
    cos_scores, matches = [], []
    for pos in range(P, P + gen_len):
        ol = our_logits[0, pos].float()
        hl = hf_logits[0, pos].float()
        cos = F.cosine_similarity(ol.unsqueeze(0), hl.unsqueeze(0)).item()
        match = ol.argmax().item() == hl.argmax().item()
        cos_scores.append(cos)
        matches.append(match)
    return cos_scores, matches


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
    hf = load_hf(args.weight_dir)  # noqa: requires transformers==4.53.x
    print()

    sep = "=" * 72
    all_cos, all_matches = [], []

    torch.manual_seed(42)
    for prompt_text in PROMPTS:
        prompt_ids = tok(prompt_text, return_tensors="pt")["input_ids"].to("cuda:0")
        x, P = make_diffusion_input(prompt_ids, GEN_LEN)

        with torch.no_grad():
            our_logits = ours(x)         # [1, P+GEN_LEN, V]
            hf_logits  = hf(x).logits   # [1, P+GEN_LEN, V]

        cos_scores, matches = compare_all_masked_positions(our_logits, hf_logits, P, GEN_LEN, tok)
        avg_cos   = sum(cos_scores) / len(cos_scores)
        top1_rate = sum(matches) / len(matches)
        all_cos.extend(cos_scores)
        all_matches.extend(matches)

        # Show first-mask-pos top-5 for qualitative check
        ol0 = our_logits[0, P].float()
        hl0 = hf_logits[0, P].float()

        print(sep)
        print(f"PROMPT : {repr(prompt_text[:90])}")
        print(f"  input  : [prompt({P} tok) | MASK×{GEN_LEN}]  — comparing all {GEN_LEN} masked positions")
        print(f"  avg_cosine={avg_cos:.4f}  top1_match={sum(matches)}/{GEN_LEN} ({top1_rate*100:.0f}%)")
        print(f"  cosine range : [{min(cos_scores):.4f}, {max(cos_scores):.4f}]")
        print(f"  pos[0] ours top-5 : {topk_tokens(ol0, tok)}")
        print(f"  pos[0] HF   top-5 : {topk_tokens(hl0, tok)}")

        if not args.no_gen:
            our_gen = diffusion_generate(ours, prompt_ids, args.gen_length, args.steps,
                                          args.block_length, is_hf=False)
            hf_gen  = diffusion_generate(hf,   prompt_ids, args.gen_length, args.steps,
                                          args.block_length, is_hf=True)
            our_ids = our_gen.tolist()
            hf_ids  = hf_gen.tolist()
            if tok.eos_token_id in our_ids: our_ids = our_ids[:our_ids.index(tok.eos_token_id)]
            if tok.eos_token_id in hf_ids:  hf_ids  = hf_ids[:hf_ids.index(tok.eos_token_id)]
            print(f"  ours gen : {repr(tok.decode(our_ids, skip_special_tokens=True)[:200])}")
            print(f"  HF   gen : {repr(tok.decode(hf_ids, skip_special_tokens=True)[:200])}")
        print()

    print(sep)
    overall_cos   = sum(all_cos) / len(all_cos)
    overall_top1  = sum(all_matches) / len(all_matches)
    print(f"OVERALL  positions={len(all_cos)}")
    print(f"         avg_cosine={overall_cos:.4f}")
    print(f"         top1_match={sum(all_matches)}/{len(all_matches)} ({overall_top1*100:.1f}%)")


if __name__ == "__main__":
    main()
