"""
All-in-one: download + load LLaDA-MoE-7B in INT8 (8-bit) + generate.

Uses HuggingFace's built-in 8-bit quantization via bitsandbytes.
Model weights: ~7.4 GB VRAM (fits on Colab free T4 or 8GB GPUs).

Usage (from project root):
    python run_full_8bit.py
    python run_full_8bit.py --prompt "Explain quantum computing"
    python run_full_8bit.py --gen-length 256 --steps 256

Colab quick-start:
    !pip install torch transformers==4.53.2 accelerate bitsandbytes
    %cd inference_engine_LLaDA-MoE-7B-A1B-Instruct
    !python run_full_8bit.py
"""

import argparse
import time
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModel

# ── Generation constants ──────────────────────────────────────────────────────
MASK_ID = 156895
REPO_ID = "inclusionAI/LLaDA-MoE-7B-A1B-Instruct"


# ── Generation functions (from src/generate.py, adapted for HF model) ────────
def add_gumbel_noise(logits, temperature):
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer = torch.zeros(
        mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64
    ) + base
    for i in range(mask_num.size(0)):
        num_transfer[i, : remainder[i]] += 1
    return num_transfer


@torch.no_grad()
def generate(model, prompt_ids, gen_length=128, steps=128, block_length=128,
             temperature=0.0, cfg_scale=0.0, remasking="low_confidence"):
    """Masked diffusion generation using HF model (returns .logits)."""
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    steps_per_block = steps // num_blocks

    device = prompt_ids.device
    P = prompt_ids.shape[1]

    x = torch.full((1, P + gen_length), MASK_ID, dtype=torch.long, device=device)
    x[:, :P] = prompt_ids
    prompt_index = (x != MASK_ID)

    for block_idx in range(num_blocks):
        block_start = P + block_idx * block_length
        block_end = P + (block_idx + 1) * block_length

        block_mask_index = (x[:, block_start:block_end] == MASK_ID)
        num_transfer = get_num_transfer_tokens(block_mask_index, steps_per_block)

        for step in range(steps_per_block):
            mask_index = (x == MASK_ID)

            if cfg_scale > 0.0:
                un_x = x.clone()
                un_x[prompt_index] = MASK_ID
                x_cat = torch.cat([x, un_x], dim=0)
                logits = model(x_cat).logits           # HF model returns .logits
                logits, un_logits = logits.chunk(2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits               # HF model returns .logits

            logits_with_noise = add_gumbel_noise(logits, temperature)
            x0 = logits_with_noise.argmax(dim=-1)

            if remasking == "low_confidence":
                p = F.softmax(logits.float(), dim=-1)
                x0_p = p.gather(-1, x0.unsqueeze(-1)).squeeze(-1)
            elif remasking == "random":
                x0_p = torch.rand(x0.shape, device=device)
            else:
                raise ValueError(f"Unknown remasking: {remasking}")

            x0_p[:, block_end:] = -torch.inf
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, torch.full_like(x0_p, -torch.inf))

            transfer_index = torch.zeros_like(x0, dtype=torch.bool)
            for j in range(confidence.shape[0]):
                k = num_transfer[j, step].item()
                if k > 0:
                    _, sel = torch.topk(confidence[j], k=int(k))
                    transfer_index[j, sel] = True

            x[transfer_index] = x0[transfer_index]

    return x[:, P:]


def main():
    ap = argparse.ArgumentParser(description="Run LLaDA-MoE-7B in 8-bit (INT8)")
    ap.add_argument("--prompt", default="What is machine learning?")
    ap.add_argument("--gen-length", type=int, default=128)
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--block-length", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--cfg-scale", type=float, default=0.0)
    args = ap.parse_args()

    print("=" * 60)
    print(" LLaDA-MoE-7B-A1B-Instruct — 8-bit (INT8) Inference")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  GPU  : {gpu_name}")
        print(f"  VRAM : {vram_gb:.1f} GB")
    else:
        print("  ⚠ No GPU detected — running on CPU (will be very slow)")
    print()

    # ── Step 1: Load tokenizer ────────────────────────────────────────────────
    print("[1/3] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(REPO_ID, trust_remote_code=True)
    print("  Tokenizer ready.")
    print()

    # ── Step 2: Load model in 8-bit ───────────────────────────────────────────
    print("[2/3] Loading model in 8-bit quantization...")
    print("  (Downloads ~15 GB on first run, then loads in ~7.4 GB VRAM)")
    print()

    t0 = time.time()
    if torch.cuda.is_available():
        model = AutoModel.from_pretrained(
            REPO_ID,
            trust_remote_code=True,
            load_in_8bit=True,
            device_map="auto",
        )
    else:
        # CPU fallback — no quantization, uses fp32
        model = AutoModel.from_pretrained(
            REPO_ID,
            trust_remote_code=True,
            torch_dtype=torch.float32,
        )
    model.eval()
    load_time = time.time() - t0
    print(f"  Model loaded in {load_time:.1f}s")

    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  VRAM used: {used:.1f} / {total:.1f} GB")
    print()

    # ── Step 3: Generate ──────────────────────────────────────────────────────
    gen_length = args.gen_length
    block_length = args.block_length
    if gen_length % block_length != 0:
        gen_length = ((gen_length // block_length) + 1) * block_length

    steps = args.steps
    num_blocks = gen_length // block_length
    if steps % num_blocks != 0:
        steps = num_blocks * max(1, steps // num_blocks)

    messages = [{"role": "user", "content": args.prompt}]
    prompt_str = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False,
    )
    input_ids = tokenizer(prompt_str, return_tensors="pt")["input_ids"].to(model.device)

    print(f"[3/3] Generating...")
    print(f"  Prompt: \"{args.prompt}\"")
    print(f"  Prompt tokens: {input_ids.shape[1]}")
    print(f"  gen_length={gen_length}, steps={steps}, "
          f"block_length={block_length}, temp={args.temperature}")
    print()

    t0 = time.time()
    out_ids = generate(
        model, input_ids,
        gen_length=gen_length,
        steps=steps,
        block_length=block_length,
        temperature=args.temperature,
        cfg_scale=args.cfg_scale,
    )
    elapsed = time.time() - t0

    # ── Decode ────────────────────────────────────────────────────────────────
    generated = out_ids[0].tolist()
    eos_id = tokenizer.eos_token_id
    if eos_id in generated:
        generated = generated[:generated.index(eos_id)]
    while generated and generated[-1] == MASK_ID:
        generated.pop()

    text = tokenizer.decode(generated, skip_special_tokens=True)
    tokens_generated = len(generated)

    print("=" * 60)
    print(" RESULT")
    print("=" * 60)
    print(f"  Tokens: {tokens_generated}")
    if elapsed > 0:
        print(f"  Time:   {elapsed:.2f}s ({tokens_generated / elapsed:.1f} tok/s)")
    print()
    print(text)
    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
