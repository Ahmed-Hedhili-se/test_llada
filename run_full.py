"""
All-in-one: download weights + load full LLaDA-MoE-7B + generate.

Usage (from project root, e.g. on Google Colab):
    python run_full.py
    python run_full.py --prompt "Explain quantum computing in simple terms"
    python run_full.py --gen-length 256 --steps 256

Colab quick-start:
    !pip install torch safetensors transformers==4.53.2 accelerate huggingface_hub
    %cd inference_engine_LLaDA-MoE-7B-A1B-Instruct
    !python run_full.py
"""

import argparse
import os
import time
import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from src.model import LLaDAMoE, load_weights, MASK_ID
from src.generate import generate


REPO_ID = "inclusionAI/LLaDA-MoE-7B-A1B-Instruct"

IGNORE_PATTERNS = [
    "*.msgpack", "*.h5", "flax_model*", "tf_model*",
    "rust_model.ot", "coreml*", "onnx*",
]


def download_weights(dest: str):
    """Download weights from HuggingFace if not already present."""
    index_file = os.path.join(dest, "model.safetensors.index.json")
    if os.path.exists(index_file):
        print(f"  Weights already present at {dest}, skipping download.")
        return

    print(f"  Downloading {REPO_ID} → {dest}")
    print("  (This is ~15 GB, may take a while on first run...)\n")
    os.makedirs(dest, exist_ok=True)
    snapshot_download(
        repo_id=REPO_ID,
        local_dir=dest,
        ignore_patterns=IGNORE_PATTERNS,
    )
    print(f"  Download complete.\n")


def main():
    ap = argparse.ArgumentParser(description="Run full LLaDA-MoE-7B-A1B-Instruct")
    ap.add_argument("--prompt", default="What is machine learning?",
                    help="Input prompt")
    ap.add_argument("--gen-length", type=int, default=128,
                    help="Number of tokens to generate")
    ap.add_argument("--steps", type=int, default=128,
                    help="Number of denoising steps")
    ap.add_argument("--block-length", type=int, default=32,
                    help="Block length for generation")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="Gumbel noise temperature (0 = greedy)")
    ap.add_argument("--cfg-scale", type=float, default=0.0,
                    help="Classifier-free guidance scale (0 = disabled)")
    ap.add_argument("--weight-dir", default="weights",
                    help="Directory for model weights")
    ap.add_argument("--device", default=None,
                    help="Device (default: auto-detect)")
    args = ap.parse_args()

    # ── Auto-detect device and dtype ──────────────────────────────────────────
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    # T4 GPU doesn't support bf16 well — use fp16 for Colab T4, bf16 for A100+
    if device != "cpu" and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_mem / 1024**3
        if "T4" in gpu_name or "T1000" in gpu_name or vram_gb < 20:
            dtype = torch.float16
        else:
            dtype = torch.bfloat16
    else:
        dtype = torch.float32

    print("=" * 60)
    print(" LLaDA-MoE-7B-A1B-Instruct — Full Model Generation")
    print("=" * 60)
    print(f"  Device : {device}")
    if torch.cuda.is_available():
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM   : {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")
    print(f"  Dtype  : {dtype}")
    print()

    # ── Step 1: Download weights ──────────────────────────────────────────────
    print("[1/4] Checking weights...")
    download_weights(args.weight_dir)

    # ── Step 2: Load tokenizer ────────────────────────────────────────────────
    print("[2/4] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.weight_dir,
        trust_remote_code=True,
    )

    # ── Step 3: Load model ────────────────────────────────────────────────────
    print("[3/4] Loading model (this takes ~30-60 seconds)...")
    t0 = time.time()
    model = LLaDAMoE()
    model = model.to(dtype).to(device).eval()
    load_weights(model, args.weight_dir, verbose=True)
    load_time = time.time() - t0
    print(f"  Model loaded in {load_time:.1f}s")

    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated() / 1024**3
        total = torch.cuda.get_device_properties(0).total_mem / 1024**3
        print(f"  VRAM used: {used:.1f} / {total:.1f} GB")
    print()

    # ── Step 4: Generate ──────────────────────────────────────────────────────
    # Ensure gen_length is divisible by block_length
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
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    input_ids = tokenizer(prompt_str, return_tensors="pt")["input_ids"].to(device)

    print(f"[4/4] Generating...")
    print(f"  Prompt: \"{args.prompt}\"")
    print(f"  Prompt tokens: {input_ids.shape[1]}")
    print(f"  gen_length={gen_length}, steps={steps}, "
          f"block_length={block_length}, temp={args.temperature}")
    print()

    t0 = time.time()
    with torch.no_grad():
        out_ids = generate(
            model,
            input_ids,
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

    # ── Output ────────────────────────────────────────────────────────────────
    print("=" * 60)
    print(" GENERATION RESULT")
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
