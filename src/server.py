"""
OpenAI-compatible chat completions server for LLaDA-MoE-7B-A1B-Instruct.

Differences from a standard AR server:
  - Generation is masked diffusion, not token-by-token
  - No streaming (all tokens generated at once per request)
  - temperature maps to Gumbel noise, top_p is ignored
  - max_tokens controls gen_length

Usage:
    python3 -m src.server --weight-dir ./weights --port 8000
"""

import argparse
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import AutoTokenizer

# Add parent dir so `src.model` resolves when run as module
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.model import LLaDAMoE, load_weights
from src.generate import generate, MASK_ID

app = FastAPI(title="LLaDA-MoE Inference Server")

# Globals set at startup
MODEL: Optional[LLaDAMoE] = None
TOKENIZER = None
DEVICE = "cuda:0"

# Generation defaults
DEFAULT_STEPS        = 128
DEFAULT_GEN_LENGTH   = 128
DEFAULT_BLOCK_LENGTH = 32
DEFAULT_TEMPERATURE  = 0.0
DEFAULT_CFG_SCALE    = 0.0
DEFAULT_REMASKING    = "low_confidence"


# ── Request / response schemas ─────────────────────────────────────────────────
class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "inclusionAI/LLaDA-MoE-7B-A1B-Instruct"
    messages: list[Message]
    max_tokens: int = DEFAULT_GEN_LENGTH
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    # LLaDA-specific (passed via extra_body or ignored)
    steps: int = DEFAULT_STEPS
    block_length: int = DEFAULT_BLOCK_LENGTH
    cfg_scale: float = DEFAULT_CFG_SCALE
    remasking: str = DEFAULT_REMASKING


# ── Health ─────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model": "LLaDA-MoE-7B-A1B-Instruct"}


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{"id": "inclusionAI/LLaDA-MoE-7B-A1B-Instruct", "object": "model"}],
    }


# ── Chat completions ───────────────────────────────────────────────────────────
@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    messages = [{"role": m.role, "content": m.content} for m in req.messages]

    # Apply chat template → prompt string
    prompt = TOKENIZER.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    input_ids = TOKENIZER(prompt, return_tensors="pt")["input_ids"].to(DEVICE)

    # Clamp gen_length to a multiple of block_length
    gen_length   = req.max_tokens
    block_length = req.block_length
    if gen_length % block_length != 0:
        gen_length = ((gen_length // block_length) + 1) * block_length

    steps = req.steps
    if steps % (gen_length // block_length) != 0:
        steps = (gen_length // block_length) * max(1, steps // (gen_length // block_length))

    t0 = time.time()
    with torch.no_grad():
        if BACKEND == "ours":
            out_ids = generate(
                MODEL,
                input_ids,
                gen_length=gen_length,
                steps=steps,
                block_length=block_length,
                temperature=req.temperature,
                cfg_scale=req.cfg_scale,
                remasking=req.remasking,
            )
        elif BACKEND == "ours_kv":
            from src.generate_KVcache import generate as generate_kv
            out_ids = generate_kv(
                MODEL,
                input_ids,
                gen_length=gen_length,
                steps=steps,
                block_length=block_length,
                temperature=req.temperature,
                cfg_scale=req.cfg_scale,
                remasking=req.remasking,
            )
        elif BACKEND == "hf":
            from eval.check_time_inference import diffusion_generate
            out_ids = diffusion_generate(
                MODEL,
                input_ids,
                gen_length=gen_length,
                steps=steps,
                block_length=block_length,
                is_hf=True
            ).unsqueeze(0)
    elapsed = time.time() - t0

    # Decode — stop at first EOS if present, trim trailing masks
    generated = out_ids[0].tolist()
    eos_id = TOKENIZER.eos_token_id
    if eos_id in generated:
        generated = generated[: generated.index(eos_id)]
    # Remove trailing mask tokens
    while generated and generated[-1] == MASK_ID:
        generated.pop()

    text = TOKENIZER.decode(generated, skip_special_tokens=True)
    tokens_generated = len(generated)

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": input_ids.shape[1],
            "completion_tokens": tokens_generated,
            "total_tokens": input_ids.shape[1] + tokens_generated,
        },
        "timing": {"generation_seconds": round(elapsed, 3)},
    }


# ── Startup ────────────────────────────────────────────────────────────────────
def load_model(weight_dir: str, device: str, backend: str):
    global MODEL, TOKENIZER, DEVICE, BACKEND
    DEVICE = device
    BACKEND = backend

    print(f"Loading tokenizer from {weight_dir}...")
    TOKENIZER = AutoTokenizer.from_pretrained(weight_dir, trust_remote_code=True)

    print(f"Loading model with backend '{backend}'...")
    if backend == "ours":
        MODEL = LLaDAMoE().to(torch.bfloat16).to(device).eval()
        load_weights(MODEL, weight_dir, verbose=True)
    elif backend == "ours_kv":
        from src.Model_KVcache import LLaDAMoEKV
        MODEL = LLaDAMoEKV().to(torch.bfloat16).to(device).eval()
        load_weights(MODEL, weight_dir, verbose=True)
    elif backend == "hf":
        from transformers import AutoModelForCausalLM
        MODEL = AutoModelForCausalLM.from_pretrained(
            weight_dir,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
        ).to(device).eval()
    else:
        raise ValueError(f"Unknown backend: {backend}")
    print("Model ready.\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight-dir", default="weights")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--backend", choices=["ours", "ours_kv", "hf"], default="ours")
    args = ap.parse_args()

    load_model(args.weight_dir, args.device, args.backend)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
