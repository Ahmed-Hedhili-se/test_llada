"""
OpenAI-compatible chat completions server for LLaDA-MoE-7B-A1B-Instruct.

Backends:
  - "ours"        : Dense baseline (src.generate)
  - "ours_kv"     : Original sparse-dLLM + SparseD (src.generate_KVcache)
  - "fast_dense"  : NEW - Fast dense cached + conservative dynamic experts (Option A)
  - "dyn_experts" : NEW - Sparse path + dynamic expert pruning
  - "hf"          : HuggingFace reference

Usage:
    python3 -m src.server --weight-dir ./weights --port 8000 --backend fast_dense
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

sys.path.insert(0, str(Path(__file__).parent.parent))

app = FastAPI(title="LLaDA-MoE Inference Server")

MODEL: Optional[torch.nn.Module] = None
TOKENIZER = None
DEVICE = "cuda:0"
BACKEND = "ours"

DEFAULT_STEPS        = 128
DEFAULT_GEN_LENGTH   = 128
DEFAULT_BLOCK_LENGTH = 32
DEFAULT_TEMPERATURE  = 0.0
DEFAULT_CFG_SCALE    = 0.0
DEFAULT_REMASKING    = "low_confidence"


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
    steps: int = DEFAULT_STEPS
    block_length: int = DEFAULT_BLOCK_LENGTH
    cfg_scale: float = DEFAULT_CFG_SCALE
    remasking: str = DEFAULT_REMASKING


@app.get("/health")
def health():
    return {"status": "ok", "model": "LLaDA-MoE-7B-A1B-Instruct", "backend": BACKEND}


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{"id": "inclusionAI/LLaDA-MoE-7B-A1B-Instruct", "object": "model"}],
    }


# ── Config switching endpoint (for automated testing) ──────────────────────────
@app.post("/v1/config")
def set_config(config: dict):
    """Switch generation config at runtime (for testing only)."""
    global BACKEND
    new_backend = config.get("backend")
    if new_backend in ["ours", "ours_kv", "fast_dense", "dyn_experts", "hf"]:
        BACKEND = new_backend
        return {"status": "ok", "backend": BACKEND}
    return JSONResponse(
        status_code=400,
        content={"error": f"Unknown backend: {new_backend}. Valid: ours, ours_kv, fast_dense, dyn_experts, hf"}
    )


@app.get("/v1/config")
def get_config():
    return {"backend": BACKEND}


# ── Chat completions ───────────────────────────────────────────────────────────
@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    messages = [{"role": m.role, "content": m.content} for m in req.messages]

    prompt = TOKENIZER.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    input_ids = TOKENIZER(prompt, return_tensors="pt")["input_ids"].to(DEVICE)

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
            from src.generate import generate
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
            from src.generate_KVcache import generate_cached as generate_kv
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
        elif BACKEND == "fast_dense":
            # Option A: Fast dense cached + conservative dynamic experts
            from model_update.generate import generate_dense_cached
            out_ids = generate_dense_cached(
                MODEL,
                input_ids,
                gen_length=gen_length,
                steps=steps,
                block_length=block_length,
                temperature=req.temperature,
                cache_budget=2048,
                use_dynamic_experts=True,
                base_k=8,
                min_k=4,
                expert_threshold=0.03,
            )
        elif BACKEND == "dyn_experts":
            # Sparse path + dynamic expert pruning
            from model_update.generate import generate_sparse_cached
            out_ids = generate_sparse_cached(
                MODEL,
                input_ids,
                gen_length=gen_length,
                steps=steps,
                block_length=block_length,
                temperature=req.temperature,
                cache_budget=2048,
                saliency_update_interval=8,
                sparse_pattern=None,
                use_dynamic_experts=True,
                base_k=8,
                min_k=4,
                expert_threshold=0.03,
            )
        elif BACKEND == "hf":
            # Use standard HuggingFace generate
            out_ids = MODEL.generate(
                input_ids,
                max_new_tokens=gen_length,
                do_sample=False,
                temperature=req.temperature if req.temperature > 0 else None,
                top_p=req.top_p if req.temperature > 0 else None,
                pad_token_id=TOKENIZER.pad_token_id,
                eos_token_id=TOKENIZER.eos_token_id,
            )
    elapsed = time.time() - t0

    generated = out_ids[0].tolist()
    eos_id = TOKENIZER.eos_token_id
    if eos_id in generated:
        generated = generated[: generated.index(eos_id)]
    while generated and generated[-1] == 156895:  # MASK_ID
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
        from src.model import LLaDAMoE, load_weights
        MODEL = LLaDAMoE().to(torch.bfloat16).to(device).eval()
        load_weights(MODEL, weight_dir, verbose=True)
    elif backend in ("fast_dense", "dyn_experts"):
        from model_update.model import LLaDAMoE, load_weights
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
    ap.add_argument("--backend", choices=["ours", "ours_kv", "fast_dense", "dyn_experts", "hf"], default="ours")
    args = ap.parse_args()

    load_model(args.weight_dir, args.device, args.backend)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()