"""
OpenAI-compatible chat completions server for LLaDA-MoE-7B-A1B-Instruct.

Backends:
  - "ours"        : Dense baseline (src.generate)
  - "ours_kv"     : Original sparse-dLLM + SparseD (src.generate_KVcache)
  - "fast_dense"  : Fast dense cached (Option A) — TP+EP, Triton fused MoE,
                    block-wise KV cache. Static top-8 experts (the model's
                    native routing config).
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
import threading
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

DEFAULT_STEPS        = 512
DEFAULT_GEN_LENGTH   = 512
DEFAULT_BLOCK_LENGTH = 32
DEFAULT_TEMPERATURE  = 0.0
DEFAULT_CFG_SCALE    = 0.0
DEFAULT_REMASKING    = "low_confidence"

request_lock = threading.Lock()


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
    if new_backend in ["ours", "ours_kv", "fast_dense", "hf"]:
        BACKEND = new_backend
        return {"status": "ok", "backend": BACKEND}
    return JSONResponse(
        status_code=400,
        content={"error": f"Unknown backend: {new_backend}. Valid: ours, ours_kv, fast_dense, hf"}
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
    with torch.no_grad(), request_lock:
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
            from model_update.model import LLaDAMoEKV
            from model_update.generate import generate_cached
            from model_update.distributed import get_tp_size

            if get_tp_size() > 1:
                import torch.distributed as dist
                req_obj = {
                    "type": "generate_cached",
                    "input_ids": input_ids.cpu(),
                    "gen_length": gen_length,
                    "steps": steps,
                    "block_length": block_length,
                    "temperature": req.temperature,
                }
                dist.broadcast_object_list([req_obj], src=0)

            out_ids = generate_cached(
                MODEL,
                input_ids,
                gen_length=gen_length,
                steps=steps,
                block_length=block_length,
                temperature=req.temperature,
            )
        elif BACKEND == "hf":
            # HuggingFace's .generate() blocks diffusion models in newer versions.
            # We wrap the HF model to return logits and use our diffusion generate loop.
            from src.generate import generate
            class HFWrapper:
                def __init__(self, m): self.m = m
                def __call__(self, x, **kwargs): return self.m(x, **kwargs).logits
            
            out_ids = generate(
                HFWrapper(MODEL),
                input_ids,
                gen_length=gen_length,
                steps=steps,
                block_length=block_length,
                temperature=req.temperature,
                cfg_scale=req.cfg_scale,
                remasking=req.remasking,
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
    
    if "cuda" in device:
        torch.cuda.set_device(device)

    print(f"Loading tokenizer from {weight_dir}...")
    TOKENIZER = AutoTokenizer.from_pretrained(weight_dir, trust_remote_code=True)

    print(f"Loading model with backend '{backend}'...")
    if backend == "ours":
        from src.model import LLaDAMoE, load_weights
        MODEL = LLaDAMoE().to(torch.bfloat16).to(device).eval()
        load_weights(MODEL, weight_dir, verbose=True)
    elif backend == "fast_dense":
        from model_update.model import LLaDAMoEKV, TritonFusedMoEBlock
        from model_update.distributed import load_weights_tp, get_tp_rank

        print(f"Instantiating unfused model to load weights on Rank {get_tp_rank()}...")
        MODEL = LLaDAMoEKV(use_fused_moe=False).to(torch.bfloat16).eval()
        load_weights_tp(MODEL, weight_dir, verbose=True)

        print("Converting to Fused MoE blocks...")
        for i, layer in enumerate(MODEL.layers):
            fused_mlp = TritonFusedMoEBlock(layer.mlp.cfg).to(torch.bfloat16)
            fused_mlp.load_state_dict_from_unfused(layer.mlp)
            layer.mlp = fused_mlp

        MODEL = MODEL.to(device)
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
    from model_update.distributed import init_distributed, get_tp_rank, get_tp_size
    init_distributed()
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight-dir", default="weights")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="0.0.0.0")
    # In distributed mode, we override device with local rank
    ap.add_argument("--device", default=f"cuda:{get_tp_rank()}" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--backend", choices=["ours", "ours_kv", "fast_dense", "hf"], default="ours")
    args = ap.parse_args()

    # Override device if TP is active
    if get_tp_size() > 1:
        args.device = f"cuda:{get_tp_rank()}"

    load_model(args.weight_dir, args.device, args.backend)
    
    if get_tp_size() > 1 and get_tp_rank() != 0:
        print(f"Rank {get_tp_rank()} waiting for generation tasks...")
        worker_loop()
    else:
        uvicorn.run(app, host=args.host, port=args.port)


def worker_loop():
    import torch.distributed as dist
    while True:
        objs = [None]
        dist.broadcast_object_list(objs, src=0)
        req = objs[0]
        if req is None:
            continue
        
        if req["type"] == "generate_cached":
            from model_update.generate import generate_cached
            generate_cached(
                MODEL,
                req["input_ids"].to(DEVICE),
                gen_length=req["gen_length"],
                steps=req["steps"],
                block_length=req["block_length"],
                temperature=req["temperature"],
            )



if __name__ == "__main__":
    main()
