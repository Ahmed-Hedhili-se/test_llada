import argparse
import torch
import time
from collections import defaultdict
import os
import sys

# Add parent directory to path so we can import dminfr.engine
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dminfr.engine.model import LLaDAMoEKV, SMALL_CFG, FULL_CFG, KVCacheBuffer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=["small", "full"], default="small")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--prefix-len", type=int, default=128)
    parser.add_argument("--active-len", type=int, default=64)
    parser.add_argument("--future-len", type=int, default=0,
                         help="Extra MASK-placeholder tokens after the active block, "
                              "to simulate a non-final block (which also attends over "
                              "still-masked future blocks, per generate.py).")
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    cfg = FULL_CFG if args.config == "full" else SMALL_CFG
    print(f"Initializing model with Config: {cfg}, Fused MoE: True")
    model = LLaDAMoEKV(cfg, use_fused_moe=True).to(device)
    model.eval()

    event_records = []

    def pre_hook(name):
        def hook(module, input):
            if torch.cuda.is_available():
                start = torch.cuda.Event(enable_timing=True)
                start.record()
                module._start_event = start
            else:
                module._start_time = time.perf_counter()
        return hook

    def post_hook(name):
        def hook(module, input, output):
            if torch.cuda.is_available():
                end = torch.cuda.Event(enable_timing=True)
                end.record()
                event_records.append((name, module._start_event, end))
            else:
                elapsed = (time.perf_counter() - module._start_time) * 1000  # ms
                event_records.append((name, elapsed))
        return hook

    def attach(module, name):
        module.register_forward_pre_hook(pre_hook(name))
        module.register_forward_hook(post_hook(name))

    attach(model.embed_tokens, "embed")
    attach(model.norm, "final_norm")
    attach(model.lm_head, "lm_head")
    for layer in model.layers:
        attach(layer.self_attn, "attention")
        attach(layer.mlp, "mlp")
        attach(layer.mlp.gate, "router")  # subset of 'mlp' time, reported separately below

    B = args.batch_size
    P = args.prefix_len
    T_active = args.active_len
    T_future = args.future_len
    total_len = P + T_active + T_future

    print(f"Batch Size: {B}, Prefix: {P}, Active: {T_active}, Future(masked): {T_future}")

    prompt_ids = torch.randint(0, cfg.VS, (B, P), device=device)
    # Active + future positions start as MASK in real generation; random ids are fine
    # for timing purposes since compute cost doesn't depend on token identity.
    gen_ids = torch.randint(0, cfg.VS, (B, T_active + T_future), device=device)

    cache_buffer = KVCacheBuffer(
        num_layers=cfg.NL,
        batch_size=B,
        kvh_local=model.layers[0].self_attn.KVH_local,
        max_len=total_len,
        head_dim=cfg.HD,
        dtype=next(model.parameters()).dtype,
        device=device,
    )

    print("Priming cache with prompt (matches generate_cached's prime call)...")
    with torch.no_grad():
        full_ids = torch.cat([prompt_ids, gen_ids], dim=1)
        model(full_ids, position_offset=0, cache_buffer=cache_buffer, write_pos=0)
    cache_buffer.commit(P)

    active_and_future_ids = full_ids[:, P:]

    print("Warming up active block processing (cache_buffer path)...")
    with torch.no_grad():
        for _ in range(3):
            model(active_and_future_ids, position_offset=P, cache_buffer=cache_buffer, write_pos=P)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

    event_records.clear()

    print("Measuring...")
    num_steps = args.steps
    start_total = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_steps):
            model(active_and_future_ids, position_offset=P, cache_buffer=cache_buffer, write_pos=P)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
    end_total = time.perf_counter()

    timing_data = defaultdict(float)
    if torch.cuda.is_available():
        for name, start, end in event_records:
            timing_data[name] += start.elapsed_time(end)
    else:
        for name, elapsed in event_records:
            timing_data[name] += elapsed

    total_time_ms = (end_total - start_total) * 1000
    attention_time = timing_data["attention"]
    mlp_time = timing_data["mlp"]
    router_time = timing_data["router"]  # subset of mlp_time, shown for context only
    embed_time = timing_data["embed"]
    norm_time = timing_data["final_norm"]
    lm_head_time = timing_data["lm_head"]

    accounted = attention_time + mlp_time + embed_time + norm_time + lm_head_time
    unaccounted = total_time_ms - accounted

    print(f"\n--- Timing Results over {num_steps} steps ---")
    print(f"Total time (wall clock):     {total_time_ms:.2f} ms")
    print(f"Attention:                   {attention_time:.2f} ms")
    print(f"MoE FFN (mlp, incl. router): {mlp_time:.2f} ms  (of which router/gate: {router_time:.2f} ms)")
    print(f"Embedding lookup:            {embed_time:.2f} ms")
    print(f"Final RMSNorm:                {norm_time:.2f} ms")
    print(f"LM head (H -> vocab):        {lm_head_time:.2f} ms")
    print(f"Unaccounted (Python/dispatch/launch overhead): {unaccounted:.2f} ms")

    print(f"\nFraction of total wall-clock time:")
    for label, val in [
        ("Attention", attention_time),
        ("MoE FFN", mlp_time),
        ("Embedding", embed_time),
        ("Final norm", norm_time),
        ("LM head", lm_head_time),
        ("Unaccounted overhead", unaccounted),
    ]:
        print(f"  {label:<22}: {val / total_time_ms * 100:5.2f}%")


if __name__ == "__main__":
    main()
