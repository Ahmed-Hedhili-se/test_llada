import torch
import time
from collections import defaultdict
import os
import sys

# Add parent directory to path so we can import model_update
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_update.model import LLaDAMoEKV, SMALL_CFG, FULL_CFG

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize the model using the small config for quick testing by default.
    # To test with full config, you can change SMALL_CFG to FULL_CFG
    cfg = SMALL_CFG 
    print(f"Initializing model with Config: {cfg}")
    model = LLaDAMoEKV(cfg).to(device)
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
                elapsed = (time.perf_counter() - module._start_time) * 1000 # in ms
                event_records.append((name, elapsed))
        return hook

    for i, layer in enumerate(model.layers):
        layer.self_attn.register_forward_pre_hook(pre_hook('attention'))
        layer.self_attn.register_forward_hook(post_hook('attention'))
        layer.mlp.register_forward_pre_hook(pre_hook('mlp'))
        layer.mlp.register_forward_hook(post_hook('mlp'))

    # Dummy input
    B, T = 2, 128
    print(f"Creating dummy input with Batch Size: {B}, Sequence Length: {T}")
    input_ids = torch.randint(0, cfg.VS, (B, T), device=device)
    
    # Warmup
    print("Warming up...")
    with torch.no_grad():
        for _ in range(3):
            model(input_ids)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

    event_records.clear()

    # Measure
    print("Measuring...")
    num_steps = 10
    start_total = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_steps):
            model(input_ids)
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
    attention_time = timing_data['attention']
    mlp_time = timing_data['mlp']
    
    measured_total = attention_time + mlp_time
    
    print(f"\n--- Timing Results over {num_steps} steps ---")
    print(f"Total time (including overhead): {total_time_ms:.2f} ms")
    print(f"Attention total time: {attention_time:.2f} ms")
    print(f"MoE FFN total time: {mlp_time:.2f} ms")
    
    if measured_total > 0:
        print(f"\nFraction of time (Attention vs MoE FFN):")
        print(f"Attention: {attention_time / measured_total * 100:.2f}%")
        print(f"MoE FFN:   {mlp_time / measured_total * 100:.2f}%")
        
        print(f"\nFraction of total forward pass time:")
        print(f"Attention: {attention_time / total_time_ms * 100:.2f}%")
        print(f"MoE FFN:   {mlp_time / total_time_ms * 100:.2f}%")
        print(f"Other (Embeddings, Norms, LM Head, Overhead): {(total_time_ms - measured_total) / total_time_ms * 100:.2f}%")

if __name__ == '__main__':
    main()
