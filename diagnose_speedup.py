import os
import json
import torch
import time

def check_env():
    print("=== Environment ===")
    print(f"PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU: NOT FOUND")

def check_json():
    print("\n=== Config JSON ===")
    config_path = "moe_tune_config.json"
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            data = json.load(f)
        print("Found moe_tune_config.json:")
        for k, v in data.items():
            print(f"  M={k}: {v}")
    else:
        print("No moe_tune_config.json found.")

def check_kernel_config():
    from model_update.fused_moe_triton import get_best_config
    
    print("\n=== Active Triton Config for M=16, 64, 128 (Benchmarked M) ===")
    for M in [16, 64, 128]:
        config = get_best_config(M, 16)
        print(f"\nM={M}: Raw returned config: {config}")
        if config:
            bm = config.get('BLOCK_SIZE_M', 64)
            bn = config.get('BLOCK_SIZE_N', 64)
            bk = config.get('BLOCK_SIZE_K', 32)
            ns = config.get('num_stages', 2)
            shmem = (bm * bk + bk * bn) * ns * 2
            print(f"Calculated Shared Memory: {shmem} bytes (Hardware Limit: ~101376 bytes)")

def microbenchmark_kernel():
    print("\n=== Microbenchmark Fused MoE vs PyTorch Dense ===")
    # Simulate a single MoE layer forward pass during generation (batch=1)
    M, E, EI, H, top_k = 64, 16, 256, 512, 5
    N = EI * 2 # Gate+Up combined
    device = torch.device('cuda')
    dtype = torch.bfloat16
    
    hidden_states = torch.randn((M, H), device=device, dtype=dtype)
    w1 = torch.randn((E, N, H), device=device, dtype=dtype) 
    w2 = torch.randn((E, H, EI), device=device, dtype=dtype) 
    
    topk_weights = torch.rand((M, top_k), device=device, dtype=torch.float32)
    topk_ids = torch.randint(0, E, (M, top_k), device=device, dtype=torch.int32)
    
    from model_update.fused_moe_triton import fused_moe
    
    # 1. Triton Benchmark
    try:
        # Warmup
        for _ in range(5):
            fused_moe(hidden_states, w1, w2, topk_weights, topk_ids)
            
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(100):
            fused_moe(hidden_states, w1, w2, topk_weights, topk_ids)
        torch.cuda.synchronize()
        triton_ms = (time.perf_counter() - t0) * 1000 / 100
        print(f"Triton Fused MoE latency (M={M}, topk={top_k}): {triton_ms:.3f} ms")
    except Exception as e:
        print(f"Triton Kernel Failed: {e}")

def main():
    if not torch.cuda.is_available():
        print("CUDA required.")
        return
    check_env()
    check_json()
    check_kernel_config()
    microbenchmark_kernel()

if __name__ == "__main__":
    main()
