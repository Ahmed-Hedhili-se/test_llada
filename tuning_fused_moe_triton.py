import torch
import triton
import json
import itertools
from model_update.fused_moe_triton import fused_moe, invoke_fused_moe_kernel, moe_align_block_size

def get_config_grid():
    # Power of 2 grid to keep search space reasonable (around ~288 configs per M)
    block_m = [16, 32, 64, 128]
    block_n = [32, 64, 128]
    block_k = [32, 64, 128]
    group_m = [1, 4, 8]
    num_warps = [4, 8]
    num_stages = [2, 3, 4]
    
    configs = []
    for bm, bn, bk, gm, nw, ns in itertools.product(block_m, block_n, block_k, group_m, num_warps, num_stages):
        configs.append({
            'BLOCK_SIZE_M': bm,
            'BLOCK_SIZE_N': bn,
            'BLOCK_SIZE_K': bk,
            'GROUP_SIZE_M': gm,
            'num_warps': nw,
            'num_stages': ns
        })
    return configs

def benchmark_moe_config(M, E, N, K, top_k, config):
    device = torch.device('cuda')
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    # Dummy inputs matching dimensions
    hidden_states = torch.randn((M, K), device=device, dtype=dtype)
    w1 = torch.randn((E, N, K), device=device, dtype=dtype) 
    
    # Random routing
    topk_weights = torch.rand((M, top_k), device=device, dtype=torch.float32)
    topk_ids = torch.randint(0, E, (M, top_k), device=device, dtype=torch.int32)
    
    gating_output = topk_weights
    
    # 1. Align block size
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(topk_ids, config['BLOCK_SIZE_M'], E)
    
    intermediate_cache1 = torch.empty((M, top_k, N), device=device, dtype=dtype)
    
    # Run only the first GEMM (x @ W1) for benchmarking purposes to isolate kernel tuning,
    # since second GEMM is symmetric and uses the exact same config block shape.
    def run_first_gemm():
        invoke_fused_moe_kernel(
            hidden_states, w1, intermediate_cache1,
            gating_output, topk_ids, sorted_token_ids, expert_ids, num_tokens_post_padded,
            False, top_k, config
        )
        
    # Prune configs exceeding GPU shared memory limits (approx 96 KB)
    shmem_bytes = (config['BLOCK_SIZE_M'] * config['BLOCK_SIZE_K'] + config['BLOCK_SIZE_K'] * config['BLOCK_SIZE_N']) * config.get('num_stages', 2) * 2
    if shmem_bytes > 96000:
        return float('inf')

    try:
        ms = triton.testing.do_bench(run_first_gemm, warmup=10, rep=50)
        return ms
    except BaseException:
        # Catch OutOfResources or Triton compilation errors for invalid configs
        return float('inf')

def main():
    m_buckets = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    
    # LLaDA-MoE-7B representative shapes
    E = 64
    N = 2048 # W1 (Gate+Up) output dim (1024 * 2)
    K = 2048 # hidden dim
    top_k = 8
    
    if not torch.cuda.is_available():
        print("CUDA is required to run Triton benchmarks.")
        return

    all_configs = get_config_grid()
    print(f"Total configurations to test per M bucket: {len(all_configs)}\n")
    
    best_configs = {}
    
    for M in m_buckets:
        print(f"--- Tuning for M = {M} ---")
        best_ms = float('inf')
        best_config = None
        
        for i, config in enumerate(all_configs):
            ms = benchmark_moe_config(M, E, N, K, top_k, config)
            if ms < best_ms:
                best_ms = ms
                best_config = config
                
            if (i + 1) % 100 == 0:
                print(f"  Tested {i+1}/{len(all_configs)} configs. Best so far: {best_ms:.4f} ms")
                
        if best_config is not None:
            best_configs[str(M)] = best_config
            print(f"Best config for M={M}: {best_config} (Latency: {best_ms:.4f} ms)\n")
        else:
            print(f"Failed to find a valid config for M={M}.\n")
            
    with open("moe_tune_config.json", "w") as f:
        json.dump(best_configs, f, indent=4)
        
    print("Tuning complete! Saved to moe_tune_config.json")

if __name__ == "__main__":
    main()
