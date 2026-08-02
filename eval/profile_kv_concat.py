"""
Isolates two costs inside Attention.forward (model_update/model.py):
  1. torch.cat([k_prefix, k], dim=2) / same for v  — rebuilding the growing KV cache
  2. F.scaled_dot_product_attention itself

...across increasing prefix (context) lengths, at the real model's shapes
(KVH=16, HD=128, bf16). Run this on the A6000 box:

    python eval/profile_kv_concat.py

Purpose: decide whether swapping torch.cat for a preallocated/in-place KV
cache (e.g. flash_attn_with_kvcache) is worth it, or whether the concat is
already negligible next to the attention math.
"""
import time
import torch
import torch.nn.functional as F

B = 1
KVH = 16
HD = 128
T_ACTIVE = 16  # one denoising step's worth of new K/V (block_length-ish slice)
PREFIX_LENGTHS = [128, 512, 1024, 2048, 4096, 8192]
N_WARMUP = 5
N_ITERS = 50


def make_tensors(prefix_len, device, dtype):
    k_prefix = torch.randn(B, KVH, prefix_len, HD, device=device, dtype=dtype)
    v_prefix = torch.randn(B, KVH, prefix_len, HD, device=device, dtype=dtype)
    q = torch.randn(B, KVH, T_ACTIVE, HD, device=device, dtype=dtype)
    k_new = torch.randn(B, KVH, T_ACTIVE, HD, device=device, dtype=dtype)
    v_new = torch.randn(B, KVH, T_ACTIVE, HD, device=device, dtype=dtype)
    return k_prefix, v_prefix, q, k_new, v_new


def time_fn(fn, device):
    for _ in range(N_WARMUP):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_ITERS):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / N_ITERS * 1000  # ms/iter


def main():
    if not torch.cuda.is_available():
        print("CUDA required for a meaningful measurement (CPU won't reflect flash-attn dispatch).")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    print(f"{'prefix_len':>10} | {'concat (ms)':>12} | {'sdpa (ms)':>12} | {'concat %':>9}")
    print("-" * 52)

    for prefix_len in PREFIX_LENGTHS:
        k_prefix, v_prefix, q, k_new, v_new = make_tensors(prefix_len, device, dtype)

        def do_concat():
            k_full = torch.cat([k_prefix, k_new], dim=2)
            v_full = torch.cat([v_prefix, v_new], dim=2)
            return k_full, v_full

        concat_ms = time_fn(do_concat, device)

        k_full, v_full = do_concat()

        def do_sdpa():
            return F.scaled_dot_product_attention(q, k_full, v_full, attn_mask=None, is_causal=False)

        sdpa_ms = time_fn(do_sdpa, device)

        pct = concat_ms / (concat_ms + sdpa_ms) * 100
        print(f"{prefix_len:>10} | {concat_ms:>12.4f} | {sdpa_ms:>12.4f} | {pct:>8.1f}%")

    if device.type == "cuda":
        backend = torch.backends.cuda.flash_sdp_enabled()
        print(f"\nflash_sdp_enabled: {backend}")
        print("If True and no warnings were printed above, SDPA is very likely using the flash-attention kernel already.")


if __name__ == "__main__":
    main()
