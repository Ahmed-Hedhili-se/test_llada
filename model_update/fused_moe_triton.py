import os
import json
import torch
import triton
import triton.language as tl
from typing import Any, Dict, Optional, Tuple

TUNED_CONFIGS = {}
config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "moe_tune_config.json")
try:
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            TUNED_CONFIGS = json.load(f)
except Exception as e:
    print(f"Warning: Failed to load {config_path}: {e}")

def get_best_config(M: int, E: int) -> Dict[str, Any]:
    config = None
    if TUNED_CONFIGS:
        m_keys = [int(k) for k in TUNED_CONFIGS.keys()]
        closest_m = min(m_keys, key=lambda k: abs(k - M))
        candidate = TUNED_CONFIGS[str(closest_m)].copy()
        
        # Calculate shared memory requirement
        bm = candidate.get('BLOCK_SIZE_M', 64)
        bn = candidate.get('BLOCK_SIZE_N', 64)
        bk = candidate.get('BLOCK_SIZE_K', 32)
        ns = candidate.get('num_stages', 2)
        shmem = (bm * bk + bk * bn) * ns * 2
        
        if shmem <= 96000:
            config = candidate
        else:
            # Try reducing num_stages to 2
            candidate['num_stages'] = 2
            shmem_reduced = (bm * bk + bk * bn) * 2 * 2
            if shmem_reduced <= 96000:
                config = candidate

    if config is None:
        if M <= E:
            config = {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 2}
        else:
            config = {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'num_warps': 4, 'num_stages': 2}
            
    return config

@triton.jit
def fused_moe_kernel(
        a_ptr, b_ptr, c_ptr,
        a_scale_ptr, b_scale_ptr,
        topk_weights_ptr, sorted_token_ids_ptr, expert_ids_ptr,
        num_tokens_post_padded_ptr,
        N, K, EM, num_valid_tokens,
        stride_am, stride_ak,
        stride_be, stride_bk, stride_bn,
        stride_cm, stride_cn,
        stride_bse, stride_bsn,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr, MUL_ROUTED_WEIGHT: tl.constexpr, top_k: tl.constexpr,
        compute_type: tl.constexpr, use_fp8_w8a8: tl.constexpr, use_int8_w8a16: tl.constexpr,
        is_first_gemm: tl.constexpr):
    
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    token_mask = offs_token < num_valid_tokens

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    if is_first_gemm:
        a_ptrs = a_ptr + (offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak)
    else:
        a_ptrs = a_ptr + (offs_token[:, None] * stride_am + offs_k[None, :] * stride_ak)

    off_experts = tl.load(expert_ids_ptr + pid_m)
    b_ptrs = b_ptr + off_experts * stride_be + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K), other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        accumulator = accumulator * moe_weight[:, None]

    accumulator = accumulator.to(compute_type)

    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)

def moe_align_block_size(topk_ids: torch.Tensor, block_size: int, num_experts: int):
    """
    Sort/pad (token, expert) assignment pairs into block_size-aligned
    per-expert groups for the grouped-GEMM kernel below.

    Fully zero-host-sync: output shapes depend only on topk_ids.shape and
    the static block_size/num_experts config, never on runtime tensor
    VALUES. This matters beyond raw speed -- it's what makes this function
    (and therefore the whole MoE forward) safe to capture as a CUDA graph,
    which requires every tensor allocation's size to be known before
    replay, not read back from a data-dependent value mid-capture.

    History: the original implementation looped over `range(num_experts)`
    in Python with 2 x num_experts `.item()` calls per call (128 for
    num_experts=64). A first vectorization pass (cumsum + scatter, no
    per-expert loop) got that down to a single `.item()` -- sizing
    `sorted_token_ids` to the EXACT padded length, which is genuinely
    data-dependent. This version removes that last sync too, by allocating
    a WORST-CASE-sized buffer instead: the maximum possible padded length
    regardless of how tokens are distributed across experts (padding adds
    at most block_size-1 per expert, so `ceil(num_valid_tokens/block_size)
    + num_experts` blocks is always enough, computed from static shape
    info alone -- no tensor read required). The Triton kernel already
    reads num_tokens_post_padded from GPU memory at kernel-launch time
    (`tl.load(num_tokens_post_padded_ptr)`), not from Python, so grid
    positions beyond the real (smaller) padded length just early-return --
    correctness is unaffected, at the cost of a few redundant, cheap,
    parallel, immediately-returning grid programs.

    Verified numerically identical to the exact-sized version for any
    input, up to the point where the real data ends (see
    eval/test_moe_align_block_size.py).

    Note: expert_counts intentionally uses zeros().scatter_add_(), not
    torch.bincount(), despite computing the exact same thing. Found on real
    hardware (not something code inspection could have caught): bincount's
    CUDA implementation raises "operation not permitted when stream is
    capturing" inside torch.cuda.graph() -- it goes through an internal
    library path with its own stream/allocation behavior that isn't
    graph-capture-safe, unlike a plain scatter kernel. Every other op in
    this function was checked against real capture and is fine.
    """
    num_tokens, top_k = topk_ids.shape
    device = topk_ids.device
    num_valid_tokens = num_tokens * top_k

    flatten_ids = topk_ids.flatten()
    sorted_indices = torch.argsort(flatten_ids, stable=True)
    sorted_expert_ids = flatten_ids[sorted_indices]

    expert_counts = torch.zeros(num_experts, dtype=torch.long, device=device)
    expert_counts.scatter_add_(
        0, sorted_expert_ids.long(), torch.ones_like(sorted_expert_ids, dtype=torch.long)
    )
    padded_expert_counts = ((expert_counts + block_size - 1) // block_size) * block_size

    # Exclusive cumulative offsets (real and padded) per expert.
    real_offsets = torch.cumsum(expert_counts, dim=0) - expert_counts
    padded_offsets = torch.cumsum(padded_expert_counts, dim=0) - padded_expert_counts

    # sorted_indices/sorted_expert_ids are already grouped by expert (the
    # stable sort above), so a real pair's rank within its own expert's
    # group is just its position minus that expert's real (unpadded)
    # starting offset -- no per-expert loop needed to compute it.
    positions = torch.arange(num_valid_tokens, device=device)
    rank_within_expert = positions - real_offsets[sorted_expert_ids]
    dest = padded_offsets[sorted_expert_ids] + rank_within_expert

    # Fixed upper bound from STATIC shape info only (num_valid_tokens,
    # num_experts, block_size are all Python ints derived from .shape /
    # config, never a tensor read) -- no .item(), no data dependence.
    max_blocks = -(-num_valid_tokens // block_size) + num_experts  # ceil(num_valid_tokens/block_size) + num_experts
    max_padded_len = max_blocks * block_size

    sorted_token_ids = torch.full(
        (max_padded_len,), num_valid_tokens, dtype=sorted_indices.dtype, device=device
    )
    sorted_token_ids.scatter_(0, dest, sorted_indices)

    # Map each of the max_blocks possible block-positions to the expert
    # that owns it. block_start_per_expert is a monotonically
    # non-decreasing, block-size-aligned cumsum, so a single searchsorted
    # stands in for what would otherwise need a per-expert range-fill loop:
    # for block position b, the owning expert is the last one whose start
    # is <= b. Position 0 always maps to expert 0 (block_start_per_expert[0]
    # is always 0 by construction of the exclusive cumsum), so this never
    # goes negative in practice; .clamp is defensive, not load-bearing.
    block_start_per_expert = padded_offsets // block_size
    block_positions = torch.arange(max_blocks, device=device)
    expert_ids = (torch.searchsorted(block_start_per_expert, block_positions, right=True) - 1)
    expert_ids = expert_ids.clamp(min=0).to(torch.int32)

    # Real (data-dependent) total, computed and consumed entirely on-device
    # -- the kernel reads this via tl.load, never the host.
    num_tokens_post_padded = padded_expert_counts.sum(dtype=torch.int32).view(1)

    return sorted_token_ids, expert_ids, num_tokens_post_padded

def invoke_fused_moe_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor,
                            topk_weights: torch.Tensor, topk_ids: torch.Tensor,
                            sorted_token_ids: torch.Tensor, expert_ids: torch.Tensor,
                            num_tokens_post_padded: torch.Tensor,
                            mul_routed_weight: bool, top_k: int, config: Dict[str, Any]) -> None:
    compute_type = tl.bfloat16 if A.dtype == torch.bfloat16 else tl.float16
    grid = lambda META: (triton.cdiv(sorted_token_ids.shape[0], META['BLOCK_SIZE_M']) * triton.cdiv(B.shape[1], META['BLOCK_SIZE_N']), )
    is_first_gemm = not mul_routed_weight
    
    kernel_kwargs = config.copy()
    num_warps = kernel_kwargs.pop('num_warps', 4)
    num_stages = kernel_kwargs.pop('num_stages', 2)
    
    fused_moe_kernel[grid](
        A, B, C, None, None,
        topk_weights, sorted_token_ids, expert_ids, num_tokens_post_padded,
        B.shape[1], B.shape[2], sorted_token_ids.shape[0], topk_ids.numel(),
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(2), B.stride(1),
        C.stride(1), C.stride(2),
        0, 0,
        MUL_ROUTED_WEIGHT=mul_routed_weight, top_k=top_k,
        compute_type=compute_type, use_fp8_w8a8=False, use_int8_w8a16=False,
        is_first_gemm=is_first_gemm,
        num_warps=num_warps, num_stages=num_stages,
        **kernel_kwargs,
    )

def fused_moe(hidden_states: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor,
              gating_output: torch.Tensor, topk_ids: torch.Tensor):
    # This standalone fused_moe handles the W1 (Gate+Up) and W2 (Down) projections.
    M, K = hidden_states.shape
    E, N, _ = w1.shape
    top_k = topk_ids.shape[1]
    
    config = get_best_config(M, E)

    # 1. Align Block Size
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(topk_ids, config['BLOCK_SIZE_M'], E)

    # 2. First GEMM: x @ W1
    intermediate_cache1 = torch.empty((M, topk_ids.shape[1], N), device=hidden_states.device, dtype=hidden_states.dtype)
    invoke_fused_moe_kernel(
        hidden_states, w1, intermediate_cache1,
        gating_output, topk_ids, sorted_token_ids, expert_ids, num_tokens_post_padded,
        mul_routed_weight=False, top_k=top_k, config=config
    )

    # 3. Activation: F.silu(gate) * up
    # W1 is usually [Gate, Up] concatenated.
    gate, up = intermediate_cache1.chunk(2, dim=-1)
    intermediate_cache2 = (torch.nn.functional.silu(gate) * up).view(M * topk_ids.shape[1], N // 2)

    # 4. Second GEMM: (activated) @ W2
    intermediate_cache3 = torch.empty((M, topk_ids.shape[1], w2.shape[1]), device=hidden_states.device, dtype=hidden_states.dtype)
    invoke_fused_moe_kernel(
        intermediate_cache2, w2, intermediate_cache3,
        gating_output, topk_ids, sorted_token_ids, expert_ids, num_tokens_post_padded,
        mul_routed_weight=True, top_k=top_k, config=config
    )

    return intermediate_cache3.sum(dim=1)