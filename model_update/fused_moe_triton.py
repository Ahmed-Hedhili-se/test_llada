import torch
import triton
import triton.language as tl
from typing import Any, Dict, Optional, Tuple

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
        compute_type: tl.constexpr, use_fp8_w8a8: tl.constexpr, use_int8_w8a16: tl.constexpr):
    
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
    a_ptrs = a_ptr + (offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak)

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
    num_tokens, top_k = topk_ids.shape
    flatten_ids = topk_ids.flatten()
    sorted_indices = torch.argsort(flatten_ids, stable=True)
    sorted_expert_ids = flatten_ids[sorted_indices]

    expert_counts = torch.bincount(sorted_expert_ids, minlength=num_experts)
    padded_expert_counts = ((expert_counts + block_size - 1) // block_size) * block_size

    # Python loop is fine here since num_experts (e.g. 64) is very small.
    padded_tokens = []
    padded_experts = []
    offset = 0
    for e in range(num_experts):
        count = expert_counts[e].item()
        padded_count = padded_expert_counts[e].item()
        
        if count > 0:
            padded_tokens.append(sorted_indices[offset : offset + count])
        if padded_count > count:
            padding = torch.full((padded_count - count,), num_tokens * top_k, dtype=sorted_indices.dtype, device=sorted_indices.device)
            padded_tokens.append(padding)
            
        if padded_count > 0:
            padded_experts.extend([e] * (padded_count // block_size))
        offset += count

    sorted_token_ids = torch.cat(padded_tokens) if padded_tokens else torch.empty(0, dtype=sorted_indices.dtype, device=sorted_indices.device)
    expert_ids = torch.tensor(padded_experts, dtype=torch.int32, device=sorted_indices.device)
    num_tokens_post_padded = torch.tensor([sorted_token_ids.size(0)], dtype=torch.int32, device=sorted_indices.device)

    return sorted_token_ids, expert_ids, num_tokens_post_padded

def invoke_fused_moe_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor,
                            topk_weights: torch.Tensor, topk_ids: torch.Tensor,
                            sorted_token_ids: torch.Tensor, expert_ids: torch.Tensor,
                            num_tokens_post_padded: torch.Tensor,
                            mul_routed_weight: bool, top_k: int, config: Dict[str, Any]) -> None:
    compute_type = tl.bfloat16 if A.dtype == torch.bfloat16 else tl.float16
    grid = lambda META: (triton.cdiv(sorted_token_ids.shape[0], META['BLOCK_SIZE_M']) * triton.cdiv(B.shape[1], META['BLOCK_SIZE_N']), )
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
        **config,
    )

def fused_moe(hidden_states: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor,
              gating_output: torch.Tensor, topk_ids: torch.Tensor):
    # This standalone fused_moe handles the W1 (Gate+Up) and W2 (Down) projections.
    M, K = hidden_states.shape
    E, N, _ = w1.shape
    top_k = topk_ids.shape[1]
    
    # We use a static config that works well for most shapes
    config = {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}
    if M <= E:
        config = {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1}

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
