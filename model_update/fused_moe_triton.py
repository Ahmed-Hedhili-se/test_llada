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

def _shared_mem_limit() -> int:
    if not torch.cuda.is_available():
        return 96_000
    try:
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        limit = getattr(props, "shared_memory_per_block_optin", None)
        if limit is None:
            limit = getattr(props, "shared_memory_per_block", None)
        if limit:
            return int(limit) - 1024
    except Exception:
        pass
    return 96_000

SHARED_MEM_LIMIT = _shared_mem_limit()

# Fuse the SiLU(gate)*up activation into GEMM1's epilogue instead of running it
# as a separate elementwise pass over a materialized [M, top_k, 2*EI] tensor.
# Set LLADA_MOE_FUSED_SILU=0 to fall back to the unfused path (the A/B switch
# eval/test_fused_silu_epilogue.py uses, and the escape hatch if a future
# Triton version's tl.exp stops matching CUDA's expf -- see _silu_epilogue
# notes in fused_moe_kernel).
FUSE_SILU = os.environ.get("LLADA_MOE_FUSED_SILU", "1") != "0"


def _shmem_bytes(bm: int, bn: int, bk: int, ns: int, silu_epilogue: bool) -> int:
    """Per-block shared memory for one pipelined stage set.

    The SiLU epilogue keeps TWO B tiles in flight per K step (the gate columns
    at offs_bn and the up columns at offs_bn + N/2), so its B-side footprint
    doubles. get_best_config picks ONE config used by both GEMMs, so the guard
    has to size for the more expensive of the two.
    """
    b_tiles = 2 if silu_epilogue else 1
    return (bm * bk + b_tiles * bk * bn) * ns * 2


def get_best_config(M: int, E: int, silu_epilogue: bool = False) -> Dict[str, Any]:
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
        shmem = _shmem_bytes(bm, bn, bk, ns, silu_epilogue)

        if shmem <= SHARED_MEM_LIMIT:
            config = candidate
        else:
            # Try reducing num_stages to 2
            candidate['num_stages'] = 2
            shmem_reduced = _shmem_bytes(bm, bn, bk, 2, silu_epilogue)
            if shmem_reduced <= SHARED_MEM_LIMIT:
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
        is_first_gemm: tl.constexpr, SILU_EPILOGUE: tl.constexpr):
    """
    SILU_EPILOGUE (GEMM1 only): w1 packs [gate ; up] along N, so the unfused
    path materialized the full [M, top_k, 2*EI] product, read it back, and ran
    SiLU(gate)*up as a separate elementwise pass. Here `N` is the OUTPUT width
    (EI, i.e. half of w1's N) and each program computes two B tiles -- the gate
    columns at offs_bn and the up columns at offs_bn + N -- then applies the
    activation from registers, writing EI-wide output directly. Removes a
    ~15MB write + read + write per layer per forward at B=57, on a kernel that
    ncu measures at 96% L2 throughput.

    Bit-exactness: the K-loop is untouched, so the accumulators are identical
    to the unfused path's. The epilogue then reproduces the unfused op order
    exactly rather than taking the more accurate route -- accumulators are
    rounded to `compute_type` FIRST (reproducing the intermediate_cache1
    store), and SiLU is evaluated as x/(1+exp(-x)) in fp32 with the result
    rounded back to compute_type before the multiply, matching what ATen's
    bf16 SiLU (opmath_t=float) and bf16 mul do elementwise. Computing the
    activation straight from the fp32 accumulators would be MORE precise and
    therefore NOT bit-identical, which is not what we want here.

    The one thing this cannot guarantee by construction is that Triton's
    tl.exp lowers to the same instruction as CUDA's expf. A last-ulp fp32
    difference there would almost always vanish in the round to bf16, but
    "almost always" is not "never" -- eval/test_fused_silu_epilogue.py
    measures it directly rather than assuming, and LLADA_MOE_FUSED_SILU=0
    disables the fusion if it ever stops matching.
    """

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

    if SILU_EPILOGUE:
        # N is the output (gate) width here, so the matching `up` columns of
        # w1 sit exactly N further along the same expert's weight block.
        b_up_ptrs = b_ptr + off_experts * stride_be + (
            offs_k[:, None] * stride_bk + (offs_bn + N)[None, :] * stride_bn)
        accumulator_up = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K), other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator += tl.dot(a, b)
        if SILU_EPILOGUE:
            b_up = tl.load(b_up_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
            accumulator_up += tl.dot(a, b_up)
            b_up_ptrs += BLOCK_SIZE_K * stride_bk
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if SILU_EPILOGUE:
        # Round to compute_type FIRST -- this is the intermediate_cache1 store
        # the unfused path performed. Skipping it would be more accurate and
        # therefore not bit-identical.
        gate = accumulator.to(compute_type).to(tl.float32)
        up = accumulator_up.to(compute_type).to(tl.float32)
        # ATen's bf16 SiLU computes x/(1+exp(-x)) with opmath_t=float, then
        # rounds to bf16; the following mul then rounds again. Same order here.
        silu = gate / (1.0 + tl.exp(-gate))
        accumulator = silu.to(compute_type).to(tl.float32) * up

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

    Vectorized: the previous implementation looped over `range(num_experts)`
    in Python and called `.item()` twice per expert (2 x num_experts
    host-device syncs per call -- 128 for num_experts=64). Each `.item()`
    blocks the CPU until the GPU catches up, serializing what should be
    back-to-back kernel launches; with 16 MoE layers x dozens of forward
    passes per generation call, that loop alone issued tens of thousands of
    stalls. This version computes every expert's destination offsets with
    cumulative sums and writes them in a single scatter, entirely on GPU,
    syncing only once at the very end -- an unavoidable `.item()` to size
    the output tensor, since the total padded length is genuinely
    data-dependent -- instead of once per expert.

    Output shapes/dtypes/values are identical to the original for any
    input (see eval/test_moe_align_block_size.py, which checks this
    directly against a frozen copy of the original loop).
    """
    num_tokens, top_k = topk_ids.shape
    device = topk_ids.device
    num_valid_tokens = num_tokens * top_k

    flatten_ids = topk_ids.flatten()
    sorted_indices = torch.argsort(flatten_ids, stable=True)
    sorted_expert_ids = flatten_ids[sorted_indices]

    expert_counts = torch.bincount(sorted_expert_ids, minlength=num_experts)
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

    total_padded_len = int(padded_expert_counts.sum().item())  # one sync, not 128

    sorted_token_ids = torch.full(
        (total_padded_len,), num_valid_tokens, dtype=sorted_indices.dtype, device=device
    )
    sorted_token_ids.scatter_(0, dest, sorted_indices)

    num_blocks_per_expert = padded_expert_counts // block_size
    expert_ids = torch.repeat_interleave(
        torch.arange(num_experts, device=device, dtype=torch.int32),
        num_blocks_per_expert,
    )

    num_tokens_post_padded = torch.tensor([total_padded_len], dtype=torch.int32, device=device)

    return sorted_token_ids, expert_ids, num_tokens_post_padded

def invoke_fused_moe_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor,
                            topk_weights: torch.Tensor, topk_ids: torch.Tensor,
                            sorted_token_ids: torch.Tensor, expert_ids: torch.Tensor,
                            num_tokens_post_padded: torch.Tensor,
                            mul_routed_weight: bool, top_k: int, config: Dict[str, Any],
                            silu_epilogue: bool = False) -> None:
    compute_type = tl.bfloat16 if A.dtype == torch.bfloat16 else tl.float16
    is_first_gemm = not mul_routed_weight

    # With the SiLU epilogue, each program consumes two of w1's N tiles and
    # emits one, so the output is half as wide and the N grid halves with it.
    n_out = B.shape[1] // 2 if silu_epilogue else B.shape[1]

    # C is [M, top_k, N] on the unfused path but the already-flattened
    # [M*top_k, N] that GEMM2 reads when the epilogue is on; offs_token indexes
    # rows either way, so take the last two strides rather than fixed axes.
    stride_cm, stride_cn = C.stride(-2), C.stride(-1)

    grid = lambda META: (triton.cdiv(sorted_token_ids.shape[0], META['BLOCK_SIZE_M'])
                         * triton.cdiv(n_out, META['BLOCK_SIZE_N']), )

    kernel_kwargs = config.copy()
    num_warps = kernel_kwargs.pop('num_warps', 4)
    num_stages = kernel_kwargs.pop('num_stages', 2)

    fused_moe_kernel[grid](
        A, B, C, None, None,
        topk_weights, sorted_token_ids, expert_ids, num_tokens_post_padded,
        n_out, B.shape[2], sorted_token_ids.shape[0], topk_ids.numel(),
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(2), B.stride(1),
        stride_cm, stride_cn,
        0, 0,
        MUL_ROUTED_WEIGHT=mul_routed_weight, top_k=top_k,
        compute_type=compute_type, use_fp8_w8a8=False, use_int8_w8a16=False,
        is_first_gemm=is_first_gemm, SILU_EPILOGUE=silu_epilogue,
        num_warps=num_warps, num_stages=num_stages,
        **kernel_kwargs,
    )

def fused_moe(hidden_states: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor,
              gating_output: torch.Tensor, topk_ids: torch.Tensor,
              fuse_silu: Optional[bool] = None):
    """fuse_silu: None follows the FUSE_SILU module default (env-overridable);
    True/False force it, which is how eval/test_fused_silu_epilogue.py A/Bs the
    two paths against each other in one process."""
    # This standalone fused_moe handles the W1 (Gate+Up) and W2 (Down) projections.
    M, K = hidden_states.shape
    E, N, _ = w1.shape
    top_k = topk_ids.shape[1]

    silu_epilogue = FUSE_SILU if fuse_silu is None else fuse_silu
    config = get_best_config(M, E, silu_epilogue=silu_epilogue)

    # 1. Align Block Size
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(topk_ids, config['BLOCK_SIZE_M'], E)

    # 2. First GEMM: x @ W1, with SiLU(gate)*up folded into the epilogue when
    #    enabled -- writing the EI-wide activated result straight out instead
    #    of a 2*EI-wide intermediate that then has to be re-read.
    if silu_epilogue:
        intermediate_cache2 = torch.empty((M * top_k, N // 2), device=hidden_states.device, dtype=hidden_states.dtype)
        invoke_fused_moe_kernel(
            hidden_states, w1, intermediate_cache2,
            gating_output, topk_ids, sorted_token_ids, expert_ids, num_tokens_post_padded,
            mul_routed_weight=False, top_k=top_k, config=config, silu_epilogue=True
        )
    else:
        intermediate_cache1 = torch.empty((M, top_k, N), device=hidden_states.device, dtype=hidden_states.dtype)
        invoke_fused_moe_kernel(
            hidden_states, w1, intermediate_cache1,
            gating_output, topk_ids, sorted_token_ids, expert_ids, num_tokens_post_padded,
            mul_routed_weight=False, top_k=top_k, config=config
        )

        # 3. Activation: F.silu(gate) * up
        # W1 is usually [Gate, Up] concatenated.
        gate, up = intermediate_cache1.chunk(2, dim=-1)
        intermediate_cache2 = (torch.nn.functional.silu(gate) * up).view(M * top_k, N // 2)

    # 4. Second GEMM: (activated) @ W2
    intermediate_cache3 = torch.empty((M, top_k, w2.shape[1]), device=hidden_states.device, dtype=hidden_states.dtype)
    invoke_fused_moe_kernel(
        intermediate_cache2, w2, intermediate_cache3,
        gating_output, topk_ids, sorted_token_ids, expert_ids, num_tokens_post_padded,
        mul_routed_weight=True, top_k=top_k, config=config
    )

    return intermediate_cache3.sum(dim=1)