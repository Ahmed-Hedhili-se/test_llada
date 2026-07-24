import triton 
import torch 
import triton.language as tl 

def moe_align_block_size (topk_ids , block_size , num_experts) :
    pass 


@triton.jit
def fused_moe_kernel(a_ptr , b_ptr , c_ptr ,sorted_token_ids_ptr, num_valid_tokens,
                     M,N,K,
                     stride_am,stride_ak ,stride_bk,stride_bn, stride_cm , stride_cn,
                     BLOCK_SIZE_M :tl.constexpr , BLOCK_SIZE_N: tl.constexpr , BLOCK_SIZE_K: tl.constexpr,GROUP_SIZE_M: tl.constexpr,) :
    pid  = tl.program_id(axis = 0)
    
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n=tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M* num_pid_n
    group_id = pid// num_pid_in_group
    first_pid_m = group_id*GROUP_SIZE_M
    group_size_m = min( num_pid_m - first_pid_m  , GROUP_SIZE_M )

    pid_m = first_pid_m + ((pid%num_pid_in_group)% group_size_m)
    pid_n = (pid%num_pid_in_group) // group_size_m

    
    offs_token_id = (pid_m*BLOCK_SIZE_M + tl.arange(0,BLOCK_SIZE_M))
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    token_mask = offs_token < offs_token_id
    offs_n = pid_n * BLOCK_SIZE_N +tl.arrange(0 , BLOCK_SIZE_N)
    offs_k =  tl.arrange(0 , BLOCK_SIZE_K)

    a_ptrs = (a_ptr+ offs_token[:, None] *stride_am + offs_k[None, :]*stride_ak )
    b_ptrs = (b_ptr+ offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn )



    accumulator = tl.zeros( (BLOCK_SIZE_M , BLOCK_SIZE_N) , dtype= tl.float32  )
    for k in range (0,tl.cdiv(K, BLOCK_SIZE_K)) : 
        a= tl.load(a_ptrs , mask =(offs_token[:,None]<M) & (offs_k[None ,:] <K-K*BLOCK_SIZE_K) , other = 0.0)
        b= tl.load(b_ptrs , mask= (offs_k[: , None]<K - K*BLOCK_SIZE_K ) &(offs_n[None , : ]<N) , other = 0.0 )

        accumulator +=tl.dot(a,b)
        #move tothe bext block
        a_ptrs += BLOCK_SIZE_K*stride_ak
        b_ptrs += BLOCK_SIZE_K*stride_bk

    accumulator = accumulator.to(tl.float16)
    c_ptrs = c_ptr + offs_token[:,None]*stride_cm + offs_n[None , : ] * stride_cn
    #output mask 
    c_mask = (offs_token[:,None]<M) & (offs_n[None , : ]<N)
    
    tl.store(c_ptrs , accumulator , mask = c_mask)


    return 

def invoke_fused_moe_kernel(): 
    pass 


def fused_moe(): 
    pass 