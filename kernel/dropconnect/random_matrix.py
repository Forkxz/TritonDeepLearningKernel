import os, sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# __file__获取执行文件相对路径，整行为取上一级的上一级目录
sys.path.append(BASE_DIR)

import torch
import triton
import triton.language as tl
import numpy as np
from kernel.block_offsets import block_offsets_2d, block_offsets_3d

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 4, "BLOCK_SIZE_K": 32}),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def random_matrix_kernel(
    r_ptr,
    seed,
    # Matrix dimensions
    M,K,N,
    # The stride variables represent how much to increase the ptr by when moving by 1
    stride_dm,stride_dk,stride_dn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,BLOCK_SIZE_N: tl.constexpr,BLOCK_SIZE_K: tl.constexpr,  #
):
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    offset_m = pid_m * BLOCK_SIZE_M
    offset_n = pid_n * BLOCK_SIZE_N
    offset_k = 0
    # -----------------------------------------------------------
    d_offsets = block_offsets_3d(M, K, N, stride_dm, stride_dk, stride_dn, offset_m, offset_k, offset_n,BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_mask = offs_k[None, :, None] < K - k * BLOCK_SIZE_K
        random_masks = tl.random.rand(seed, d_offsets) > 0.5
        # store the random mask to r_tile
        tl.store(r_ptr + d_offsets, random_masks.to(tl.int8), mask=k_mask)
        d_offsets += BLOCK_SIZE_K*stride_dk

def triton_random_matrix(M,K,N,seed):
    # Allocates output.
    r = torch.empty((M, K, N), device="cuda", dtype=torch.int8)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),)
    random_matrix_kernel[grid](
        r,  #
        seed,
        M,K,N,  #
        N * K, N, 1,  #
    )
    return r
    
# main
if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    M,K,N = 64, 256, 512
    seed = 42
    r = triton_random_matrix(M,K,N,seed)
    print(r.shape)
    print(r.sum()/r.numel())