import os, sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# __file__获取执行文件相对路径，整行为取上一级的上一级目录
sys.path.append(BASE_DIR)

import torch
import triton
import triton.language as tl
import numpy as np
from kernel.block_offsets import block_offsets_2d, block_offsets_3d
from kernel.dropconnect.random_matrix import triton_random_matrix

# @triton.autotune(
#     configs=[
#         triton.Config({"BLOCK_SIZE_M": 4, "BLOCK_SIZE_N": 4, "BLOCK_SIZE_K": 32},num_warps=4,num_stages=2),
#     ],
#     key=["M", "N", "K"],
# )
@triton.jit
def dropconnect_fwd_kernel(
    # Pointers to matrices
    x_ptr,w_ptr,y_ptr,
    seed,
    # Matrix dimensions
    M,K,N,
    # The stride variables represent how much to increase the ptr by when moving by 1
    stride_xm,stride_xk,  #
    stride_wk,stride_wn,  #
    stride_ym,stride_yn,
    stride_dm,stride_dk,stride_dn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,BLOCK_SIZE_N: tl.constexpr,BLOCK_SIZE_K: tl.constexpr,  #
    ALLOWTF32: tl.constexpr,  #
):
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offset_m = pid_m * BLOCK_SIZE_M
    offset_n = pid_n * BLOCK_SIZE_N
    offset_k = 0
    # -----------------------------------------------------------
    x_offsets = block_offsets_2d(M, K, stride_xm, stride_xk, offset_m, offset_k, BLOCK_SIZE_M, BLOCK_SIZE_K)
    w_offsets = block_offsets_2d(K, N, stride_wk, stride_wn, offset_k, offset_n, BLOCK_SIZE_K, BLOCK_SIZE_N)
    d_offsets = block_offsets_3d(M, K, N, stride_dm, stride_dk, stride_dn, offset_m, offset_k, offset_n,BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N)
    x_offsets = x_offsets.reshape(BLOCK_SIZE_M, BLOCK_SIZE_K,1)
    w_offsets = w_offsets.reshape(1, BLOCK_SIZE_K, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    x_tile = x_ptr + x_offsets
    w_tile = w_ptr + w_offsets
    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    ASM: tl.constexpr = "cvt.rna.tf32.f32 $0, $1;"
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        random_masks = tl.random.rand(seed, d_offsets) > 0.5 # set 0 to debug
        k_mask = offs_k[None, :, None] < K - k * BLOCK_SIZE_K
        x_load = tl.load(x_tile, mask=k_mask, other=0.0)
        w_load = tl.load(w_tile, mask=k_mask, other=0.0)
        # if ALLOWTF32:
        #     x_load = tl.inline_asm_elementwise(ASM, "=r, r", [x_load], dtype=tl.float32, is_pure=True, pack=1)
        #     w_load = tl.inline_asm_elementwise(ASM, "=r, r", [w_load], dtype=tl.float32, is_pure=True, pack=1)
        a = tl.where(random_masks, x_load, 0.0)
        b = tl.where(random_masks, w_load, 0.0)
        # We accumulate along the K dimension.
        mul = a * b
        accumulator += tl.sum(mul, axis=1)
        # Advance the ptrs to the next K block.
        x_tile += BLOCK_SIZE_K * stride_xk
        w_tile += BLOCK_SIZE_K * stride_wk
        d_offsets += BLOCK_SIZE_K * stride_dk

    # # -----------------------------------------------------------
    # # Write back the block of the output matrix C with masks.
    y_offset, y_mask = block_offsets_2d(M, N, stride_ym, stride_yn, offset_m, offset_n, BLOCK_SIZE_M, BLOCK_SIZE_N, True)
    y_tile = y_ptr + y_offset
    y = accumulator.to(y_tile.dtype.element_ty)
    tl.store(y_tile, y, mask=y_mask)


def triton_dropconnect_fwd(x, w, seed, tf32=False):
    # Check constraints.
    assert x.shape[1] == w.shape[0], "Incompatible dimensions"
    # assert x.is_contiguous(), "Matrix x must be contiguous"
    # assert w.is_contiguous(), "Matrix w must be contiguous"
    M, K = x.shape
    K, N = w.shape
    # Allocates output.
    c = torch.empty((M, N), device=x.device, dtype=x.dtype)
    # 1D launch kernel where each block gets its own program.
    BLOCK_SIZE_M = 4
    BLOCK_SIZE_N = 4
    BLOCK_SIZE_K = 32
    # grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]),triton.cdiv(N, META["BLOCK_SIZE_N"]))
    grid = (triton.cdiv(M, BLOCK_SIZE_M),triton.cdiv(N, BLOCK_SIZE_N))
    dropconnect_fwd_kernel[grid](
        x,w,c,  #
        seed,
        M,K,N,  #
        x.stride(0),x.stride(1),  #
        w.stride(0),w.stride(1),  #
        c.stride(0),c.stride(1),  #
        N * K,N,1,  #
        BLOCK_SIZE_M=BLOCK_SIZE_M,BLOCK_SIZE_N=BLOCK_SIZE_N,BLOCK_SIZE_K=BLOCK_SIZE_K,  #
        ALLOWTF32=tf32 and (x.dtype is torch.float32),  #
    )

    return c
    
# main
if __name__ == "__main__":
    # test matmul
    torch.backends.cuda.matmul.allow_tf32 = True
    M,K,N = 1024, 384, 768
    seed = 42
    dtype = torch.float32
    x = torch.randn(M, K).cuda().to(dtype)
    w = torch.randn(N, K).cuda().to(dtype)
    c = triton_dropconnect_fwd(x, w.T, seed, tf32=torch.backends.cuda.matmul.allow_tf32)
    
    # check with torch
    mask = triton_random_matrix(M,K,N,seed)
    w_yast = w.T.view(1,K,N).broadcast_to(M,K,N)
    wd = w_yast * mask
    x_yast = x.view(M,K,1).broadcast_to(M,K,N)
    mul = x_yast * wd
    ref = mul.sum(1)
    print(f"Matches PyTorch: {torch.allclose(c, ref, atol=1e-3)}")

    ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton_dropconnect_fwd(x, w.T, seed, tf32=torch.backends.cuda.matmul.allow_tf32), quantiles = [0.5, 0.2, 0.8])
    print(f"speed: {ms} ms, min: {min_ms} ms, max: {max_ms} ms")