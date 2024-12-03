import os, sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# __file__获取执行文件相对路径，整行为取上一级的上一级目录
sys.path.append(BASE_DIR)

import torch
import triton
import triton.language as tl
from kernel.block_offsets import block_offsets_2d, block_offsets_3d
from kernel.dropconnect.random_matrix import triton_random_matrix
from kernel.dropconnect.foward import triton_dropconnect_fwd

# @triton.autotune(
#     configs=[
#         triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 4, "BLOCK_SIZE_K": 4},num_warps=4,num_stages=2),
#     ],
#     key=["M", "N", "K"],
# )
@triton.jit
def dropconnect_dx_kernel(
    # Pointers to matrices
    dy_ptr,x_ptr,
    dw_ptr,
    seed,
    # Matrix dimensions
    M,K,N,
    # The stride variables represent how much to increase the ptr by when moving by 1
    stride_dym,stride_dyn,  #
    stride_xm,stride_xk,  #
    stride_dm,stride_dk,stride_dn,
    stride_dwk,stride_dwn,  #
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,BLOCK_SIZE_N: tl.constexpr,BLOCK_SIZE_K: tl.constexpr,  #
    ALLOWTF32: tl.constexpr,  #
):
    """ 
    dY_m = Y.grad
    dO_m = dY_m.view(M,1,N).broadcast_to(M,K,N)
    dWD_m = dO_m * x_m_cast
    dw_m_cast = dWD_m * D
    dw_m = dw_m_cast.sum(dim=0) """
    # -----------------------------------------------------------
    pid_k = tl.program_id(0)
    pid_n = tl.program_id(1)
    offset_m = 0
    offset_k = pid_k * BLOCK_SIZE_K
    offset_n = pid_n * BLOCK_SIZE_N
    
    # -----------------------------------------------------------
    dy_offsets = block_offsets_2d(M, N, stride_dym, stride_dyn, offset_m, offset_n, BLOCK_SIZE_M, BLOCK_SIZE_N)
    x_offsets = block_offsets_2d(M, K, stride_xm, stride_xk, offset_m, offset_k, BLOCK_SIZE_M, BLOCK_SIZE_K)
    d_offsets = block_offsets_3d(M, K, N, stride_dm, stride_dk, stride_dn, offset_m, offset_k, offset_n,BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N)
    
    dy_offsets = dy_offsets.reshape(BLOCK_SIZE_M,1, BLOCK_SIZE_N)
    x_offsets = x_offsets.reshape(BLOCK_SIZE_M, BLOCK_SIZE_K,1)
    offs_m = tl.arange(0, BLOCK_SIZE_M)
    # ----------------------------------------------------------
    dy_tile = dy_ptr + dy_offsets
    x_tile = x_ptr + x_offsets
    # -----------------------------------------------------------
    ASM: tl.constexpr = "cvt.rna.tf32.f32 $0, $1;"
    accumulator = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_N), dtype=tl.float32)
    for m in range(0, tl.cdiv(M, BLOCK_SIZE_M)):
        # If it is out of bounds, set it to 0.
        random_masks = tl.random.rand(seed, d_offsets) > 0.5
        m_mask = offs_m[:, None, None] < M - m * BLOCK_SIZE_M
        dy_load = tl.load(dy_tile, mask=m_mask, other=0.0)
        x_load = tl.load(x_tile, mask=m_mask, other=0.0)
        # if ALLOWTF32:
        #     dy_load = tl.inline_asm_elementwise(ASM, "=r, r", [dy_load], dtype=tl.float32, is_pure=True, pack=1)
        #     x_load = tl.inline_asm_elementwise(ASM, "=r, r", [x_load], dtype=tl.float32, is_pure=True, pack=1)
        dy = tl.where(random_masks, dy_load, 0.0)
        wd = tl.where(random_masks, x_load, 0.0)
        # We accumulate along the N dimension.
        mul = dy * wd
        accumulator += tl.sum(mul, axis=0)
        # Advance the ptrs to the next K block.
        dy_tile += BLOCK_SIZE_M * stride_dym
        x_tile += BLOCK_SIZE_M * stride_xm
        d_offsets += BLOCK_SIZE_M * stride_dm

    # # -----------------------------------------------------------
    # # Write back the block of the output matrix C with masks.
    dw_offset, dw_mask = block_offsets_2d(K, N, stride_dwk, stride_dwn, offset_k, offset_n, BLOCK_SIZE_K, BLOCK_SIZE_N, True)
    dw_tile = dw_ptr + dw_offset
    dw = accumulator.to(dw_tile.dtype.element_ty)
    tl.store(dw_tile, dw, mask=dw_mask)

def triton_dropconnect_dw(dy, x, w, seed, tf32=False):
    """ dy = M,N ; x = M, K ; w = K, N """
    M, K = x.shape
    K, N = w.shape
    # Allocates output.
    dw = torch.empty((K,N), device=w.device, dtype=w.dtype)
    # 1D launch kernel where each block gets its own program.
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 4
    BLOCK_SIZE_K = 4
    # grid = lambda META: (triton.cdiv(K, META["BLOCK_SIZE_K"]),triton.cdiv(N, META["BLOCK_SIZE_N"]))
    grid = (triton.cdiv(K, BLOCK_SIZE_K),triton.cdiv(N, BLOCK_SIZE_N))
    dropconnect_dx_kernel[grid](
        dy,x,
        dw,  #
        seed,
        M,K,N,  #
        dy.stride(0),dy.stride(1),  #
        x.stride(0),x.stride(1),  #
        N * K, N, 1,  #
        dw.stride(0),dw.stride(1),  #
        BLOCK_SIZE_M=BLOCK_SIZE_M,BLOCK_SIZE_N=BLOCK_SIZE_N,BLOCK_SIZE_K=BLOCK_SIZE_K,  #
        ALLOWTF32=tf32 and (x.dtype is torch.float32),  #
    )
    return dw
    
# main
if __name__ == "__main__":
    # test matmul
    torch.backends.cuda.matmul.allow_tf32 = True
    M,K,N = 1024, 384, 768
    seed = 42
    x = torch.randn(M, K).cuda().to(torch.float32).requires_grad_()
    w = torch.randn(K, N).cuda().to(torch.float32).requires_grad_()
    # check with torch
    mask = triton_random_matrix(M,K,N,seed)
    w_cast = w.view(1,K,N).broadcast_to(M,K,N)
    wd = w_cast * mask
    x_cast = x.view(M,K,1).broadcast_to(M,K,N)
    mul = x_cast * wd
    c = mul.sum(dim=1)
    c.retain_grad()
    
    # for backward
    another_x = torch.rand(M,N).cuda().to(torch.float32)
    Y = c*another_x
    Y.sum().backward(retain_graph=True)

    # check with triton
    x_t = x.detach().clone()
    w_t = w.detach().clone()
    c_t = triton_dropconnect_fwd(x_t, w_t, seed, tf32=torch.backends.cuda.matmul.allow_tf32)
    print(f"forward match: {torch.allclose(c, c_t, atol=1e-4)}")
    dc = c.grad
    dw = triton_dropconnect_dw(dc, x, w, seed, tf32=torch.backends.cuda.matmul.allow_tf32)
    print(f"backward match: {torch.allclose(w.grad, dw, atol=1e-4)}")
    
    ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton_dropconnect_dw(dc, x, w, seed, tf32=torch.backends.cuda.matmul.allow_tf32), quantiles = [0.5, 0.2, 0.8])
    print(f"speed: {ms} ms, min: {min_ms} ms, max: {max_ms} ms")