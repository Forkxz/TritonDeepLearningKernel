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
#         triton.Config({"BLOCK_SIZE_M": 4, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 4},num_warps=4,num_stages=2),
#     ],
#     key=["M", "N", "K"],
# )
@triton.jit
def dropconnect_dx_kernel(
    # Pointers to matrices
    dy_ptr,w_ptr,
    dx_ptr,
    seed,
    # Matrix dimensions
    M,K,N,
    # The stride variables represent how much to increase the ptr by when moving by 1
    stride_dym,stride_dyn,  #
    stride_wk,stride_wn,  #
    stride_dm,stride_dk,stride_dn,
    stride_xm,stride_xk,  #
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,BLOCK_SIZE_N: tl.constexpr,BLOCK_SIZE_K: tl.constexpr,  #
    ALLOWTF32: tl.constexpr,  #
):
    """ 
    dY_m = Y.grad
    dO_m = dY_m.view(M,1,N).broadcast_to(M,K,N)
    dx_m_cast = dO_m * WD_m
    dx_m = dx_m_cast.sum(dim=2) """
    # -----------------------------------------------------------
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    offset_m = pid_m * BLOCK_SIZE_M
    offset_n = 0
    offset_k = pid_k * BLOCK_SIZE_K
    # -----------------------------------------------------------
    dy_offsets = block_offsets_2d(M, N, stride_dym, stride_dyn, offset_m, offset_n, BLOCK_SIZE_M, BLOCK_SIZE_N)
    w_offsets = block_offsets_2d(K, N, stride_wk, stride_wn, offset_k, offset_n, BLOCK_SIZE_K, BLOCK_SIZE_N)
    d_offsets = block_offsets_3d(M, K, N, stride_dm, stride_dk, stride_dn, offset_m, offset_k, offset_n,BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N)
    
    dy_offsets = dy_offsets.reshape(BLOCK_SIZE_M,1, BLOCK_SIZE_N)
    w_offsets = w_offsets.reshape(1, BLOCK_SIZE_K, BLOCK_SIZE_N)
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    # ----------------------------------------------------------
    dy_tile = dy_ptr + dy_offsets
    w_tile = w_ptr + w_offsets
    # -----------------------------------------------------------
    ASM: tl.constexpr = "cvt.rna.tf32.f32 $0, $1;"
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
    for n in range(0, tl.cdiv(N, BLOCK_SIZE_N)):
        # If it is out of bounds, set it to 0.
        random_masks = tl.random.rand(seed, d_offsets) > 0.5
        n_mask = offs_n[None, None, :] < N - n * BLOCK_SIZE_N
        dy_load = tl.load(dy_tile, mask=n_mask, other=0.0)
        w_load = tl.load(w_tile, mask=n_mask, other=0.0)
        # if ALLOWTF32:
        #     dy_load = tl.inline_asm_elementwise(ASM, "=r, r", [dy_load], dtype=tl.float32, is_pure=True, pack=1)
        #     w_load = tl.inline_asm_elementwise(ASM, "=r, r", [w_load], dtype=tl.float32, is_pure=True, pack=1)
        dy = tl.where(random_masks, dy_load, 0.0)
        wd = tl.where(random_masks, w_load, 0.0)
        # We accumulate along the N dimension.
        mul = dy * wd
        accumulator += tl.sum(mul, axis=2)
        # Advance the ptrs to the next K block.
        dy_tile += BLOCK_SIZE_N * stride_dyn
        w_tile += BLOCK_SIZE_N * stride_wn
        d_offsets += BLOCK_SIZE_N * stride_dn

    # # -----------------------------------------------------------
    # # Write back the block of the output matrix C with masks.
    dx_offset, dx_mask = block_offsets_2d(M, K, stride_xm, stride_xk, offset_m, offset_k, BLOCK_SIZE_M, BLOCK_SIZE_K, True)
    dx_tile = dx_ptr + dx_offset
    dx = accumulator.to(dx_tile.dtype.element_ty)
    tl.store(dx_tile, dx, mask=dx_mask)

def triton_dropconnect_dx(dy, x, w, seed, tf32=False):
    """ dy = M,N ; x = M, K ; w = K, N """
    M, K = x.shape
    K, N = w.shape
    # Allocates output.
    dx = torch.empty((M, K), device=x.device, dtype=x.dtype)
    # 1D launch kernel where each block gets its own program.
    BLOCK_SIZE_M = 4
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 4
    # grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]),triton.cdiv(K, META["BLOCK_SIZE_K"]))
    grid = (triton.cdiv(M, BLOCK_SIZE_M),triton.cdiv(K, BLOCK_SIZE_K))
    dropconnect_dx_kernel[grid](
        dy,w,
        dx,  #
        seed,
        M,K,N,  #
        dy.stride(0),dy.stride(1),  #
        w.stride(0),w.stride(1),  #
        N * K, N, 1,  #
        dx.stride(0),dx.stride(1),  #
        BLOCK_SIZE_M=BLOCK_SIZE_M,BLOCK_SIZE_N=BLOCK_SIZE_N,BLOCK_SIZE_K=BLOCK_SIZE_K,  #
        ALLOWTF32=tf32 and (x.dtype is torch.float32),  #
    )
    return dx
    
# main
if __name__ == "__main__":
    # test matmul
    torch.backends.cuda.matmul.allow_tf32 = True
    M,K,N = 1024, 512, 512
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
    dx = triton_dropconnect_dx(dc, x, w, seed, tf32=torch.backends.cuda.matmul.allow_tf32)
    print(f"backward match: {torch.allclose(x.grad, dx, atol=1e-4)}")
    
    ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton_dropconnect_dx(dc, x, w, seed, tf32=torch.backends.cuda.matmul.allow_tf32), quantiles = [0.5, 0.2, 0.8])
    print(f"speed: {ms} ms, min: {min_ms} ms, max: {max_ms} ms")