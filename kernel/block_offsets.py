import triton
import triton.language as tl

@triton.jit
def block_offsets_2d(shape_x, shape_y, stride_x, stride_y, offset_x,offset_y,block_shape_x, block_shape_y, require_mask=False):
    offs_x = tl.arange(0, block_shape_x) + offset_x
    offs_y = tl.arange(0, block_shape_y) + offset_y
    ptrs = offs_x[:, None] * stride_x + offs_y[None, :] * stride_y
    if require_mask:
        mask = (offs_x[:, None] < shape_x) & (offs_y[None, :] < shape_y)
        return ptrs, mask
    else:
        return ptrs

@triton.jit
def block_offsets_3d(shape_x, shape_y, shape_z, stride_x, stride_y, stride_z, offset_x,offset_y,offset_z,block_shape_x, block_shape_y, block_shape_z, require_mask=False
):
    offs_x = tl.arange(0, block_shape_x) + offset_x
    offs_y = tl.arange(0, block_shape_y) + offset_y
    offs_z = tl.arange(0, block_shape_z) + offset_z
    ptrs = (
        offs_x[:, None, None] * stride_x + offs_y[None, :, None] * stride_y + offs_z[None, None, :] * stride_z
    )
    if require_mask:
        mask = (offs_x[:, None, None] < shape_x) & (offs_y[None, :, None] < shape_y) & (offs_z[None, None, :] < shape_z)
        return ptrs, mask
    else:
        return ptrs