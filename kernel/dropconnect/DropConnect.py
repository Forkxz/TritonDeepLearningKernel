import os, sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# __file__获取执行文件相对路径，整行为取上一级的上一级目录
sys.path.append(BASE_DIR)

import torch
import triton
import numpy as np
from kernel.dropconnect.random_matrix import triton_random_matrix
from kernel.dropconnect.foward import triton_dropconnect_fwd
from kernel.dropconnect.dx import triton_dropconnect_dx
from kernel.dropconnect.dw import triton_dropconnect_dw

class _triton_dropconnect_(torch.autograd.Function):
    @classmethod
    def forward(self, ctx, x, w, tf32):
        # seed = np.random.random()
        ctx.seed = 42 # fixed seed for debugging
        ctx.tf32 = tf32
        o = triton_dropconnect_fwd(x, w, ctx.seed, tf32)
        if x.requires_grad:
            ctx.save_for_backward(x, w)
        return o

    @classmethod
    def backward(self, ctx, dy):
        x, w = ctx.saved_tensors
        dx = triton_dropconnect_dx(dy, x, w, ctx.seed, ctx.tf32)
        dw = triton_dropconnect_dw(dy, x, w, ctx.seed, ctx.tf32)
        return dx, dw, None

def triton_dropconnect(x, w, bias=None, tf32=False):
    x_shape = x.size()
    w_shape = w.size()
    assert x_shape[-1] == w_shape[0], "x and w shape mismatch"
    x = x.view(-1, x_shape[-1])
    out = _triton_dropconnect_.apply(x, w, tf32)
    if bias is not None:
        out += bias.view(1, -1)
    return out.view(*x_shape[:-1], w_shape[1])

class TritonDropconnect(torch.nn.Linear):
    """ Applies a dropconnect linear transformation to the incoming data
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``
        tf32: If set to ``True``, the layer will use tf32 for matmul. This flag is only effective when the input precision is float32.
    """
    def __init__(self, in_features, out_features, bias=True, tf32=True):
        super(TritonDropconnect, self).__init__(in_features, out_features, bias)
        self.tf32 = tf32

    def forward(self, x):
        return triton_dropconnect(x, self.weight.T, self.bias, self.tf32)

# main
if __name__ == "__main__":
    # test matmul
    torch.backends.cuda.matmul.allow_tf32 = True
    M,K,N = 64, 256, 512
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
    another_x = torch.rand(M,N).cuda().to(torch.float32)
    Y = c*another_x
    Y.sum().backward()

    # check with triton
    x_t = x.detach().clone().requires_grad_()
    w_t = w.detach().clone().requires_grad_()
    c_t = triton_dropconnect(x_t, w_t, tf32=torch.backends.cuda.matmul.allow_tf32)
    Y_t = c_t*another_x
    Y_t.sum().backward()
    print(f"forward match: {torch.allclose(c, c_t, atol=1e-4)}")
    print(f"dx match: {torch.allclose(x.grad, x_t.grad, atol=1e-4)}")
    print(f"dw match: {torch.allclose(w.grad, w_t.grad, atol=1e-4)}")