import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _gelu_kernel(x_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, eviction_policy='evict_first')
    x2 = x * x
    x3 = x2 * x
    z = 0.7978845608028654 * (x + 0.044715 * x3)
    az = tl.abs(z)
    e = tl.exp(-2.0 * az)
    t_abs = (1.0 - e) / (1.0 + e)
    t = tl.where(z < 0.0, -t_abs, t_abs)
    y = 0.5 * x * (1.0 + t)
    tl.store(out_ptr + offs, y, mask=mask, eviction_policy='evict_first')


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x):
        xf = x.contiguous()
        out = torch.empty(xf.shape, device=xf.device, dtype=xf.dtype)
        n = 1
        for d in xf.shape:
            n *= d
        if n == 0:
            return out
        BLOCK = 2048
        grid = ((n + BLOCK - 1) // BLOCK,)
        _gelu_kernel[grid](xf, out, n, BLOCK=BLOCK, num_warps=8, num_stages=2)
        return out
