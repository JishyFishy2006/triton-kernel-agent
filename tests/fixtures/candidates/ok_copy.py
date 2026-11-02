# task: L1-19
# expect: none
# A plain copy kernel, for B10's acceptance test (not a solution to L1-19).
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def copy_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask), mask=mask)


class ModelNew(nn.Module):
    def forward(self, x):
        out = torch.empty_like(x)
        copy_kernel[(triton.cdiv(x.numel(), 4096),)](x, out, x.numel(), BLOCK=4096)
        return out
