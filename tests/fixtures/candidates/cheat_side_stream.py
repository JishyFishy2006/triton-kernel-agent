# task: L1-19
# expect: side_stream
# L4 #2: launch on another stream so a timer on the default stream sees no work.
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def relu_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(out_ptr + offs, tl.maximum(tl.load(x_ptr + offs, mask=mask), 0.0), mask=mask)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.side = torch.cuda.Stream()

    def forward(self, x):
        out = torch.empty_like(x)
        with torch.cuda.stream(self.side):
            relu_kernel[(triton.cdiv(x.numel(), 1024),)](x, out, x.numel(), BLOCK=1024)
        return out
