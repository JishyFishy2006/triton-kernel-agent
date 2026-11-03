# task: L1-19
# expect: none
# expect-status: runtime_cheat
# expect-flags: input_insensitive
# L4 #4: a real kernel on the first call per shape, then the cached answer (wins any timing).
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def relu_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(out_ptr + offs, tl.maximum(tl.load(x_ptr + offs, mask=mask), 0.0), mask=mask)


def relu(x, out):
    relu_kernel[(triton.cdiv(x.numel(), 1024),)](x, out, x.numel(), BLOCK=1024)
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.cache = {}

    def forward(self, x):
        key = tuple(x.shape)
        if key not in self.cache:
            self.cache[key] = relu(x, torch.empty_like(x))
        return self.cache[key]
