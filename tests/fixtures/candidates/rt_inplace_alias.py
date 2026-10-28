# task: L1-19
# expect: none
# expect-runtime: output_aliases_input
# expect-correct: yes
# L4 #5: no output buffer at all: overwrite the input and hand it back.
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
    def forward(self, x):
        return relu(x, x)
