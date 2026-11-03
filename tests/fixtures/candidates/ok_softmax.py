# task: L1-23
# expect: none
# expect-status: correct
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(x_ptr, out_ptr, n_cols, stride, BLOCK: tl.constexpr):
    # Three passes over a row of any length: max, sum of exp, then write.
    row = tl.program_id(0)
    x_row = x_ptr + row * stride
    out_row = out_ptr + row * stride
    row_max = -float("inf")
    for start in range(0, n_cols, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        x = tl.load(x_row + offs, mask=offs < n_cols, other=-float("inf"))
        row_max = tl.maximum(row_max, tl.max(x, axis=0))
    denom = 0.0
    for start in range(0, n_cols, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        x = tl.load(x_row + offs, mask=offs < n_cols, other=-float("inf"))
        denom += tl.sum(tl.exp(x - row_max), axis=0)
    for start in range(0, n_cols, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < n_cols
        x = tl.load(x_row + offs, mask=mask, other=-float("inf"))
        tl.store(out_row + offs, tl.exp(x - row_max) / denom, mask=mask)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.contiguous()
        out = torch.empty_like(x)
        rows, cols = x.shape
        softmax_kernel[(rows,)](x, out, cols, x.stride(0), BLOCK=1024)
        return out
