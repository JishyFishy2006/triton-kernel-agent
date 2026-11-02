# task: L1-19
# expect: none
# B11: writes far outside its output buffer (no mask): an illegal memory access that breaks this
# process's CUDA context. The next eval must still run.
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def bad_kernel(out_ptr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(out_ptr + offs * 1_000_000_000, 1.0)


class ModelNew(nn.Module):
    def forward(self, x):
        out = torch.empty_like(x)
        bad_kernel[(1024,)](out, BLOCK=1024)
        return out
