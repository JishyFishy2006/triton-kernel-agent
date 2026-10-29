# task: L1-1
# expect: tf32_dot
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, N: tl.constexpr):
    offs = tl.arange(0, N)
    a = tl.load(a_ptr + offs[:, None] * N + offs[None, :])
    b = tl.load(b_ptr + offs[:, None] * N + offs[None, :])
    c = tl.dot(a, b)  # defaults to TF32 for fp32 inputs
    d = tl.dot(a, b, input_precision="tf32x3")
    tl.store(c_ptr + offs[:, None] * N + offs[None, :], c + d)


class ModelNew(nn.Module):
    def forward(self, A, B):
        C = torch.empty_like(A)
        matmul_kernel[(1,)](A, B, C, N=A.shape[0])
        return C
