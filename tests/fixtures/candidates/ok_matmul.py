# task: L1-1
# expect: none
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_offs = k * BLOCK_K + offs_k
        a = tl.load(
            a_ptr + offs_m[:, None] * stride_am + k_offs[None, :] * stride_ak,
            mask=(offs_m[:, None] < M) & (k_offs[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            b_ptr + k_offs[:, None] * stride_bk + offs_n[None, :] * stride_bn,
            mask=(k_offs[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )
        acc = tl.dot(a, b, acc, input_precision="ieee")
    c = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def matmul(a, b):
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = (triton.cdiv(M, 64), triton.cdiv(N, 64))
    matmul_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_M=64,
        BLOCK_N=64,
        BLOCK_K=32,
    )
    return c


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        return matmul(A, B)
