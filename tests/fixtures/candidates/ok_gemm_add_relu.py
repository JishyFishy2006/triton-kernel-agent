# task: L2-76
# expect: none
# expect-runtime: none
# expect-correct: yes
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def gemm_add_relu_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    out_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_wn,
    stride_wk,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # out = relu(x @ w.T + bias), with the bias and relu fused into the epilogue.
    offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_offs = k * BLOCK_K + offs_k
        x = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + k_offs[None, :] * stride_xk,
            mask=(offs_m[:, None] < M) & (k_offs[None, :] < K),
            other=0.0,
        )
        w = tl.load(
            w_ptr + offs_n[None, :] * stride_wn + k_offs[:, None] * stride_wk,
            mask=(offs_n[None, :] < N) & (k_offs[:, None] < K),
            other=0.0,
        )
        acc = tl.dot(x, w, acc, input_precision="ieee")
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = tl.maximum(acc + bias[None, :], 0.0)
    out = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias_shape):
        super().__init__()
        # Same construction order as the reference, so the seeded weights match.
        self.gemm = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        w = self.gemm.weight
        M, K = x.shape
        N = w.shape[0]
        out = torch.empty((M, N), device=x.device, dtype=x.dtype)
        grid = (triton.cdiv(M, 64), triton.cdiv(N, 64))
        gemm_add_relu_kernel[grid](
            x,
            w,
            self.bias,
            out,
            M,
            N,
            K,
            x.stride(0),
            x.stride(1),
            w.stride(0),
            w.stride(1),
            out.stride(0),
            out.stride(1),
            BLOCK_M=64,
            BLOCK_N=64,
            BLOCK_K=32,
        )
        return out
