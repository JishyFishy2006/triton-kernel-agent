# task: L1-1
# expect: tf32_dot
# red-team: skip-static
# expect-status: mismatch
# Y4 red team (D/d5_tf32_alias): Plain TF32 through a tl.dot alias: the fp64-relative tolerance catches it.
# Plain TF32 GEMM through the alias: does the tolerance alone catch it on every hidden case?
import torch
import torch.nn as nn
import triton
import triton.language as tl

_dot = tl.dot  # module-level alias: the scanner only resolves import aliases


@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, sam, sak, sbk, sbn, scm, scn,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        kk = k * BLOCK_K + offs_k
        a = tl.load(a_ptr + offs_m[:, None] * sam + kk[None, :] * sak,
                    mask=(offs_m[:, None] < M) & (kk[None, :] < K), other=0.0)
        b = tl.load(b_ptr + kk[:, None] * sbk + offs_n[None, :] * sbn,
                    mask=(kk[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc = _dot(a, b, acc, input_precision="tf32")
    tl.store(c_ptr + offs_m[:, None] * scm + offs_n[None, :] * scn, acc,
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        M, K = A.shape
        _, N = B.shape
        C = torch.empty((M, N), device=A.device, dtype=A.dtype)
        grid = (triton.cdiv(M, 128), triton.cdiv(N, 128))
        matmul_kernel[grid](A, B, C, M, N, K, A.stride(0), A.stride(1), B.stride(0), B.stride(1),
                            C.stride(0), C.stride(1), BLOCK_M=128, BLOCK_N=128, BLOCK_K=32,
                            num_warps=8, num_stages=3)
        return C
