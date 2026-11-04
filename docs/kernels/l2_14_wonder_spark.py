import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _col_sum_kernel(w_ptr, p_ptr, N, K, stride_wn, stride_wk, stride_ps, rows_per_split, SPLIT: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)
    num_k_blocks = tl.cdiv(K, BLOCK_K)
    split_id = pid // num_k_blocks
    k_block = pid % num_k_blocks
    offs_k = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_k = offs_k < K
    n_start = split_id * rows_per_split
    n_end = n_start + rows_per_split
    if n_end > N:
        n_end = N
    acc = tl.zeros((BLOCK_K,), dtype=tl.float64)
    for n0 in range(n_start, n_end, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk
        mask = (offs_n[:, None] < n_end) & mask_k[None, :]
        vals = tl.load(ptrs, mask=mask, other=0.0)
        acc += tl.sum(vals.to(tl.float64), axis=0)
    tl.store(p_ptr + split_id * stride_ps + offs_k, acc, mask=mask_k)


@triton.jit
def _reduce_kernel(p_ptr, s_ptr, K, stride_ps, SPLIT: tl.constexpr, BLOCK_K: tl.constexpr):
    offs_k = tl.program_id(0) * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_s = tl.arange(0, SPLIT)
    mask_k = offs_k < K
    vals = tl.load(p_ptr + offs_s[:, None] * stride_ps + offs_k[None, :], mask=mask_k[None, :], other=0.0)
    total = tl.sum(vals, axis=0)
    tl.store(s_ptr + offs_k, total, mask=mask_k)


@triton.jit
def _row_dot_kernel(x_ptr, s_ptr, out_ptr, K, stride_xm, stride_xk, stride_om, factor: tl.float64, BLOCK_D: tl.constexpr):
    pid = tl.program_id(0)
    acc = tl.zeros((BLOCK_D,), dtype=tl.float64)
    for k0 in range(0, K, BLOCK_D):
        offs = k0 + tl.arange(0, BLOCK_D)
        mask = offs < K
        xv = tl.load(x_ptr + pid * stride_xm + offs * stride_xk, mask=mask, other=0.0)
        sv = tl.load(s_ptr + offs, mask=mask, other=0.0)
        acc += xv.to(tl.float64) * sv
    total = tl.sum(acc, axis=0)
    tl.store(out_ptr + pid * stride_om, total * factor)


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, scaling_factor):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.scaling_factor = scaling_factor

    def forward(self, x):
        w = self.weight
        M = x.shape[0]
        K = x.shape[1]
        N = w.shape[0]
        out = torch.empty((M, 1), device=x.device, dtype=x.dtype)
        if M == 0:
            return out
        if K == 0:
            _row_dot_kernel[(M,)](x, x, out, K, x.stride(0), x.stride(1), out.stride(0), float(self.scaling_factor) * 0.5, BLOCK_D=1024, num_warps=8)
            return out
        split_n = 8
        rows_per_split = (N + split_n - 1) // split_n
        partial = torch.empty((split_n, K), device=w.device, dtype=torch.float64)
        col_sums = torch.empty((K,), device=w.device, dtype=torch.float64)
        block_k = 64
        grid_col = (split_n * ((K + block_k - 1) // block_k),)
        _col_sum_kernel[grid_col](w, partial, N, K, w.stride(0), w.stride(1), partial.stride(0), rows_per_split, SPLIT=split_n, BLOCK_N=64, BLOCK_K=block_k, num_warps=8)
        grid_red = ((K + 63) // 64,)
        _reduce_kernel[grid_red](partial, col_sums, K, partial.stride(0), SPLIT=split_n, BLOCK_K=64, num_warps=4)
        factor = float(self.scaling_factor) * 0.5
        _row_dot_kernel[(M,)](x, col_sums, out, K, x.stride(0), x.stride(1), out.stride(0), factor, BLOCK_D=1024, num_warps=8)
        return out
