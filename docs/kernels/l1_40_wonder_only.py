import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


@triton.jit
def _ln_partials_kernel(x_ptr, ps_ptr, pq_ptr, D, ROWS, SPLIT, CHUNK, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    row = pid % ROWS
    split = pid // ROWS
    chunk_start = split * CHUNK
    chunk_end = tl.minimum(chunk_start + CHUNK, D)
    base = row.to(tl.int64) * D
    offs0 = tl.arange(0, BLOCK)
    acc_s = tl.zeros([BLOCK], dtype=tl.float64)
    acc_q = tl.zeros([BLOCK], dtype=tl.float64)
    for start in range(chunk_start, chunk_end, BLOCK):
        offs = start + offs0
        mask = offs < chunk_end
        vals = tl.load(x_ptr + base + offs, mask=mask, other=0.0, eviction_policy='evict_first')
        vf = vals.to(tl.float64)
        acc_s += vf
        acc_q += vf * vf
    tl.store(ps_ptr + row * SPLIT + split, tl.sum(acc_s))
    tl.store(pq_ptr + row * SPLIT + split, tl.sum(acc_q))


@triton.jit
def _ln_norm_kernel(x_ptr, w_ptr, b_ptr, y_ptr, ps_ptr, pq_ptr, D, ROWS, SPLIT, CHUNK, EPS, BLOCK: tl.constexpr, BLOCK_S: tl.constexpr):
    pid = tl.program_id(0)
    row = pid % ROWS
    split = pid // ROWS
    offs_s = tl.arange(0, BLOCK_S)
    mask_s = offs_s < SPLIT
    total_s = tl.sum(tl.load(ps_ptr + row * SPLIT + offs_s, mask=mask_s, other=0.0))
    total_q = tl.sum(tl.load(pq_ptr + row * SPLIT + offs_s, mask=mask_s, other=0.0))
    mean = total_s / D
    var = total_q / D - mean * mean
    var = tl.maximum(var, 0.0)
    rstd = 1.0 / tl.sqrt(var + EPS)
    mean_f = mean.to(tl.float32)
    rstd_f = rstd.to(tl.float32)
    chunk_start = split * CHUNK
    chunk_end = tl.minimum(chunk_start + CHUNK, D)
    base = row.to(tl.int64) * D
    offs0 = tl.arange(0, BLOCK)
    for start in range(chunk_start, chunk_end, BLOCK):
        offs = start + offs0
        mask = offs < chunk_end
        vals = tl.load(x_ptr + base + offs, mask=mask, other=0.0, eviction_policy='evict_first')
        weight = tl.load(w_ptr + offs, mask=mask, other=0.0, eviction_policy='evict_last')
        bias = tl.load(b_ptr + offs, mask=mask, other=0.0, eviction_policy='evict_last')
        out = (vals - mean_f) * rstd_f * weight + bias
        tl.store(y_ptr + base + offs, out, mask=mask)


@triton.jit
def _ln_row_kernel(x_ptr, w_ptr, b_ptr, y_ptr, D, EPS, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    base = row.to(tl.int64) * D
    offs0 = tl.arange(0, BLOCK)
    acc_s = tl.zeros([BLOCK], dtype=tl.float64)
    acc_q = tl.zeros([BLOCK], dtype=tl.float64)
    for start in range(0, D, BLOCK):
        offs = start + offs0
        mask = offs < D
        vals = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
        vf = vals.to(tl.float64)
        acc_s += vf
        acc_q += vf * vf
    total_s = tl.sum(acc_s)
    total_q = tl.sum(acc_q)
    mean = total_s / D
    var = total_q / D - mean * mean
    var = tl.maximum(var, 0.0)
    rstd = 1.0 / tl.sqrt(var + EPS)
    mean_f = mean.to(tl.float32)
    rstd_f = rstd.to(tl.float32)
    for start in range(0, D, BLOCK):
        offs = start + offs0
        mask = offs < D
        vals = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
        weight = tl.load(w_ptr + offs, mask=mask, other=0.0)
        bias = tl.load(b_ptr + offs, mask=mask, other=0.0)
        out = (vals - mean_f) * rstd_f * weight + bias
        tl.store(y_ptr + base + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, normalized_shape: tuple):
        super(ModelNew, self).__init__()
        self.ln = nn.LayerNorm(normalized_shape=normalized_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_c = x.contiguous()
        w_c = self.ln.weight.contiguous()
        b_c = self.ln.bias.contiguous()
        eps = self.ln.eps
        normed = math.prod(self.ln.normalized_shape)
        total = math.prod(x_c.shape)
        out = torch.empty(x_c.shape, device=x_c.device, dtype=x_c.dtype)
        if total == 0:
            return out
        rows = total // normed
        if rows >= 256 or normed <= 4096:
            _ln_row_kernel[(rows,)](x_c, w_c, b_c, out, normed, eps, BLOCK=1024, num_warps=4, num_stages=2)
            return out
        split_count = (normed + 16383) // 16384
        if split_count < 1:
            split_count = 1
        if split_count > 512:
            split_count = 512
        if rows * split_count < 1024:
            needed = (1024 + rows - 1) // rows
            max_by_chunk = (normed + 4095) // 4096
            if needed > split_count:
                split_count = needed if needed < max_by_chunk else max_by_chunk
        chunk = (normed + split_count - 1) // split_count
        block_s = 1 << (split_count - 1).bit_length()
        total_blocks = rows * split_count
        part_s = torch.empty((total_blocks,), device=x_c.device, dtype=torch.float64)
        part_q = torch.empty((total_blocks,), device=x_c.device, dtype=torch.float64)
        _ln_partials_kernel[(total_blocks,)](x_c, part_s, part_q, normed, rows, split_count, chunk, BLOCK=1024, num_warps=8, num_stages=3)
        _ln_norm_kernel[(total_blocks,)](x_c, w_c, b_c, out, part_s, part_q, normed, rows, split_count, chunk, eps, BLOCK=2048, BLOCK_S=block_s, num_warps=8, num_stages=3)
        return out
