"""
Triton tutorial kernels (adapted): fused softmax + layer norm.

Baseline label: "Triton tutorial kernels (adapted)" -- NOT hand-written from scratch.
The single-block ("fused") paths follow the official Triton tutorials step for step:
  02 fused softmax: https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html
  05 layer norm:    https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html

Additions on top of the tutorials (not in the originals), needed for the task shapes:
  * Softmax (L1-23, 4096 x 393,216): rows that don't fit in one block go through a
    split-row "online softmax": per-chunk (max, sum of exp) partials -> per-row
    merge -> normalize. Works for any row length and any number of rows.
  * LayerNorm (L1-40, 16 x (64, 256, 256)): normalizes over ALL of normalized_shape
    (N = product of its dims, weight/bias flattened to N). When N is too big for one
    block, or there are too few rows to fill the GPU, each row is split across many
    programs: per-chunk (mean, M2) partials merged with Chan's parallel-variance
    formula, then a 2D-grid normalize. The backward pass has a matching split path.
  * Hidden cases: any N (odd sizes, size 1), any batch size, empty tensors,
    non-contiguous inputs, fp16/bf16 inputs (all math in fp32), int64 row offsets.

Harness note: this file is the kernels only. The self-test (_check_softmax,
_check_layernorm, __main__) lives in the author's copy; it calls torch.softmax and
F.layer_norm, which the static scan rightly flags in a candidate. baselines/l1_23.py and
baselines/l1_40.py add the ModelNew wrappers; scripts/run_baselines.py joins them.
"""
import functools
import math

import torch
import triton
import triton.language as tl
from triton.runtime import driver


@functools.lru_cache(maxsize=None)
def _device_props(index):
    p = driver.active.utils.get_device_properties(index)
    return p["multiprocessor_count"], p["max_num_regs"], p["max_shared_mem"], p["warpSize"]


def _num_sms(device):
    return _device_props(device.index if device.index is not None else torch.cuda.current_device())[0]


# =============================================================================
# 1. SOFTMAX
# =============================================================================
SOFTMAX_MAX_FUSED_COLS = 16384   # above this, a whole row in registers spills -> split path
SOFTMAX_CHUNK = 4096             # columns per program in the split path


# --- 1a. Fused single-block kernel (tutorial 02) -----------------------------
# Each row is loaded ONCE, max/exp/sum/div happen on-chip, result written ONCE.
@triton.jit
def softmax_kernel(out_ptr, in_ptr, in_row_stride, out_row_stride,
                   n_rows, n_cols,
                   BLOCK: tl.constexpr, STAGES: tl.constexpr):
    # Persistent grid: ~(#SMs * occupancy) programs, each walks rows with a grid stride.
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    offs = tl.arange(0, BLOCK)
    valid = offs < n_cols
    # num_stages -> compiler software-pipelines: next row's load overlaps this row's math.
    for r in tl.range(pid, n_rows, nprog, num_stages=STAGES):
        r64 = r.to(tl.int64)
        # Pad with -inf: padded lanes never win the max and add exp(-inf)=0 to the sum.
        x = tl.load(in_ptr + r64 * in_row_stride + offs, mask=valid,
                    other=-float("inf")).to(tl.float32)
        x = x - tl.max(x, axis=0)
        e = tl.exp(x)
        y = e / tl.sum(e, axis=0)
        tl.store(out_ptr + r64 * out_row_stride + offs, y, mask=valid)


# --- 1b. Split-row online softmax (addition, for long rows) ------------------
# Stage 1: grid (rows, chunks). Each program reduces one chunk to (m_c, s_c) where
#          m_c = max(chunk), s_c = sum(exp(chunk - m_c)).
@triton.jit
def softmax_partial_kernel(X, PMax, PSum, row_stride, n_cols, n_chunks,
                           CHUNK: tl.constexpr):
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    offs = chunk * CHUNK + tl.arange(0, CHUNK)
    x = tl.load(X + row.to(tl.int64) * row_stride + offs, mask=offs < n_cols,
                other=-float("inf")).to(tl.float32)
    m = tl.max(x, axis=0)
    s = tl.sum(tl.exp(x - m), axis=0)
    s = tl.where(m == -float("inf"), 0.0, s)        # all -inf chunk contributes nothing
    tl.store(PMax + row * n_chunks + chunk, m)
    tl.store(PSum + row * n_chunks + chunk, s)


# Stage 2: grid (rows,). Merge partials: M = max(m_c), S = sum(s_c * exp(m_c - M)).
@triton.jit
def softmax_combine_kernel(PMax, PSum, RowMax, RowSum, n_chunks,
                           BLOCK_C: tl.constexpr):
    row = tl.program_id(0)
    base = row * n_chunks
    mx = tl.full([BLOCK_C], -float("inf"), tl.float32)
    for start in range(0, n_chunks, BLOCK_C):
        c = start + tl.arange(0, BLOCK_C)
        mx = tl.maximum(mx, tl.load(PMax + base + c, mask=c < n_chunks, other=-float("inf")))
    M = tl.max(mx, axis=0)
    acc = tl.zeros([BLOCK_C], dtype=tl.float32)
    for start in range(0, n_chunks, BLOCK_C):
        c = start + tl.arange(0, BLOCK_C)
        mc = c < n_chunks
        m = tl.load(PMax + base + c, mask=mc, other=-float("inf"))
        s = tl.load(PSum + base + c, mask=mc, other=0.0)
        acc += tl.where(m == -float("inf"), 0.0, s * tl.exp(m - M))
    tl.store(RowMax + row, M)
    tl.store(RowSum + row, tl.sum(acc, axis=0))


# Stage 3: grid (rows, chunks). y = exp(x - M) / S.
@triton.jit
def softmax_normalize_kernel(X, Y, RowMax, RowSum, in_row_stride, out_row_stride,
                             n_cols, CHUNK: tl.constexpr):
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    offs = chunk * CHUNK + tl.arange(0, CHUNK)
    msk = offs < n_cols
    r64 = row.to(tl.int64)
    x = tl.load(X + r64 * in_row_stride + offs, mask=msk, other=-float("inf")).to(tl.float32)
    y = tl.exp(x - tl.load(RowMax + row)) / tl.load(RowSum + row)
    tl.store(Y + r64 * out_row_stride + offs, y, mask=msk)


# --- launchers ---------------------------------------------------------------
_softmax_occupancy_cache = {}


def _softmax_fused(x2d, y2d):
    n_rows, n_cols = x2d.shape
    BLOCK = triton.next_power_of_2(n_cols)
    num_warps = 8
    n_sm, n_regs_sm, smem_sm, warp = _device_props(x2d.device.index)
    stages = 4 if smem_sm > 200_000 else 2

    key = (x2d.device.index, x2d.dtype, BLOCK, stages, num_warps)
    if key not in _softmax_occupancy_cache:
        # Compile once (no launch) to read register + shared-memory usage, then size the
        # grid to fill every SM exactly to its occupancy limit (NVIDIA path of the tutorial).
        k = softmax_kernel.warmup(y2d, x2d, x2d.stride(0), y2d.stride(0), n_rows, n_cols,
                                  BLOCK=BLOCK, STAGES=stages, num_warps=num_warps, grid=(1,))
        k._init_handles()
        occ = n_regs_sm // max(k.n_regs * warp * num_warps, 1)
        occ = min(occ, smem_sm // max(k.metadata.shared, 1))
        _softmax_occupancy_cache[key] = max(occ, 1) * n_sm
    n_progs = min(_softmax_occupancy_cache[key], n_rows)

    softmax_kernel[(n_progs, 1, 1)](y2d, x2d, x2d.stride(0), y2d.stride(0), n_rows, n_cols,
                                    BLOCK=BLOCK, STAGES=stages, num_warps=num_warps)


def _softmax_split(x2d, y2d):
    n_rows, n_cols = x2d.shape
    n_chunks = triton.cdiv(n_cols, SOFTMAX_CHUNK)
    dev = x2d.device
    pmax = torch.empty((n_rows, n_chunks), dtype=torch.float32, device=dev)
    psum = torch.empty_like(pmax)
    rmax = torch.empty(n_rows, dtype=torch.float32, device=dev)
    rsum = torch.empty_like(rmax)

    softmax_partial_kernel[(n_rows, n_chunks)](x2d, pmax, psum, x2d.stride(0), n_cols, n_chunks,
                                               CHUNK=SOFTMAX_CHUNK, num_warps=8)
    BLOCK_C = min(1024, triton.next_power_of_2(n_chunks))
    softmax_combine_kernel[(n_rows,)](pmax, psum, rmax, rsum, n_chunks, BLOCK_C=BLOCK_C)
    softmax_normalize_kernel[(n_rows, n_chunks)](x2d, y2d, rmax, rsum, x2d.stride(0), y2d.stride(0),
                                                 n_cols, CHUNK=SOFTMAX_CHUNK, num_warps=8)


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Softmax along `dim` for a tensor of any shape."""
    if x.numel() == 0:
        return torch.empty_like(x)
    xt = x.movedim(dim, -1).contiguous()
    n_cols = xt.shape[-1]
    x2d = xt.view(-1, n_cols)
    y2d = torch.empty_like(x2d)
    if n_cols <= SOFTMAX_MAX_FUSED_COLS:
        _softmax_fused(x2d, y2d)          # tutorial path
    else:
        _softmax_split(x2d, y2d)          # long rows (e.g. L1-23)
    return y2d.view(xt.shape).movedim(-1, dim)


# =============================================================================
# 2. LAYER NORMALIZATION   y = (x - mean) / sqrt(var + eps) * w + b
# =============================================================================
LN_CHUNK = 4096   # elements per program in the split path


# --- 2a. Fused single-block kernels (tutorial 05) ----------------------------
@triton.jit
def layernorm_fwd_kernel(X, Y, W, B, Mean, Rstd, row_stride, N, eps,
                         BLOCK: tl.constexpr):
    row = tl.program_id(0)
    X += row.to(tl.int64) * row_stride
    Y += row.to(tl.int64) * row_stride
    # pass 1: mean (fp32 vector accumulator, one reduction at the end)
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for start in range(0, N, BLOCK):
        c = start + tl.arange(0, BLOCK)
        acc += tl.load(X + c, mask=c < N, other=0.0).to(tl.float32)
    mean = tl.sum(acc, axis=0) / N
    # pass 2: variance, two-pass for stability
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for start in range(0, N, BLOCK):
        c = start + tl.arange(0, BLOCK)
        v = tl.load(X + c, mask=c < N, other=0.0).to(tl.float32)
        d = tl.where(c < N, v - mean, 0.0)
        acc += d * d
    var = tl.sum(acc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)
    # pass 3: normalize + affine
    for start in range(0, N, BLOCK):
        c = start + tl.arange(0, BLOCK)
        m = c < N
        w = tl.load(W + c, mask=m).to(tl.float32)
        b = tl.load(B + c, mask=m).to(tl.float32)
        v = tl.load(X + c, mask=m, other=0.0).to(tl.float32)
        tl.store(Y + c, (v - mean) * rstd * w + b, mask=m)


@triton.jit
def layernorm_bwd_dx_kernel(DX, DY, DW_partial, DB_partial, X, W, Mean, Rstd, Lock,
                            row_stride, N,
                            GROUP_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """dx for one row + accumulate this row's dw/db into partial buffer row % GROUP_M
    under a spinlock. Lock layout: [0,GROUP_M) locks, [GROUP_M,2*GROUP_M) written-flags."""
    row = tl.program_id(0)
    c = tl.arange(0, BLOCK_N)
    m = c < N
    off = row.to(tl.int64) * row_stride
    X += off
    DY += off
    DX += off
    slot = row % GROUP_M
    lock = Lock + slot
    seen = lock + GROUP_M
    dw_ptr = DW_partial + slot * N + c
    db_ptr = DB_partial + slot * N + c

    x = tl.load(X + c, mask=m, other=0).to(tl.float32)
    dy = tl.load(DY + c, mask=m, other=0).to(tl.float32)
    w = tl.load(W + c, mask=m).to(tl.float32)
    mean = tl.load(Mean + row)
    rstd = tl.load(Rstd + row)
    # dx = rstd * (w*dy - xhat*c1 - c2),  c1 = mean(xhat*w*dy),  c2 = mean(w*dy)
    xhat = tl.where(m, (x - mean) * rstd, 0.0)
    wdy = tl.where(m, w * dy, 0.0)
    c1 = tl.sum(xhat * wdy, axis=0) / N
    c2 = tl.sum(wdy, axis=0) / N
    dx = (wdy - (xhat * c1 + c2)) * rstd
    tl.store(DX + c, dx, mask=m)

    pdw = (dy * xhat).to(DW_partial.dtype.element_ty)
    pdb = dy.to(DB_partial.dtype.element_ty)
    while tl.atomic_cas(lock, 0, 1) == 1:
        pass
    if tl.load(seen) == 0:
        tl.atomic_xchg(seen, 1)
    else:
        pdw += tl.load(dw_ptr, mask=m)
        pdb += tl.load(db_ptr, mask=m)
    tl.store(dw_ptr, pdw, mask=m)
    tl.store(db_ptr, pdb, mask=m)
    tl.debug_barrier()
    tl.atomic_xchg(lock, 0)


@triton.jit
def layernorm_bwd_dwdb_kernel(DW_partial, DB_partial, DW, DB, M, N,
                              BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """Sum the M partial buffers into final dw/db; each program owns a column strip."""
    pid = tl.program_id(0)
    cols = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    acc_w = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_b = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for start in range(0, M, BLOCK_M):
        rows = start + tl.arange(0, BLOCK_M)
        msk = (rows[:, None] < M) & (cols[None, :] < N)
        off = rows[:, None] * N + cols[None, :]
        acc_w += tl.load(DW_partial + off, mask=msk, other=0.0).to(tl.float32)
        acc_b += tl.load(DB_partial + off, mask=msk, other=0.0).to(tl.float32)
    tl.store(DW + cols, tl.sum(acc_w, axis=0), mask=cols < N)
    tl.store(DB + cols, tl.sum(acc_b, axis=0), mask=cols < N)


# --- 2b. Split-row forward (addition): same two-stage idea as the tutorial's ---
#         dw/db reduction, applied to mean/variance.
# Stage 1: grid (rows, chunks). Each chunk -> (mean_c, M2_c) over its count n_c.
@triton.jit
def ln_partial_stats_kernel(X, PMean, PM2, row_stride, N, n_chunks,
                            CHUNK: tl.constexpr):
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    offs = chunk * CHUNK + tl.arange(0, CHUNK)
    m = offs < N
    x = tl.load(X + row.to(tl.int64) * row_stride + offs, mask=m, other=0.0).to(tl.float32)
    cnt = tl.minimum(N - chunk * CHUNK, CHUNK).to(tl.float32)
    mean_c = tl.sum(x, axis=0) / cnt
    d = tl.where(m, x - mean_c, 0.0)
    tl.store(PMean + row * n_chunks + chunk, mean_c)
    tl.store(PM2 + row * n_chunks + chunk, tl.sum(d * d, axis=0))


# Stage 2: grid (rows,). Chan's merge:  mean = sum(n_c*mean_c)/N,
#          M2 = sum(M2_c + n_c*(mean_c - mean)^2),  var = M2/N.
@triton.jit
def ln_combine_stats_kernel(PMean, PM2, Mean, Rstd, N, n_chunks, eps,
                            CHUNK: tl.constexpr, BLOCK_C: tl.constexpr):
    row = tl.program_id(0)
    base = row * n_chunks
    acc = tl.zeros([BLOCK_C], dtype=tl.float32)
    for start in range(0, n_chunks, BLOCK_C):
        c = start + tl.arange(0, BLOCK_C)
        mc = c < n_chunks
        cnt = tl.where(mc, tl.minimum(N - c * CHUNK, CHUNK), 0).to(tl.float32)
        acc += cnt * tl.load(PMean + base + c, mask=mc, other=0.0)
    mean = tl.sum(acc, axis=0) / N
    acc = tl.zeros([BLOCK_C], dtype=tl.float32)
    for start in range(0, n_chunks, BLOCK_C):
        c = start + tl.arange(0, BLOCK_C)
        mc = c < n_chunks
        cnt = tl.where(mc, tl.minimum(N - c * CHUNK, CHUNK), 0).to(tl.float32)
        d = tl.load(PMean + base + c, mask=mc, other=0.0) - mean
        acc += tl.load(PM2 + base + c, mask=mc, other=0.0) + cnt * d * d
    var = tl.sum(acc, axis=0) / N
    tl.store(Mean + row, mean)
    tl.store(Rstd + row, 1.0 / tl.sqrt(var + eps))


# Stage 3: grid (rows, chunks). Normalize + affine.
@triton.jit
def ln_apply_kernel(X, Y, W, B, Mean, Rstd, row_stride, N, CHUNK: tl.constexpr):
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    offs = chunk * CHUNK + tl.arange(0, CHUNK)
    m = offs < N
    off = row.to(tl.int64) * row_stride
    mean = tl.load(Mean + row)
    rstd = tl.load(Rstd + row)
    x = tl.load(X + off + offs, mask=m, other=0.0).to(tl.float32)
    w = tl.load(W + offs, mask=m, other=0.0).to(tl.float32)
    b = tl.load(B + offs, mask=m, other=0.0).to(tl.float32)
    tl.store(Y + off + offs, (x - mean) * rstd * w + b, mask=m)


# --- 2c. Split-row backward (addition) ---------------------------------------
@triton.jit
def ln_bwd_partial_kernel(X, DY, W, Mean, Rstd, P1, P2, row_stride, N, n_chunks,
                          CHUNK: tl.constexpr):
    """grid (rows, chunks): partial sums of xhat*w*dy and w*dy."""
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    offs = chunk * CHUNK + tl.arange(0, CHUNK)
    m = offs < N
    off = row.to(tl.int64) * row_stride
    x = tl.load(X + off + offs, mask=m, other=0.0).to(tl.float32)
    dy = tl.load(DY + off + offs, mask=m, other=0.0).to(tl.float32)
    w = tl.load(W + offs, mask=m, other=0.0).to(tl.float32)
    xhat = tl.where(m, (x - tl.load(Mean + row)) * tl.load(Rstd + row), 0.0)
    wdy = tl.where(m, w * dy, 0.0)
    tl.store(P1 + row * n_chunks + chunk, tl.sum(xhat * wdy, axis=0))
    tl.store(P2 + row * n_chunks + chunk, tl.sum(wdy, axis=0))


@triton.jit
def ln_bwd_combine_kernel(P1, P2, C1, C2, N, n_chunks, BLOCK_C: tl.constexpr):
    """grid (rows,): c1 = sum(P1)/N, c2 = sum(P2)/N."""
    row = tl.program_id(0)
    base = row * n_chunks
    a1 = tl.zeros([BLOCK_C], dtype=tl.float32)
    a2 = tl.zeros([BLOCK_C], dtype=tl.float32)
    for start in range(0, n_chunks, BLOCK_C):
        c = start + tl.arange(0, BLOCK_C)
        mc = c < n_chunks
        a1 += tl.load(P1 + base + c, mask=mc, other=0.0)
        a2 += tl.load(P2 + base + c, mask=mc, other=0.0)
    tl.store(C1 + row, tl.sum(a1, axis=0) / N)
    tl.store(C2 + row, tl.sum(a2, axis=0) / N)


@triton.jit
def ln_bwd_dx_split_kernel(DX, DY, X, W, Mean, Rstd, C1, C2, row_stride, N,
                           CHUNK: tl.constexpr):
    """grid (rows, chunks): dx = rstd * (w*dy - xhat*c1 - c2)."""
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    offs = chunk * CHUNK + tl.arange(0, CHUNK)
    m = offs < N
    off = row.to(tl.int64) * row_stride
    x = tl.load(X + off + offs, mask=m, other=0.0).to(tl.float32)
    dy = tl.load(DY + off + offs, mask=m, other=0.0).to(tl.float32)
    w = tl.load(W + offs, mask=m, other=0.0).to(tl.float32)
    rstd = tl.load(Rstd + row)
    xhat = (x - tl.load(Mean + row)) * rstd
    dx = (w * dy - (xhat * tl.load(C1 + row) + tl.load(C2 + row))) * rstd
    tl.store(DX + off + offs, dx, mask=m)


@triton.jit
def ln_bwd_dwdb_split_kernel(X, DY, Mean, Rstd, DW, DB, M, N, row_stride,
                             BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """grid (cdiv(N, BLOCK_N),): each program owns a column strip and loops over the
    (few) rows -- no locks needed when rows are few and N is huge."""
    pid = tl.program_id(0)
    cols = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    cm = cols < N
    acc_w = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_b = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for start in range(0, M, BLOCK_M):
        rows = start + tl.arange(0, BLOCK_M)
        rm = rows < M
        msk = rm[:, None] & cm[None, :]
        off = rows[:, None].to(tl.int64) * row_stride + cols[None, :]
        x = tl.load(X + off, mask=msk, other=0.0).to(tl.float32)
        dy = tl.load(DY + off, mask=msk, other=0.0).to(tl.float32)
        mean = tl.load(Mean + rows, mask=rm, other=0.0)
        rstd = tl.load(Rstd + rows, mask=rm, other=0.0)
        xhat = (x - mean[:, None]) * rstd[:, None]
        acc_w += tl.where(msk, dy * xhat, 0.0)
        acc_b += dy
    tl.store(DW + cols, tl.sum(acc_w, axis=0), mask=cm)
    tl.store(DB + cols, tl.sum(acc_b, axis=0), mask=cm)


# --- autograd wrapper --------------------------------------------------------
def _ln_use_split(M, N, element_size, device):
    if N > 65536 // element_size:          # row doesn't fit in one block (tutorial's 64KB cap)
        return True
    # Too few rows to fill the GPU and rows long enough to be worth splitting.
    return M < _num_sms(device) and N > LN_CHUNK


class TritonLayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, normalized_shape, weight, bias, eps):
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        normalized_shape = tuple(normalized_shape)
        k = len(normalized_shape)
        if tuple(x.shape[x.dim() - k:]) != normalized_shape:
            raise ValueError(f"normalized_shape {normalized_shape} doesn't match input {tuple(x.shape)}")

        N = math.prod(normalized_shape)                 # normalize over ALL trailing dims
        x2d = x.contiguous().view(-1, N)
        M = x2d.shape[0]
        dev = x.device
        w = weight.contiguous().view(-1) if weight is not None else torch.ones(N, dtype=x.dtype, device=dev)
        b = bias.contiguous().view(-1) if bias is not None else torch.zeros(N, dtype=x.dtype, device=dev)
        y2d = torch.empty_like(x2d)
        mean = torch.empty(M, dtype=torch.float32, device=dev)
        rstd = torch.empty(M, dtype=torch.float32, device=dev)

        split = M > 0 and N > 0 and _ln_use_split(M, N, x.element_size(), dev)
        BLOCK, num_warps = 0, 4
        if M > 0 and N > 0:
            if not split:
                # Tutorial path: one program per row, whole row in one block.
                BLOCK = triton.next_power_of_2(N)
                num_warps = min(max(BLOCK // 256, 1), 8)
                layernorm_fwd_kernel[(M,)](x2d, y2d, w, b, mean, rstd, x2d.stride(0), N, eps,
                                           BLOCK=BLOCK, num_warps=num_warps)
            else:
                # Split path: rows spread over (M, n_chunks) programs + two-stage stats reduction.
                n_chunks = triton.cdiv(N, LN_CHUNK)
                pmean = torch.empty((M, n_chunks), dtype=torch.float32, device=dev)
                pm2 = torch.empty_like(pmean)
                ln_partial_stats_kernel[(M, n_chunks)](x2d, pmean, pm2, x2d.stride(0), N, n_chunks,
                                                       CHUNK=LN_CHUNK, num_warps=8)
                BLOCK_C = min(1024, triton.next_power_of_2(n_chunks))
                ln_combine_stats_kernel[(M,)](pmean, pm2, mean, rstd, N, n_chunks, eps,
                                              CHUNK=LN_CHUNK, BLOCK_C=BLOCK_C)
                ln_apply_kernel[(M, n_chunks)](x2d, y2d, w, b, mean, rstd, x2d.stride(0), N,
                                               CHUNK=LN_CHUNK, num_warps=8)

        ctx.save_for_backward(x2d, w, mean, rstd)
        ctx.split, ctx.N, ctx.BLOCK, ctx.num_warps = split, N, BLOCK, num_warps
        ctx.x_shape = x.shape
        ctx.w_shape = weight.shape if weight is not None else None
        ctx.b_shape = bias.shape if bias is not None else None
        return y2d.view(x.shape)

    @staticmethod
    def backward(ctx, dy):
        x2d, w, mean, rstd = ctx.saved_tensors
        N = ctx.N
        M = x2d.shape[0]
        dev = x2d.device
        dy2d = dy.contiguous().view(-1, N)
        dx2d = torch.empty_like(x2d)
        dw = torch.zeros(N, dtype=w.dtype, device=dev)
        db = torch.zeros(N, dtype=w.dtype, device=dev)

        if M > 0 and N > 0 and not ctx.split:
            # Tutorial path: fused dx + lock-based partial dw/db, then strip reduction.
            GROUP_M = 256 if N <= 1024 else 128 if N <= 4096 else 96 if N <= 8192 else 64
            locks = torch.zeros(2 * GROUP_M, dtype=torch.int32, device=dev)
            dw_part = torch.zeros((GROUP_M, N), dtype=x2d.dtype, device=dev)
            db_part = torch.zeros((GROUP_M, N), dtype=x2d.dtype, device=dev)
            layernorm_bwd_dx_kernel[(M,)](dx2d, dy2d, dw_part, db_part, x2d, w, mean, rstd, locks,
                                          x2d.stride(0), N,
                                          GROUP_M=GROUP_M, BLOCK_N=ctx.BLOCK, num_warps=ctx.num_warps)
            grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]),)
            layernorm_bwd_dwdb_kernel[grid](dw_part, db_part, dw, db, min(GROUP_M, M), N,
                                            BLOCK_M=32, BLOCK_N=128)
        elif M > 0 and N > 0:
            # Split path.
            n_chunks = triton.cdiv(N, LN_CHUNK)
            p1 = torch.empty((M, n_chunks), dtype=torch.float32, device=dev)
            p2 = torch.empty_like(p1)
            c1 = torch.empty(M, dtype=torch.float32, device=dev)
            c2 = torch.empty_like(c1)
            ln_bwd_partial_kernel[(M, n_chunks)](x2d, dy2d, w, mean, rstd, p1, p2, x2d.stride(0), N,
                                                 n_chunks, CHUNK=LN_CHUNK, num_warps=8)
            BLOCK_C = min(1024, triton.next_power_of_2(n_chunks))
            ln_bwd_combine_kernel[(M,)](p1, p2, c1, c2, N, n_chunks, BLOCK_C=BLOCK_C)
            ln_bwd_dx_split_kernel[(M, n_chunks)](dx2d, dy2d, x2d, w, mean, rstd, c1, c2,
                                                  x2d.stride(0), N, CHUNK=LN_CHUNK, num_warps=8)
            BLOCK_N = 256
            ln_bwd_dwdb_split_kernel[(triton.cdiv(N, BLOCK_N),)](x2d, dy2d, mean, rstd, dw, db, M, N,
                                                                x2d.stride(0), BLOCK_M=16, BLOCK_N=BLOCK_N)

        grad_w = dw.view(ctx.w_shape) if ctx.needs_input_grad[2] else None
        grad_b = db.view(ctx.b_shape) if ctx.needs_input_grad[3] else None
        return dx2d.view(ctx.x_shape), None, grad_w, grad_b, None


def layer_norm(x, normalized_shape, weight=None, bias=None, eps=1e-5):
    return TritonLayerNorm.apply(x, normalized_shape, weight, bias, eps)
