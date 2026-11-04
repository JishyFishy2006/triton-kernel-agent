# Three kernels worth reading (Y7)

Picked from the 2026-11-03 batch (`batch-*-20261103`, harness v1, H100). Times are the
harness's median kernel time per forward call; "eager" and "compile" are PyTorch eager and
`torch.compile` on the same inputs. The full source of each kernel is in `docs/kernels/`.

| Task | Kernel | Setup, attempt | Time | vs eager | vs compile | % peak bandwidth |
|---|---|---|---|---|---|---|
| L1-40 LayerNorm | `l1_40_wonder_only.py` | wonder_only, 1 | 0.307 ms | 26.99× | 1.49× | 55.5% |
| L2-14 GEMM → sum | `l2_14_wonder_spark.py` | wonder_spark, 2 | 0.118 ms | 23.20× | 22.85× | 76.4% |
| L1-88 tanh GELU | `l1_88_wonder_spark.py` | wonder_spark, 1 | 0.181 ms | 8.96× | 1.01× | 88.5% |

## L1-40 LayerNorm: spread 16 rows over the whole GPU

The input is `(16, 64, 256, 256)` normalized over the last three dimensions: 16 rows of
4,194,304 values each. Eager PyTorch takes 8.28 ms because its LayerNorm kernel gives each row
to one block, so 16 blocks run on a GPU with 132 SMs and the rest sit idle.

Wonder's kernel notices the shape (`rows < 256` and a long row) and switches to a split plan:

1. `_ln_partials_kernel` cuts each row into 256 chunks of 16,384 values, giving 4,096 programs.
   Each one writes the sum and sum of squares of its chunk, accumulated in fp64.
2. `_ln_norm_kernel`, again 4,096 programs, adds up its row's 256 partials, gets the mean and
   variance, and normalizes its own chunk in fp32.

That reads `x` twice instead of once, but every SM is busy, so it runs at 0.307 ms. It also
beats `torch.compile` (0.458 ms) and roughly ties the adapted Triton tutorial kernel used as the
human baseline (27.99× eager). Two details help: the fp64 accumulation keeps
`E[x²] − E[x]²` accurate over 4 million values, and the `evict_first` / `evict_last` hints
keep the 16 MiB weight and bias in L2 while `x` streams past.

Spark's kernel uses the same split idea but also does the normalize pass in fp64, element by
element, and reaches 0.498 ms (16.6×), slower than `torch.compile`. For short rows the Wonder
kernel falls back to one program per row (`_ln_row_kernel`), which the hidden cases exercise.

## L2-14 GEMM, divide, sum, scale: an algebraic shortcut

The task computes `sum(x @ W.T / 2, dim=1) * 1.5` for `x` of `(1024, 8192)` and `W` of
`(8192, 8192)`: a 137 GFLOP GEMM whose output is immediately summed away. Because the sum is
linear, `Σⱼ (x · Wⱼ) = x · (Σⱼ Wⱼ)`, so the agent never does the GEMM:

1. `_col_sum_kernel` and `_reduce_kernel` sum `W` over its rows into one vector `s` (8 row
   splits, fp64 partials).
2. `_row_dot_kernel` computes `x · s × 0.75` for each of the 1,024 rows, in fp64.

That is a 256 MiB read of `W` plus a 32 MiB read of `x`, so the kernel is memory bound at
76% of peak bandwidth instead of compute bound. `torch.compile` does not find this rewrite
(2.70 ms). The column sum is recomputed on every call, so nothing is cached across inputs.

This is allowed by the Y2 rules (decision 3: algebraic shortcuts count as correct), and the
fp64-relative tolerance accepts it. It is still a different kind of win from a faster GEMM,
which is why the README gives results with and without it. All three setups found it on their
first or second attempt.

## L1-88 tanh GELU: one pass instead of seven

Eager PyTorch runs the MinGPT GELU formula one op at a time: `pow`, two multiplies, an add,
`tanh`, another add and two more multiplies, each a full read and write of the 256 MiB input.
The Triton kernel does the whole formula in registers, in one read and one write, at 88.5% of
the H100's bandwidth. That is close to the practical ceiling for a pure streaming kernel, so it
ties `torch.compile` (which fuses the same way) and both are about 9× eager.

Two choices are worth noting. It computes `tanh` as `(1 − e^{−2|z|}) / (1 + e^{−2|z|})`
with the sign put back afterwards, which never overflows `exp` for large `|z|`. And it
marks loads and stores `evict_first`, since no value is used twice.
