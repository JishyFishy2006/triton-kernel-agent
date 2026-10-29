# Y2: KernelBench tasks chosen

Status: **decided** (2026-10-26). Full reasoning, screening chart and evidence:
[Y2: KernelBench Task Selection](https://claude.ai/artifact/NtTW78RHjMTkPJ7JjvJaa2).

## Decision

- Use the proposed 43-task list with one swap: **L2-23 out, L2-27 in**. L2-23's output is
  0 ± 1e-8 for any input, so `return zeros` passes it.
- Split: **12 dev** (for tuning prompts) / **31 held-out** (for reporting).
- Unscored canaries, run on every harness version: **L2-23** (`return zeros` must fail) and
  **L2-80** (output is always zero).

## Pinned benchmark

[KernelBench](https://github.com/ScalingIntelligence/KernelBench) at
`423217d9fda91e0c2d67e4a43bf62f96f6d104f1` (2026-03-05). Tasks changed after the paper (sizes
scaled up in `b08d959`, inputs switched to `torch.rand` in `6b1cea2`, fp32 tolerance tightened to
1e-4 in `2c3dbda`, L1-12 reference rewritten in `b6c5259`), so the paper's numbers are compared
qualitatively only.

## Tasks

Task IDs are `L<level>-<problem number>` in KernelBench; `C-<n>` are our own tasks in KernelBench
format. File names drop the number prefix. Eager times are fp32 on a Modal H100 SXM (torch
2.14.1+cu130) at the default input sizes.

| Task | KernelBench file | Group | Split | Eager (ms) | Note |
| --- | --- | --- | --- | --- | --- |
| L1-12 | `Matmul_with_diagonal_matrices_` | Paper | held-out | 0.08 | Canary: scored, but left out of fast_p |
| L1-88 | `MinGPTNewGelu` | Paper | held-out | 1.64 | |
| L1-54 | `conv_standard_3D__square_input__square_kernel` | Paper | held-out | 2.76 | Paper never solved it |
| L1-56 | `conv_standard_2D__asymmetric_input__asymmetric_kernel` | Paper | held-out | 3.56 | |
| L2-14 | `Gemm_Divide_Sum_Scaling` | Paper | held-out | 2.75 | Collapses to a matrix-vector product |
| L2-73 | `Conv2d_BatchNorm_Scaling` | Paper | held-out | 2.57 | BatchNorm, training mode |
| L2-27 | `Conv3d_HardSwish_GroupNorm_Mean` | Paper | held-out | 10.24 | Replaces L2-23 |
| L1-19 | `ReLU` | Elementwise | dev | 4.30 | Identity under `rand`: tests the input mix |
| L1-25 | `Swish` | Elementwise | dev | 10.63 | |
| L1-29 | `Softplus` | Elementwise | held-out | 4.30 | Already ~90% of peak bandwidth |
| L1-23 | `Softmax` | Reduction / norm | held-out | 8.69 | Rows of 393,216 elements |
| L1-35 | `GroupNorm_` | Reduction / norm | held-out | 9.98 | |
| L1-36 | `RMSNorm_` | Reduction / norm | dev | 14.25 | |
| L1-40 | `LayerNorm` | Reduction / norm | held-out | 8.21 | torch.compile 17× faster than eager |
| L1-47 | `Sum_reduction_over_a_dimension` | Reduction / norm | dev | 2.96 | Already ~90% of peak bandwidth |
| L1-51 | `Argmax_over_a_dimension` | Reduction / norm | held-out | 3.17 | Already ~90% of peak bandwidth |
| L1-89 | `cumsum` | Reduction / norm | held-out | 4.70 | Scan |
| L1-1 | `Square_matrix_multiplication_` | Matmul | dev | 2.67 | |
| L1-3 | `Batched_matrix_multiplication` | Matmul | held-out | 5.31 | |
| L1-7 | `Matmul_with_small_K_dimension_` | Matmul | held-out | 4.05 | K = 64, memory-bound |
| L1-8 | `Matmul_with_irregular_shapes_` | Matmul | held-out | 6.43 | 8205 × 2949 × 5921: masking |
| L1-9 | `Tall_skinny_matrix_multiplication_` | Matmul | dev | 2.58 | |
| L1-14 | `Matmul_for_upper_triangular_matrices` | Matmul | held-out | 2.72 | Half the work can be skipped |
| L1-42 | `Max_Pooling_2D` | Pool / loss / attention | dev | 10.84 | |
| L1-95 | `CrossEntropyLoss` | Pool / loss / attention | held-out | 0.49 | |
| L1-98 | `KLDivLoss` | Pool / loss / attention | held-out | 3.93 | Needs probability-valued inputs |
| L1-97 | `ScaledDotProductAttention` | Pool / loss / attention | held-out | 24.11 | Stretch: head dim 1024 |
| L2-76 | `Gemm_Add_ReLU` | L2 GEMM chain | dev | 2.77 | |
| L2-86 | `Matmul_Divide_GELU` | L2 GEMM chain | dev | 2.75 | |
| L2-95 | `Matmul_Add_Swish_Tanh_GELU_Hardtanh` | L2 GEMM chain | held-out | 2.86 | |
| L2-56 | `Matmul_Sigmoid_Sum` | L2 GEMM chain | dev | 5.29 | |
| L2-55 | `Matmul_MaxPool_Sum_Scale` | L2 GEMM chain | held-out | 5.32 | |
| L2-99 | `Matmul_GELU_Softmax` | L2 GEMM chain | held-out | 2.77 | |
| L2-30 | `Gemm_GroupNorm_Hardtanh` | L2 GEMM chain | dev | 2.83 | |
| L2-88 | `Gemm_GroupNorm_Swish_Multiply_Swish` | L2 GEMM chain | held-out | 3.05 | |
| L2-45 | `Gemm_Sigmoid_LogSumExp` | L2 GEMM chain | held-out | 8.52 | |
| L2-64 | `Gemm_LogSumExp_LeakyReLU_LeakyReLU_GELU_GELU` | L2 GEMM chain | held-out | 2.81 | |
| L2-1 | `Conv2D_ReLU_BiasAdd` | L2 conv chain | dev | 4.49 | |
| L2-67 | `Conv2d_GELU_GlobalAvgPool` | L2 conv chain | held-out | 6.56 | |
| C-1 | `Gemm_GELU_GroupSum_LogSumExp` | Own | held-out | 2.77 | Rewrite of L2-18 |
| C-2 | `Matmul_Softmax_GELU` | Own | held-out | 2.77 | L2-99 reordered |
| C-3 | `Gemm_RMSNorm_SiLU` | Own | held-out | 2.84 | |
| C-4 | `Gemm_Add_Mish_odd_shapes` | Own | held-out | 2.00 | From L2-76; 1000 × 8000 × 6000 |

Dev: L1-1, L1-9, L1-19, L1-25, L1-36, L1-42, L1-47, L2-1, L2-30, L2-56, L2-76, L2-86.

## Rejected tasks

| Task | Reason |
| --- | --- |
| L2-23 | Output is 0 ± 1e-8 for any input (kept as an unscored canary) |
| L2-80 | Output is all zeros (kept as an unscored canary) |
| L2-83 | Output is all zeros: `min(x, 0)` then `clamp(0, 1)` |
| L2-18 | Matrix-vector shortcut matches the output to 1.5e-4 |
| L2-51 | Matrix-vector shortcut matches the output to 1.2e-7 |
| L2-66 | Nondeterministic: dropout is live in training mode |

## Accepted decisions

1. **TF32: strict fp32 only.** Reference and baselines run with
   `torch.backends.cuda.matmul.allow_tf32 = False` and `torch.backends.cudnn.allow_tf32 = False`.
   Candidates must pass `input_precision="ieee"` to every fp32 `tl.dot` and may not touch the
   backend flags. TF32 speeds up 21 of 43 tasks by 1.5–7×, and 11 of those still pass 1e-4, so
   tolerance alone can't catch it. (B5 sets the flags, B7 adds the static check.)
2. **Tolerance: relative to an fp64 reference.** Compare both the fp32 reference and the
   candidate against an fp64 run; accept when the candidate's error is at most a small multiple of
   the fp32 reference's own error, with an absolute floor. A fixed 1e-4 rejects exact algebra
   (the L2-14 shortcut is off by 0.018); a fixed 1e-2 lets TF32 through on 20 of 21 tasks. (B5)
   *Amended in B5 (2026-10-27):* the multiple is 8x and the floor is 4 fp32 ulps of the output's
   largest magnitude, not an absolute 1e-4. On an H100, torch.compile stayed within 2.6x of the
   reference's error on every hidden case of every task, while an absolute 1e-4 floor let TF32
   (50-300x) through on tasks with small outputs such as L2-86, L2-99 and L2-64. With the new
   floor TF32 is caught on 146 of the 149 hidden cases where it changes the output.
   *Amended in B11 (2026-10-29):* the multiple is 16x. A correct Triton GEMM that accumulates
   over K = 8192 in order (L2-76, visible size) has 9.6x cuBLAS's error and failed at 8x. At
   16x TF32 is still caught on 141 of 149 cases; the static tl.dot check is its main defence.
3. **Algebraic shortcuts count as correct**, but are reported as their own result category and
   kept out of the speedup medians. (B5, reporting)
4. **L1-12 stays in the 43** as a canary: scored for correctness, left out of fast_p.
5. **Success metrics per level.** A 1.2× median speedup is the Level 1 target; on Level 2 GEMM
   chains in strict fp32, near 1.0× is the honest expectation (torch.compile gains a median 1.01×
   on those 13 tasks; cuBLAS already reaches ~51 of 67 TFLOPS).
6. **Own tasks C-1 to C-4 go into the repo as part of B4.** All four were screened; none is
   degenerate.

## Harness policies these force

| Policy | Rule | Build task |
| --- | --- | --- |
| Input distributions | Hidden tests mix `rand`, `randn` and one shifted or scaled distribution, with per-task domain overrides (L1-98 needs probabilities) | B6 |
| Precision | Strict IEEE fp32, as in decision 1 | B5, B7 |
| Tolerance | fp64-relative, as in decision 2 | B5 |
| Train or eval mode | Training mode, as KernelBench runs it; vary the batch size in hidden shapes (BatchNorm output differs from eval mode by up to 9.6 on L2-73) | B5, B6 |
| Hidden-shape size | Cap each hidden-shape input at ~1 GB; keep the default size only for the timed run (12 tasks have 2.1–8.6 GB inputs; L1-25 peaks at 19.3 GB) | B6 |
