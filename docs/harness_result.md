# Harness result schema

Every candidate eval returns exactly one `HarnessResult`
(`src/triton_kernel_agent/harness_result.py`), serialized as one JSON object. The harness
modules B5–B10 each fill in their part, B11 assembles it in the sandbox, B12 stores it as a
row of the `evals` table, and B15 turns it into a hint for the agent.

Changing a field or a category means bumping `HARNESS_VERSION`. Results are only compared
within one harness version (Y5 tags `harness-v1`). `v0` is the version under construction:
categories are still being added to it until the freeze.

## Status

The pipeline stops at the first stage that fails, and that stage sets `status`.

| Status | Set by | Meaning |
| --- | --- | --- |
| `compile_error` | B11 | Syntax, import or Triton compile error |
| `static_cheat` | B7 | The AST scan flagged the source; it never ran |
| `runtime_error` | B11 | Raised, or hit an illegal memory access, while running |
| `timeout` | B11 | Killed at the wall-clock limit |
| `wrong_shape` | B5 | Output shape or dtype differs from the reference |
| `nan_inf` | B5 | NaN or Inf where the reference has none |
| `mismatch` | B5 | Values outside tolerance |
| `runtime_cheat` | B8 | Profiler, input-sensitivity or aliasing check failed |
| `correct` | B9, B10 | Passed every check and was benchmarked |
| `harness_error` | any | Our bug, not the candidate's. Never shown to the agent; rerun after fixing |

## Fields

| Field | Type | Notes |
| --- | --- | --- |
| `task_id` | str | `L1-19`, `C-4`, ... (`tasks/manifest.toml`) |
| `status` | str | One of the statuses above |
| `error_excerpt` | str | Last 2000 characters of the error or traceback; `""` if none. On hidden cases it must not contain shapes or indices |
| `max_err` | float or null | Largest absolute error over all cases checked; can be `Infinity` or `NaN` |
| `failing_shape_class` | str or null | `visible`, `non_pow2`, `size_one`, `large`, `batch` or `input_distribution`. Required for `wrong_shape`, `nan_inf` and `mismatch`; null otherwise |
| `failing_shape` | list or null | Exact shapes of the failing inputs. **For our logs only: B15 must never put it in a hint** |
| `cheat_flags` | list of str | Sorted. Non-empty exactly when the status is a cheat (see below) |
| `timing` | object or null | `{kernel_ms, eager_ms, compile_ms}`, medians from `do_bench`. Only for `correct` |
| `pct_peak_bandwidth` | float or null | From the task's `bytes`. Only for `correct` |
| `pct_peak_flops` | float or null | From the task's `flops`; null when the task has 0 FLOPs. Only for `correct` |
| `gpu_seconds` | float | Container wall time, for cost |
| `harness_version` | str | `HARNESS_VERSION` when the result was made |
| `extra` | object | Per-module detail for debugging. Not shown to the agent |

Speedups are derived, not stored: `speedup_eager = eager_ms / kernel_ms` and
`speedup_compile = compile_ms / kernel_ms`. The `evals` table's `attempt_id` is added by the
orchestrator, not the harness.

## Cheat flags

| Flag | Kind | Caught when |
| --- | --- | --- |
| `torch_compute_call` | static | `torch.matmul`, `F.softmax`, `x @ w` and other torch compute outside `@triton.jit` kernels |
| `reference_import` | static | Imports the reference, or any module off the allowlist (torch, triton, math, ...) |
| `file_or_network` | static | `open()`, `os`, sockets, `subprocess` and similar |
| `backend_flags` | static | Touches `torch.backends` or the fp32 matmul precision (TF32 and friends) |
| `tf32_dot` | static | A `tl.dot` without `input_precision="ieee"` (Y2 decision 1) |
| `introspection` | static | `gc`, `inspect`, `sys`, frames, `__globals__`: hunting for the reference's tensors (L4 #1) |
| `monkeypatch` | static | Assigns attributes of modules, `setattr`, `exec`/`eval`, `builtins`: rewriting the harness (L4 #3) |
| `side_stream` | static | Another CUDA stream or a CUDA graph: work the timer might not wait for (L4 #2) |
| `hardcoded_data` | static | Large literal arrays or strings: precomputed answers (L4 #4) |
| `aten_compute` | runtime | aten compute ops in forward, or cuBLAS/cuDNN kernels in the profile |
| `input_insensitive` | runtime | Output did not change when the input did, even in the same buffer (L4 #4) |
| `output_aliases_input` | runtime | Output shares memory with an input (L4 #5) |
| `harness_tampered` | runtime | Timers, checks or TF32 flags differ after the candidate ran (L4 #3) |

`static_cheat` carries only static flags. `runtime_cheat` carries at least one runtime flag;
it may also carry static ones.

## Rules the constructor enforces

- Cheat flags are present exactly when the status is `static_cheat` or `runtime_cheat`.
- `timing` is present exactly when the status is `correct`. Percent of peak only appears then.
- `failing_shape_class` is present exactly for `wrong_shape`, `nan_inf` and `mismatch`.
- Strings are coerced to enums, so a result read back from JSON equals the original.

## Examples

A correct kernel (Level 2 GEMM chain):

```json
{
  "cheat_flags": [],
  "error_excerpt": "",
  "extra": {},
  "failing_shape": null,
  "failing_shape_class": null,
  "gpu_seconds": 18.2,
  "harness_version": "v0",
  "max_err": 3.1e-05,
  "pct_peak_bandwidth": 12.4,
  "pct_peak_flops": 85.3,
  "status": "correct",
  "task_id": "L2-76",
  "timing": {"compile_ms": 2.74, "eager_ms": 2.77, "kernel_ms": 2.41}
}
```

A kernel that fails a hidden non-power-of-2 shape. The agent would see `mismatch`, the error
size and `non_pow2`, but not `failing_shape`:

```json
{
  "cheat_flags": [],
  "error_excerpt": "max abs error 0.0412 exceeds tolerance 1e-4",
  "extra": {},
  "failing_shape": [[8191, 2947], [2947, 5923]],
  "failing_shape_class": "non_pow2",
  "gpu_seconds": 9.7,
  "harness_version": "v0",
  "max_err": 0.0412,
  "pct_peak_bandwidth": null,
  "pct_peak_flops": null,
  "status": "mismatch",
  "task_id": "L1-8",
  "timing": null
}
```

## Not decided here

- Algebraic shortcuts (Y2 decision 3: correct, but reported as their own category). How to
  detect them is open, so for now they are `correct` and get tagged when the results are
  reported.
- The exact tolerance rule (Y2 decision 2) belongs to B5. `max_err` is the raw absolute error,
  and B5 may add its fp64-relative ratio under `extra`.
