# Y4: red team, and the isolated harness

Status: **harness redesigned** (2026-11-02). Six red-team agents attacked the B5–B11 harness
on Modal H100s, one area each, starting from the L4 cheat list. 26 attacks got a wrong or lazy
kernel graded `correct`, or bought an unearned speedup. Every attack is now a fixture
(`tests/fixtures/candidates/y4_*.py`) with the result it must get.

## What broke, and why

All 26 had one root cause: the candidate ran in the same Python process as the reference model,
the checker, the timer and the code that printed the verdict. Anything in that process can be
reached (`torch.sys.modules`, stack frames, `gc`) and rewritten (`type.__setattr__`, hooks,
dispatch modes), and in-process guards only see what they were told to watch.

| Group | Attacks | Example |
| --- | --- | --- |
| Rewrite the verdict or the checker | A3, A6, B3, C02, C06, C07, C08b, F1, F4, F5 | forge the result object: `empty_like` graded correct at 100x |
| Outputs not checked when it mattered | A4, B1, D1, D2, D3, D9, E3 | memoize on `(pointer, shape, version)`: 225x |
| A tensor subclass lies to the comparison | C01, E1, E2 | zeros pass a GEMM with max error 0 |
| Slow the eager / compile baselines | B2, B5, B7 | a global hook stalls the reference: 1.3-1.8x for an identical kernel, cached into later attempts |
| Detect the audit, compute with PyTorch otherwise | A5, C04, E4 | correct at cuBLAS speed |
| State carried between evals | F3 | an honest-looking attempt patches the installed harness; every later eval in that sandbox passes |
| Hidden sizes leaked into feedback | F2 | sizes spelled as single digits survive redaction |

## The design now

The harness process is a **trusted parent that never imports candidate code**.

- **Worker processes.** The candidate runs in `worker.py`, started by the parent for each eval
  as the unprivileged user `nobody`, with `python -I`, its own HOME and Triton cache, and no
  inherited stdout. It cannot read or write the harness (`/root` is locked to 0700), signal or
  trace the parent, or see the request and its nonce. Every process the user owns is killed
  between phases and before the next eval.
- **The parent decides.** It builds every input and computes every reference output (fp32 and
  fp64) itself. Outputs cross as raw bytes in files with validated headers, no pickle, so the
  parent compares plain tensors. Whatever the worker says about itself is a claim.
- **Same buffers, new values.** After the hidden cases, the worker runs a case, then gets new
  values written into the same input tensors. Its second output is compared with the reference
  for the new values (input_insensitive).
- **Clean baselines.** Eager and torch.compile are timed in workers that never see candidate
  code, while no candidate process exists. Cached baselines therefore come only from clean runs.
- **The parent's clock.** The worker runs calls back to back and ends each batch with a
  device-wide sync; the parent times the request with `time.perf_counter`.
- **Checked timing.** Batches alternate between two visible-shape input sets, both checked in
  the correctness stage, written into the same buffers between batches. Each batch makes a
  random number of calls, and its last output is checked against the candidate's own checked
  output for that set (changed_after_checks).
- **Physics.** A time faster than moving the task's bytes (inputs, weights, outputs) at peak
  bandwidth, with a 105% margin, means work was skipped: caching or lazy calls
  (impossible_speed). FLOPs give no floor: eager L1-56 beats its FLOP count's floor (8.70 ms
  vs 3.56 ms), and L2-14 collapses to a matrix-vector product.
- **Secrecy.** Once a hidden case has run, nothing the worker says reaches the agent: messages
  are withheld, exception names come from a fixed list (a custom class could be named after a
  size), crashes are reported without their exit code, and mismatches without error sizes.
  A worker can stash a hidden size and raise it on the visible shape later.
- **Cleanup.** Workers die with the harness (`PR_SET_PDEATHSIG`); every process and every
  file of the worker user in the world-writable directories is removed when a worker stops,
  and before and after each eval; the runner cleans up after its own wall-clock kill.
- **Static scan.** Now a speed bump in front, not the defence: module attributes past the
  allowlist (`torch.sys`), private torch internals, dunders, computed `getattr`, aliases of
  `torch.cuda` and `tl.dot`, `tl.core`, banned builtins used as values (`f = exec`), tensor
  subclasses, buffer identity and version counters, hooks, decorators around `@triton.jit`,
  and fp16/bf16/fp8 values (decision below).

GPU memory freed by one process reads back as zeros in another (checked on an H100 with a
10 GB pattern), so a worker cannot find the parent's reference outputs in fresh allocations.

## Result on H100s

Every attack was rerun through the new harness twice: normally, and with the static scan
skipped to test the layers behind it.

- **Stopped without the static scan:** 32 of 37 attack files. 26 fail outright: verdict,
  checker and subclass attacks get `mismatch` or `compile_error`; caching and lazy timing get
  `impossible_speed` or `input_insensitive`; the shape leak returns "message withheld". 6 still
  run but gain nothing: timer patches, baseline stalls and side streams measure an honest
  kernel at 1.00x.
- **Stopped only by the static scan:** 5, listed as known limits below.

## Known limits

- **PyTorch doing the math behind a detected audit** (A5, C04, E4). The aten and kernel audit
  runs inside the worker, which the candidate controls, so it can be detected. Only the static
  scan stops these. The gain is bounded: they run at eager speed, so they inflate the correct
  count, not the speedups.
- **Split-precision GEMMs** (D4 3xTF32, D8 bf16x6). They pass the fp64-relative tolerance, and
  D8 is 1.22x faster than cuBLAS fp32. Ruled out by policy (below), enforced statically.
- **Partial work below the roofline.** Within a batch the inputs repeat, so a cache keyed on
  the input values, used on some calls but not all, can win a speedup that stays under the
  physical limit. Full caching crosses it (impossible_speed). Y6 (hand audit of passing
  kernels) is the check for the rest.
- **Hidden cases are deterministic** (`HIDDEN_SEED`). The agent never sees the repo; rotate the
  seed for published results if that changes.

## Decision: low precision is out

Strict fp32 (Y2 decision 1) now also excludes emulating fp32 with fp16, bf16 or fp8 pieces,
even when every `tl.dot` says `input_precision="ieee"`. Any fp16/bf16/fp8 dtype or cast in a
candidate is `low_precision` (static). Reason: the benchmark compares fp32 kernels; a split
GEMM is a different numerical contract, even if it fits the tolerance today.

## Cost

Each worker start imports torch and initialises CUDA, about 3 s. An eval uses one worker when
the task's baselines are cached, or four (check, eager, compile, timing) the first time.

## Review of the first isolated design (2026-11-02)

A review of the first version found: timed inputs never changed and only the final output was
checked; error messages raised after the hidden cases, custom exception names, exit codes and
error sizes could carry hidden sizes; files and processes could outlive a worker; bare
`exec`/`getattr` and `tl.core` passed the scan; and the FLOP floor would flag honest kernels
on L1-56. All are fixed above, with tests in `tests/test_isolation.py`,
`tests/test_benchmark.py` and `tests/test_static_check.py`.
