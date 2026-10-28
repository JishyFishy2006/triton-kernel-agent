"""Benchmark module (B9): time the candidate, eager PyTorch and torch.compile the same way.

- triton.testing.do_bench: warm-up, L2 flush between runs, median of many runs.
- The timed function ends with torch.cuda.synchronize(), a device-wide wait. Work a candidate
  pushes onto a side stream (L4 #2) is then inside the timed region; without it the timer on the
  current stream stops before that work runs. Baselines get the same wrapper, so the small sync
  cost is in every number and the speedups stay fair.
- REPEATS independent do_bench runs; the reported time is their median.
- Everything runs in strict fp32 (Y2 decision 1), and baselines are the reference model itself.
- After timing, the candidate runs once more and its output is checked again: a kernel that is
  correct only while being checked (e.g. counting calls) is changed_after_checks.

Runs where torch (and, for the real timer, triton and CUDA) is installed.
"""

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from statistics import median
from typing import Any

from triton_kernel_agent.correctness import (
    CaseCheck,
    _clone,
    _leaves,
    _reference_fp64,
    compare,
    poison_free_memory,
    strict_fp32,
)
from triton_kernel_agent.harness_result import Timing

REPEATS = 3
WARMUP_MS = 25
REP_MS = 100


def bench(fn: Callable[[], Any], repeats: int = REPEATS) -> list[float]:
    """Median ms of `fn` per do_bench run, `repeats` runs. Ends each call with a device sync."""
    import torch
    from triton.testing import do_bench

    def synced() -> None:
        fn()
        torch.cuda.synchronize()

    return [
        do_bench(synced, warmup=WARMUP_MS, rep=REP_MS, return_mode="median") for _ in range(repeats)
    ]


@dataclass(frozen=True)
class BenchResult:
    timing: Timing
    post_check: CaseCheck  # the candidate's output after timing, against the reference
    extra: dict[str, Any] = field(default_factory=dict)  # every run, compile time

    @property
    def consistent(self) -> bool:
        return self.post_check.passed


def benchmark(
    ref_model,
    cand_model,
    inputs: Sequence[Any],
    atol: float,
    rtol: float,
    compile_baseline: bool = True,
    timer: Callable[[Callable[[], Any]], list[float]] = bench,
) -> BenchResult:
    """Time candidate, eager and torch.compile on the same inputs; then re-check the candidate.

    The candidate has already passed correctness (B5) and the runtime checks (B8).
    `timer` is swappable for tests without a GPU.
    """
    import torch

    with strict_fp32(), torch.no_grad():
        ref32 = [r.clone() for r in _leaves(ref_model(*_clone(inputs)))]
    ref64 = _reference_fp64(ref_model, inputs)

    with strict_fp32(), torch.no_grad():
        eager_runs = timer(lambda: ref_model(*inputs))
        compile_runs, compile_s = eager_runs, 0.0
        if compile_baseline:
            compiled = torch.compile(ref_model)
            start = time.perf_counter()
            compiled(*inputs)
            compile_s = time.perf_counter() - start
            compile_runs = timer(lambda: compiled(*inputs))
        kernel_runs = timer(lambda: cand_model(*inputs))

        poison_free_memory()
        after = _leaves(cand_model(*inputs))
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    post_check = _recheck(after, ref32, ref64, atol, rtol)
    timing = Timing(
        kernel_ms=median(kernel_runs), eager_ms=median(eager_runs), compile_ms=median(compile_runs)
    )
    extra = {
        "kernel_runs_ms": kernel_runs,
        "eager_runs_ms": eager_runs,
        "compile_runs_ms": compile_runs if compile_baseline else None,
        "compile_s": compile_s,
        "repeats": len(kernel_runs),
    }
    return BenchResult(timing, post_check, extra)


def _recheck(after, ref32, ref64, atol, rtol) -> CaseCheck:
    from triton_kernel_agent.harness_result import Status

    if len(after) != len(ref32):
        msg = f"returned {len(after)} tensors after timing, {len(ref32)} before"
        return CaseCheck(Status.WRONG_SHAPE, float("nan"), float("nan"), None, "exact", msg)
    worst = None
    for i, (a, r32) in enumerate(zip(after, ref32, strict=True)):
        r64 = ref64[i] if ref64 is not None and r32.is_floating_point() else None
        check = compare(a, r32, r64, atol, rtol, hidden=False)
        if not check.passed:
            return check
        if worst is None or check.max_err > worst.max_err:
            worst = check
    return worst
