"""Benchmark (B9), timed with the trusted parent's own clock (Y4 isolation).

- Candidate, eager PyTorch and torch.compile each run in their own worker process
  (isolation.py). The parent asks a worker for `calls` forward calls back to back on the
  visible inputs and times the request with time.perf_counter; the worker ends each batch with
  a device-wide sync, so work on any stream is inside the time (L4 #2). Nothing the candidate
  runs can stop or slow the parent's clock (Y4 C03b), and the baselines run in workers that
  never see candidate code, while no candidate process exists (Y4 B2, B5, B7).
- calls is calibrated so one batch takes about BATCH_MS; REPEATS batches; the median counts.
  Back-to-back calls measure steady-state time per call, the same way for all three.
- Batches alternate between two visible-shape input sets, written into the same buffers
  between batches, and every batch's last output is checked against the candidate's own
  output for that set from the correctness stage. Batches make a random number of calls. A
  kernel that caches across calls, or works only while checked, is changed_after_checks
  (Y4 E3, D1-D9; review of the first isolated design).
- The time must be physically possible: an honest kernel reads its inputs and weights and
  writes its output, so it cannot beat moving the task's bytes at peak bandwidth. Faster means
  work was skipped: caching or lazy timed calls (impossible_speed; Y4 A4, B1, D1-D9). FLOPs
  give no such floor: eager L1-56 runs at 3.56 ms against a FLOP floor of 8.70 ms (cuDNN does
  fewer FLOPs than the count), and L2-14 collapses algebraically to a matrix-vector product.
"""

import math
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from statistics import median
from typing import Any

from triton_kernel_agent.roofline import GpuPeak

BATCH_MS = 200.0
REPEATS = 5
MAX_CALLS = 10_000
# Percent of peak bandwidth above which a time is impossible. Honest copies reach ~87% on an
# H100 (L9); 105 leaves room for spec-sheet rounding.
IMPOSSIBLE_PCT = 105.0
# One batch (BATCH_MS of calls, or one slow first call that compiles) must answer within this.
BATCH_TIMEOUT_S = 240.0


class TimingFailed(Exception):
    """The worker raised while being timed."""

    def __init__(self, answer: dict[str, Any]):
        super().__init__(answer.get("message", ""))
        self.answer = answer


@dataclass(frozen=True)
class WorkerTiming:
    runs_ms: list[float]  # per-call ms of each batch
    first_call_s: float  # the first call: Triton JIT, or torch.compile for the compile baseline
    calls: int  # calls per batch

    @property
    def median_ms(self) -> float:
        return median(self.runs_ms)


def _batch_ms(worker, calls: int) -> float:
    start = time.perf_counter()
    answer = worker.call({"cmd": "time", "calls": calls}, timeout_s=BATCH_TIMEOUT_S)
    elapsed = time.perf_counter() - start
    if not answer["ok"]:
        raise TimingFailed(answer)
    return 1e3 * elapsed


def time_worker(worker, after_batch: Callable[[int], None] | None = None) -> WorkerTiming:
    """Time a worker that has its model built and inputs loaded. Per-call ms per batch.

    after_batch(i) runs after measured batch i, outside the timed window: the harness checks
    the batch's last output there and writes new input values for the next batch. Each batch
    makes a random number of calls, so the last (checked) call cannot be predicted.
    """
    first_s = _batch_ms(worker, 1) / 1e3
    one = _batch_ms(worker, 1)
    calls = max(1, min(MAX_CALLS, math.ceil(BATCH_MS / max(one, 1e-3))))
    runs = []
    for i in range(REPEATS):
        n = calls + random.randrange(calls // 2 + 1)
        runs.append(_batch_ms(worker, n) / n)
        if after_batch is not None:
            after_batch(i)
    return WorkerTiming(runs, first_s, calls)


def impossible_speed(nbytes: int, kernel_ms: float, peak: GpuPeak) -> str | None:
    """A message if kernel_ms beats moving the task's bytes at peak bandwidth, else None."""
    floor_ms = 1e3 * nbytes / peak.bandwidth_bytes_per_s
    if floor_ms and kernel_ms * IMPOSSIBLE_PCT / 100 < floor_ms:
        return (
            f"{kernel_ms:.4f} ms per call is faster than the {peak.name} can move this task's "
            f"inputs, weights and outputs ({floor_ms:.4f} ms at peak bandwidth)"
        )
    return None
