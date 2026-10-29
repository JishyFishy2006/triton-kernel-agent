"""B9 logic on CPU with a stand-in timer. Real timing is tests/test_live_benchmark.py."""

import pytest

from triton_kernel_agent.benchmark import benchmark
from triton_kernel_agent.harness_result import Status

torch = pytest.importorskip("torch")
nn = torch.nn


class Double(nn.Module):
    def forward(self, x):
        return x * 2


class CountingCandidate(nn.Module):
    """Correct for the first `honest_calls` calls, then returns garbage (behaves while checked)."""

    def __init__(self, honest_calls=10**9):
        super().__init__()
        self.calls, self.honest_calls = 0, honest_calls

    def forward(self, x):
        self.calls += 1
        return x * 2 if self.calls <= self.honest_calls else torch.zeros_like(x)


def fake_timer(times):
    def timer(fn):
        for _ in range(5):  # a real timer calls fn many times
            fn()
        return times.pop(0)

    return timer


def run(cand, times=None, compile_baseline=False):
    times = times or [[2.0, 2.2, 2.1], [1.0, 0.9, 1.1]]
    torch.manual_seed(0)
    x = torch.randn(64, 64)
    return benchmark(Double(), cand, [x], 1e-4, 1e-4, compile_baseline, timer=fake_timer(times))


def test_timing_is_the_median_of_repeats_and_speedups_follow():
    result = run(CountingCandidate())
    assert result.consistent
    # compile_baseline=False: compile_ms falls back to eager.
    assert (result.timing.eager_ms, result.timing.kernel_ms) == (2.1, 1.0)
    assert result.timing.speedup_eager == pytest.approx(2.1)
    assert result.extra["kernel_runs_ms"] == [1.0, 0.9, 1.1]


def test_a_candidate_that_changes_after_the_checks_is_caught():
    result = run(CountingCandidate(honest_calls=3))
    assert not result.consistent
    assert result.post_check.status == Status.MISMATCH


def test_compile_baseline_is_timed_separately():
    times = [[2.0, 2.0, 2.0], [1.5, 1.5, 1.5], [1.0, 1.0, 1.0]]
    result = run(CountingCandidate(), times=times, compile_baseline=True)
    assert result.timing.compile_ms == 1.5
    assert result.timing.speedup_compile == pytest.approx(1.5)
    assert result.extra["compile_s"] > 0


def test_cached_baselines_skip_eager_and_compile_timing():
    torch.manual_seed(0)
    x = torch.randn(64, 64)
    cached = {"eager_runs_ms": [3.0, 3.0, 3.0], "compile_runs_ms": [2.0, 2.0, 2.0]}
    result = benchmark(
        Double(), CountingCandidate(), [x], 1e-4, 1e-4, timer=fake_timer([[1.0, 1.0, 1.0]]),
        baselines=cached,
    )  # fmt: skip
    assert (result.timing.eager_ms, result.timing.compile_ms, result.timing.kernel_ms) == (3, 2, 1)
    assert result.extra["baselines_cached"] and result.extra["compile_runs_ms"] == [2.0] * 3
