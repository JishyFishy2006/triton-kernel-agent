"""B9 timing on CPU with a stand-in worker: calibration, the median, and the physical-speed
check. Real workers: tests/test_isolation.py; on an H100: tests/test_live_benchmark.py."""

import pytest

from triton_kernel_agent import benchmark
from triton_kernel_agent.benchmark import TimingFailed, impossible_speed, time_worker
from triton_kernel_agent.roofline import H100_SXM


class FakeWorker:
    """Answers `time` commands; the parent's clock sees a fixed cost per call."""

    def __init__(self, ms_per_call, fail_after=None):
        self.ms, self.fail_after, self.calls = ms_per_call, fail_after, []

    def call(self, command, timeout_s=None):
        assert command["cmd"] == "time"
        self.calls.append(command["calls"])
        if self.fail_after is not None and len(self.calls) > self.fail_after:
            return {"ok": False, "error": "RuntimeError", "message": "boom"}
        benchmark.time.sleep(self.ms * command["calls"] / 1e3)
        return {"ok": True}


def test_batches_are_calibrated_to_the_batch_length(monkeypatch):
    monkeypatch.setattr(benchmark, "BATCH_MS", 40.0)
    monkeypatch.setattr(benchmark, "REPEATS", 3)
    worker = FakeWorker(ms_per_call=4.0)
    timing = time_worker(worker)
    assert worker.calls[:2] == [1, 1]  # first call (JIT / compile), then calibration
    assert 5 <= timing.calls <= 10 and len(worker.calls) == 5
    # Measured batches make a random number of calls, so the checked last call is unknown.
    assert all(timing.calls <= n <= 1.5 * timing.calls for n in worker.calls[2:])
    assert timing.median_ms == pytest.approx(4.0, rel=0.5)
    assert timing.first_call_s == pytest.approx(0.004, rel=1.0)


def test_after_batch_runs_after_each_measured_batch(monkeypatch):
    monkeypatch.setattr(benchmark, "BATCH_MS", 5.0)
    monkeypatch.setattr(benchmark, "REPEATS", 3)
    worker, seen = FakeWorker(1.0), []
    time_worker(worker, after_batch=lambda i: seen.append((i, len(worker.calls))))
    assert seen == [(0, 3), (1, 4), (2, 5)]  # not after the first call or the calibration


def test_a_worker_that_raises_while_timed_fails_the_timing(monkeypatch):
    monkeypatch.setattr(benchmark, "BATCH_MS", 5.0)
    with pytest.raises(TimingFailed):
        time_worker(FakeWorker(1.0, fail_after=2))


def test_faster_than_moving_the_bytes_is_impossible():
    # L1-1: 201 MB at 3.35 TB/s takes 0.060 ms.
    nbytes = 201_326_592
    assert impossible_speed(nbytes, 2.7, H100_SXM) is None  # cuBLAS fp32
    assert impossible_speed(nbytes, 0.058, H100_SXM) is None  # 103% of peak: rounding
    assert "faster than the H100 SXM" in impossible_speed(nbytes, 0.019, H100_SXM)  # memoized
    # Memory-bound ReLU: 4096 x 393216 floats in and out.
    relu_bytes = 2 * 4096 * 393216 * 4
    assert impossible_speed(relu_bytes, 4.3, H100_SXM) is None
    assert impossible_speed(relu_bytes, 0.5, H100_SXM) is not None


def test_flops_give_no_floor():
    # L1-56: eager (cuDNN) beats the FLOP count's floor of 8.70 ms; only bytes count.
    assert impossible_speed(790_000_000, 3.56, H100_SXM) is None
