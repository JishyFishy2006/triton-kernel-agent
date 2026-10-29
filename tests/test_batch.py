"""B17 acceptance: kill it mid-run, restart, and no attempt is duplicated. Plus resuming
finished runs, parallelism, and the rate limiter."""

import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime

import pytest

from triton_kernel_agent.agent import Budget, ModelReply
from triton_kernel_agent.batch import RateLimitedWriter, RateLimiter, run_batch
from triton_kernel_agent.db import ResultsDB, Run
from triton_kernel_agent.harness_result import HarnessResult, ShapeClass, Status, Timing
from triton_kernel_agent.prompts import KernelResponse
from triton_kernel_agent.tasks import get_task

TASKS = [get_task(t) for t in ("L1-19", "L1-1", "L2-76")]
RUN = Run("run-b17", {"setup": "wonder_spark"}, datetime(2026, 10, 29, tzinfo=UTC), "abc")
BUDGET = Budget(max_attempts=10, max_usd=3.0)


class Killed(BaseException):
    """Stands in for Ctrl-C / a crashed orchestrator."""


@dataclass
class FakeWriter:
    kill_after: int | None = None  # total calls before the "kill"
    calls: list[tuple[str, int]] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def write(self, role, messages, tags):
        with self.lock:
            if self.kill_after is not None and len(self.calls) >= self.kill_after:
                raise Killed()
            self.calls.append((tags["task_id"], tags["attempt"]))
        time.sleep(0.01)  # let threads interleave
        code = f"{tags['task_id']}:{tags['attempt']}"
        return ModelReply(KernelResponse("plan", code), "fake", "low", 0.01, 10, 0, 20)


class FakeHarness:
    """Each task: mismatch, mismatch, then correct and faster (stops at attempt 3)."""

    def evaluate(self, task, code):
        n = int(code.rsplit(":", 1)[1])
        if n < 3:
            return HarnessResult(task.id, Status.MISMATCH, max_err=1.0,
                                 failing_shape_class=ShapeClass.NON_POW2)  # fmt: skip
        timing = Timing(kernel_ms=1.0, eager_ms=2.0, compile_ms=2.0)
        return HarnessResult(task.id, Status.CORRECT, max_err=0.0, timing=timing)


def run(db, writer, workers=3):
    return run_batch(db, RUN, "wonder_spark", TASKS, writer, FakeHarness, BUDGET, workers=workers)


def attempts(db):
    return db.sql("SELECT task_id, n FROM attempts ORDER BY task_id, n")


def test_kill_mid_run_then_restart_duplicates_nothing(tmp_path):
    path = tmp_path / "results.duckdb"
    with ResultsDB(path) as db:
        with pytest.raises(Killed):
            run(db, FakeWriter(kill_after=7))
        before = attempts(db)
    assert 0 < len(before) < 9  # killed part-way through 3 tasks x 3 attempts

    with ResultsDB(path) as db:  # a fresh process would reopen the file like this
        writer = FakeWriter()
        outcomes = run(db, writer)
        after = attempts(db)
        evals = db.sql("SELECT count(*) FROM evals")[0][0]

    assert len(after) == len(set(after)) == 9 == evals  # every attempt once, each with its eval
    for task in TASKS:
        ns = [n for t, n in after if t == task.id]
        assert ns == [1, 2, 3], task.id  # contiguous: nothing skipped or repeated
        assert outcomes[task.id].stop_reason == "faster_than_eager"
        assert [a.n for a in outcomes[task.id].attempts] == [1, 2, 3]
    assert len(writer.calls) == 9 - len(before)  # the restart paid only for what was missing
    assert set(writer.calls).isdisjoint(set(before))


def test_rerunning_a_finished_run_makes_no_calls(tmp_path):
    with ResultsDB(tmp_path / "r.duckdb") as db:
        run(db, FakeWriter())
        writer = FakeWriter()
        outcomes = run(db, writer)
    assert writer.calls == []
    assert all(o.stop_reason == "faster_than_eager" for o in outcomes.values())


def test_tasks_run_in_parallel_with_one_evaluator_per_worker(tmp_path):
    made = []

    def make():
        made.append(threading.get_ident())
        return FakeHarness()

    with ResultsDB(tmp_path / "r.duckdb") as db:
        run_batch(db, RUN, "wonder_spark", TASKS, FakeWriter(), make, BUDGET, workers=3)
    assert 1 <= len(made) <= 3 and len(set(made)) == len(made)


def test_rate_limiter_spaces_out_call_starts():
    now = [0.0]
    limiter = RateLimiter(max_concurrent=10, per_minute=2, clock=lambda: now[0],
                          sleep=lambda s: now.__setitem__(0, now[0] + s))  # fmt: skip
    starts = []
    for _ in range(5):
        with limiter:
            starts.append(now[0])
    assert starts == [0.0, 0.0, 60.0, 60.0, 120.0]


def test_rate_limiter_caps_concurrent_calls():
    active, peak, lock = [0], [0], threading.Lock()

    class Slow:
        def write(self, role, messages, tags):
            with lock:
                active[0] += 1
                peak[0] = max(peak[0], active[0])
            time.sleep(0.02)
            with lock:
                active[0] -= 1

    writer = RateLimitedWriter(Slow(), RateLimiter(max_concurrent=2, per_minute=1000))
    threads = [threading.Thread(target=writer.write, args=("planner", [], {})) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert peak[0] == 2
