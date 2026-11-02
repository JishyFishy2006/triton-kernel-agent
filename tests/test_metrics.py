"""B18 acceptance: unit tests on a small synthetic DB give hand-computed values."""

import math
from datetime import UTC, datetime

import pytest

from triton_kernel_agent.db import Attempt, ResultsDB, Run
from triton_kernel_agent.harness_result import CheatFlag, HarnessResult, ShapeClass, Status, Timing
from triton_kernel_agent.metrics import compute_metrics
from triton_kernel_agent.tasks import load_tasks

RUN = "run-m"
PRICE = 3.6  # $/GPU-hour, so 1 GPU-second costs $0.001


def correct(task, eager, compile_, gpu_s):
    # kernel_ms = 1, so speedup over eager = eager_ms, over compile = compile_ms.
    timing = Timing(kernel_ms=1.0, eager_ms=eager, compile_ms=compile_)
    return HarnessResult(task, Status.CORRECT, max_err=0.0, timing=timing, gpu_seconds=gpu_s)


def failed(task, status, gpu_s, flags=()):
    kwargs = {"failing_shape_class": ShapeClass.NON_POW2} if status == Status.MISMATCH else {}
    return HarnessResult(task, status, cheat_flags=flags, gpu_seconds=gpu_s, **kwargs)


# task -> [(cost_usd, result)] in attempt order
SCRIPT = {
    "L1-19": [(0.10, failed("L1-19", Status.MISMATCH, 10)), (0.20, correct("L1-19", 1.5, 1.2, 20))],
    "L1-1": [(0.30, correct("L1-1", 0.8, 0.7, 30)), (0.30, correct("L1-1", 2.5, 2.0, 30))],
    "L1-12": [(0.05, correct("L1-12", 3.0, 3.0, 5))],  # in_fast_p = false (Y2)
    "L2-76": [
        (0.10, failed("L2-76", Status.STATIC_CHEAT, 0, [CheatFlag.TORCH_COMPUTE_CALL])),
        (0.10, failed("L2-76", Status.RUNTIME_CHEAT, 10,
                      [CheatFlag.ATEN_COMPUTE, CheatFlag.OUTPUT_ALIASES_INPUT])),
        (0.10, failed("L2-76", Status.MISMATCH, 10)),
    ],
    "L2-23": [(9.99, correct("L2-23", 9.0, 9.0, 999))],  # canary: never counted
}  # fmt: skip


@pytest.fixture(scope="module")
def db():
    with ResultsDB() as db:
        db.upsert_tasks(load_tasks(include_canaries=True))
        db.insert_run(Run(RUN, {}, datetime(2026, 10, 29, tzinfo=UTC), "abc"))
        for task, attempts in SCRIPT.items():
            for n, (cost, result) in enumerate(attempts, start=1):
                aid = f"{RUN}:{task}:{n}"
                db.record_attempt(
                    Attempt(
                        aid, RUN, task, n, "wonder_spark", "m", "high", 1, 0, 1, cost, "h", "c"
                    ),
                    result,
                )
        yield db


def test_all_scored_tasks(db):
    m = compute_metrics(db, RUN, gpu_usd_per_hour=PRICE)
    assert (m.tasks, m.solved) == (4, 3)  # L1-19, L1-1, L1-12 solved; L2-76 not; no canary
    # fast_p over the 3 tasks counted (L1-12 excluded): best speedups 1.5 and 2.5.
    assert m.fast_p == pytest.approx({0.0: 2 / 3, 1.0: 2 / 3, 2.0: 1 / 3})
    # vs torch.compile: best attempts' 1.2 and 2.0 (2.0 is not > 2).
    assert m.fast_p_compile == pytest.approx({0.0: 2 / 3, 1.0: 2 / 3, 2.0: 0.0})
    # first correct: L1-1 and L1-12 at 1, L1-19 at 2, of 4 tasks.
    assert m.correct_within[1] == 0.5
    assert m.correct_within[2] == m.correct_within[10] == 0.75
    assert m.median_speedup_eager == pytest.approx(2.0)
    assert m.geomean_speedup_eager == pytest.approx(math.sqrt(1.5 * 2.5))
    assert m.median_speedup_compile == pytest.approx(1.6)
    assert m.geomean_speedup_compile == pytest.approx(math.sqrt(1.2 * 2.0))
    assert m.api_usd == pytest.approx(1.25)
    assert m.gpu_usd == pytest.approx(0.115)  # 115 GPU-seconds
    assert m.cost_per_correct == pytest.approx((1.25 + 0.115) / 3)
    assert m.cheats == {"torch_compute_call": 1, "aten_compute": 1, "output_aliases_input": 1}


def test_per_level(db):
    l2 = compute_metrics(db, RUN, level=2, gpu_usd_per_hour=PRICE)
    assert (l2.tasks, l2.solved) == (1, 0)
    assert l2.fast_p[0.0] == 0.0 and l2.cost_per_correct is None
    assert l2.median_speedup_eager is None and l2.geomean_speedup_eager is None
    l1 = compute_metrics(db, RUN, level=1, gpu_usd_per_hour=PRICE)
    assert l1.fast_p[1.0] == 1.0  # both counted L1 tasks are correct and faster


def test_per_split(db):
    held_out = compute_metrics(db, RUN, split="held_out", gpu_usd_per_hour=PRICE)
    assert held_out.tasks == 1 and held_out.fast_p == {0.0: 0.0, 1.0: 0.0, 2.0: 0.0}
    dev = compute_metrics(db, RUN, split="dev", gpu_usd_per_hour=PRICE)
    assert dev.tasks == 3 and dev.correct_within[1] == pytest.approx(1 / 3)


def test_per_task_rows(db):
    rows = {r.task_id: r for r in compute_metrics(db, RUN, gpu_usd_per_hour=PRICE).per_task}
    assert rows["L1-1"].best_speedup_eager == 2.5 and rows["L1-1"].first_correct == 1
    assert rows["L2-76"].attempts == 3 and rows["L2-76"].best_speedup_eager is None
    assert "L2-23" not in rows


def test_empty_run(db):
    m = compute_metrics(db, "no-such-run", gpu_usd_per_hour=PRICE)
    assert m.tasks == 0 and m.cost_per_correct is None and m.fast_p[1.0] == 0.0
