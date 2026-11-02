"""B12 acceptance: insert then read back gives identical rows."""

import math
from datetime import UTC, datetime, timedelta, timezone

import pytest

from triton_kernel_agent.db import Attempt, ResultsDB, Run
from triton_kernel_agent.harness_result import HarnessResult, Status, Timing
from triton_kernel_agent.tasks import load_tasks

RUN = Run(
    run_id="run-1",
    config={"setup": "wonder_spark", "budget": {"max_attempts": 10}},
    started_at=datetime(2026, 10, 28, 1, 2, 3, 456789, tzinfo=UTC),
    git_sha="abc1234",
)


def attempt(n=1, task_id="L2-76", **kwargs):
    fields = dict(
        attempt_id=f"run-1:{task_id}:{n}",
        run_id="run-1",
        task_id=task_id,
        n=n,
        setup="wonder_spark",
        model="muse-wonder-1.0",
        effort="high",
        input_tokens=1200,
        cached_input_tokens=800,
        output_tokens=6400,
        cost_usd=0.1324,
        code_hash="sha256:feed",
        code="import triton\n# kernel\n",
        plan_summary="fuse bias + relu into the GEMM epilogue",
    )
    return Attempt(**{**fields, **kwargs})


RESULTS = [
    HarnessResult(
        task_id="L2-76",
        status=Status.CORRECT,
        max_err=3.1e-5,
        timing=Timing(kernel_ms=2.41, eager_ms=2.77, compile_ms=2.74),
        pct_peak_bandwidth=12.4,
        pct_peak_flops=85.3,
        gpu_seconds=18.2,
        extra={"post_check": {"rule": "fp64_relative"}},
    ),
    HarnessResult(
        task_id="L2-76",
        status="mismatch",
        max_err=0.0412,
        failing_shape_class="non_pow2",
        failing_shape=[[667, 1255]],
        error_excerpt="max abs error 0.0412 vs fp64",
        gpu_seconds=9.7,
    ),
    HarnessResult(task_id="L2-76", status="nan_inf", max_err=math.inf, failing_shape_class="large"),
    HarnessResult(
        task_id="L2-76",
        status="runtime_cheat",
        cheat_flags=["output_aliases_input", "aten_compute"],
    ),
    HarnessResult(task_id="L2-76", status="compile_error", error_excerpt="SyntaxError: ..."),
]


@pytest.fixture
def db(tmp_path):
    with ResultsDB(tmp_path / "results.duckdb") as db:
        db.upsert_tasks(load_tasks(include_canaries=True))
        db.insert_run(RUN)
        yield db


def test_run_round_trips(db):
    assert db.get_run("run-1") == RUN
    assert db.get_run("missing") is None


def test_run_times_are_stored_as_utc(db):
    eastern = timezone(timedelta(hours=-4))
    db.insert_run(Run("run-2", {}, datetime(2026, 10, 27, 21, 0, tzinfo=eastern), "def"))
    assert db.get_run("run-2").started_at == datetime(2026, 10, 28, 1, 0, tzinfo=UTC)


def test_attempt_round_trips(db):
    db.insert_attempt(attempt())
    assert db.get_attempt("run-1:L2-76:1") == attempt()


@pytest.mark.parametrize("result", RESULTS, ids=lambda r: str(r.status))
def test_eval_round_trips(db, result):
    db.insert_attempt(attempt())
    db.insert_eval("run-1:L2-76:1", result)
    assert db.get_eval("run-1:L2-76:1") == result


def test_speedups_are_columns_for_sql(db):
    db.insert_attempt(attempt())
    db.insert_eval("run-1:L2-76:1", RESULTS[0])
    ((eager, compile_),) = db.sql("SELECT speedup_eager, speedup_compile FROM evals")
    assert eager == pytest.approx(2.77 / 2.41)
    assert compile_ == pytest.approx(2.74 / 2.41)


def test_tasks_table_has_the_manifest_and_metadata(db):
    rows = dict(db.sql("SELECT task_id, flops FROM tasks"))
    assert len(rows) == 45
    assert rows["L1-1"] == 2 * 4096**3
    ((split, role),) = db.sql("SELECT split, role FROM tasks WHERE task_id = 'L2-23'")
    assert (split, role) == (None, "canary")
    db.insert_attempt(attempt())
    db.upsert_tasks(load_tasks())  # every run re-upserts, even with attempts referencing tasks
    assert db.sql("SELECT count(*) FROM tasks") == [(45,)]


def test_an_attempt_cannot_be_recorded_twice(db):
    db.insert_attempt(attempt())
    with pytest.raises(Exception, match="(?i)constraint|duplicate"):
        db.insert_attempt(attempt())
    with pytest.raises(Exception, match="(?i)constraint|duplicate"):
        db.insert_attempt(attempt(attempt_id="other-id"))  # same (run, task, n)
    assert db.attempt_ids("run-1") == {"run-1:L2-76:1"}


def test_evals_need_a_recorded_attempt(db):
    with pytest.raises(Exception, match="(?i)foreign key|constraint"):
        db.insert_eval("never-attempted", RESULTS[0])


def test_data_survives_reopening(tmp_path):
    path = tmp_path / "sub" / "results.duckdb"
    with ResultsDB(path) as db:
        db.upsert_tasks(load_tasks())
        db.insert_run(RUN)
        db.insert_attempt(attempt())
        db.insert_eval("run-1:L2-76:1", RESULTS[1])
    with ResultsDB(path) as db:
        assert db.get_eval("run-1:L2-76:1") == RESULTS[1]
        assert db.get_attempt("run-1:L2-76:1") == attempt()
