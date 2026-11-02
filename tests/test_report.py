"""B19 acceptance: the page renders from a fresh DB in one command."""

from datetime import UTC, datetime

from triton_kernel_agent.db import Attempt, ResultsDB, Run
from triton_kernel_agent.harness_result import CheatFlag, HarnessResult, ShapeClass, Status, Timing
from triton_kernel_agent.report import main, render
from triton_kernel_agent.tasks import load_tasks


def test_renders_from_a_fresh_db_in_one_command(tmp_path):
    out = main(["--db", str(tmp_path / "fresh.duckdb"), "--out", str(tmp_path / "site/index.html")])
    page = out.read_text()
    assert page.startswith("<!doctype html>") and "No runs yet" in page
    assert "<script" not in page and "http" not in page.split("<body>")[1]  # self-contained


def fill(db):
    db.upsert_tasks(load_tasks(include_canaries=True))
    for i, setup in enumerate(("wonder_spark", "spark_only")):
        run_id = f"run-{setup}"
        db.insert_run(Run(run_id, {"setup": setup}, datetime(2026, 10, 29, i, tzinfo=UTC), "abc"))
        results = {
            "L1-19": HarnessResult("L1-19", Status.CORRECT, max_err=0.0,
                                   timing=Timing(kernel_ms=4.0, eager_ms=4.3, compile_ms=4.3)),
            "L1-29": HarnessResult("L1-29", Status.MISMATCH, max_err=1.0,
                                   failing_shape_class=ShapeClass.NON_POW2),
            "L2-95": HarnessResult("L2-95", Status.RUNTIME_CHEAT,
                                   cheat_flags=[CheatFlag.ATEN_COMPUTE]),
        }  # fmt: skip
        for task, result in results.items():
            aid = f"{run_id}:{task}:1"
            attempt = Attempt(aid, run_id, task, 1, setup, "m", "high", 1, 0, 1, 0.5, "h", "<c>")
            db.record_attempt(attempt, result)


def test_renders_runs_leaderboard_tasks_and_cheats():
    with ResultsDB() as db:
        fill(db)
        page = render(db, gpu_usd_per_hour=3.95)
    for text in ("Held-out", "Dev", "wonder_spark", "spark_only", "Best kernel per task",
                 "Correct within k", "<svg", "aten_compute", "1.07×"):  # fmt: skip
        assert text in page, text
    assert page.count("<polyline") == 2  # one correct-within-k curve per run


def test_html_in_data_is_escaped():
    with ResultsDB() as db:
        db.upsert_tasks(load_tasks())
        db.insert_run(Run("<b>run</b>", {"setup": "<script>x</script>"}, datetime.now(UTC), "a"))
        page = render(db, gpu_usd_per_hour=3.95)
    assert "<script>x</script>" not in page and "&lt;script&gt;" in page
