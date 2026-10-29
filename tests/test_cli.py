"""B20: the CLI. The acceptance test (a screen recording of a full run with no manual steps) is
done by hand with `uv run tka demo <task>`; these tests run the same command on fakes."""

import dataclasses
import json

from typer.testing import CliRunner

from triton_kernel_agent import cli
from triton_kernel_agent.agent import ModelReply
from triton_kernel_agent.config import load_config
from triton_kernel_agent.db import ResultsDB
from triton_kernel_agent.harness_result import HarnessResult, ShapeClass, Status, Timing
from triton_kernel_agent.prompts import KernelResponse

runner = CliRunner()


def test_help_lists_the_commands():
    result = runner.invoke(cli.app, ["--help"])
    assert result.exit_code == 0
    for command in ("demo", "run", "report"):
        assert command in result.output


def test_unknown_setup_is_rejected():
    result = runner.invoke(cli.app, ["demo", "L1-19", "--setup", "gpt"])
    assert result.exit_code != 0


class FakeWriter:
    def __init__(self):
        self.n = 0

    def write(self, role, messages, tags):
        self.n += 1
        return ModelReply(KernelResponse(f"plan {self.n}", f"code {self.n}"), "muse-wonder-1.0",
                          "high", 0.12, 900, 0, 5000)  # fmt: skip


class FakeSandbox:
    """Attempt 1 fails a hidden case, attempt 2 is correct and 1.3x faster."""

    calls = 0

    @classmethod
    def from_config(cls):
        return cls()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass

    def evaluate(self, task, code):
        FakeSandbox.calls += 1
        if code == "code 1":
            return HarnessResult(task.id, Status.MISMATCH, max_err=0.3,
                                 failing_shape_class=ShapeClass.NON_POW2)  # fmt: skip
        return HarnessResult(task.id, Status.CORRECT, max_err=0.0,
                             timing=Timing(kernel_ms=3.3, eager_ms=4.3, compile_ms=4.3),
                             pct_peak_bandwidth=88.0)  # fmt: skip


def test_demo_runs_a_task_end_to_end_on_fakes(tmp_path, monkeypatch):
    cfg = load_config()
    cfg = dataclasses.replace(cfg, paths=dataclasses.replace(cfg.paths, db=tmp_path / "r.duckdb"))
    monkeypatch.setattr(cli, "load_config", lambda: cfg)
    monkeypatch.setattr(cli, "_writer", lambda cfg, setup: FakeWriter())
    monkeypatch.setattr("triton_kernel_agent.sandbox.SandboxRunner", FakeSandbox)

    result = runner.invoke(cli.app, ["demo", "L1-19", "--setup", "wonder_spark"])

    assert result.exit_code == 0, result.output
    assert "Best: attempt 2" in result.output and "1.30× eager" in result.output
    assert "faster_than_eager" in result.output
    with ResultsDB(tmp_path / "r.duckdb") as db:
        rows = db.sql("SELECT n, setup, role FROM attempts ORDER BY n")
        (config,) = db.sql("SELECT config FROM runs")[0]
    assert rows == [(1, "wonder_spark", "planner"), (2, "wonder_spark", "planner")]
    assert json.loads(config)["demo"] == "L1-19"


def test_live_view_renders_attempts():
    from rich.console import Console

    from triton_kernel_agent.agent import AttemptRecord
    from triton_kernel_agent.tasks import get_task

    view = cli.LiveView(get_task("L1-19"), "wonder_spark")
    reply = FakeWriter().write("planner", [], {})
    ok = FakeSandbox().evaluate(get_task("L1-19"), "code 2")
    view.rows.append(AttemptRecord(1, "planner", reply, ok, "fb"))
    view.set_phase("evaluating")
    console = Console(record=True, width=160)
    console.print(view.render())
    text = console.export_text()
    assert "L1-19" in text and "correct" in text and "1.30×" in text and "88%" in text
    assert "evaluating" in text
