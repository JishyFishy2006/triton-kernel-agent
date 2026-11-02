"""Command line (B20): `tka demo` runs one task end to end with a live view of each attempt;
`tka run` runs or resumes a batch (B17); `tka report` builds the results page (B19).

    uv run tka demo L1-19
    uv run tka run --split dev --workers 4
    uv run tka report
"""

import subprocess
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime

import typer
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from triton_kernel_agent.agent import AttemptRecord, Budget, ModelReply, Role, run_task
from triton_kernel_agent.config import MODEL_SETUPS, Config, load_config
from triton_kernel_agent.harness_result import HarnessResult, Status
from triton_kernel_agent.tasks import Task

app = typer.Typer(add_completion=False, help="Triton Kernel Agent")
console = Console()

STATUS_STYLE = {
    Status.CORRECT: "bold green",
    Status.STATIC_CHEAT: "bold magenta",
    Status.RUNTIME_CHEAT: "bold magenta",
    Status.HARNESS_ERROR: "yellow",
}


def _git_sha() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                             text=True, check=True)  # fmt: skip
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _shapes() -> dict[str, tuple[list, list]]:
    import json

    from triton_kernel_agent.tasks import METADATA_PATH

    meta = json.loads(METADATA_PATH.read_text())["tasks"]
    return {k: (v["input_shapes"], v["output_shape"]) for k, v in meta.items()}


def _budget(cfg: Config) -> Budget:
    return Budget(cfg.budget.max_attempts, cfg.budget.max_usd_per_task)


def _writer(cfg: Config, setup: str):
    from triton_kernel_agent.env import load_secrets
    from triton_kernel_agent.llm import ModelClient
    from triton_kernel_agent.setups import ModelWriter

    return ModelWriter(ModelClient.from_config(cfg, load_secrets()), setup, cfg.models)


def _check_setup(setup: str) -> None:
    if setup not in MODEL_SETUPS:
        raise typer.BadParameter(f"setup must be one of {', '.join(MODEL_SETUPS)}")


# --- live view of one task ---


@dataclass
class LiveView:
    """What the demo shows: one row per attempt, plus what is happening right now."""

    task: Task
    setup: str
    rows: list[AttemptRecord] = field(default_factory=list)
    phase: str = "starting"
    phase_started: float = field(default_factory=time.monotonic)

    def set_phase(self, text: str) -> None:
        self.phase, self.phase_started = text, time.monotonic()

    def render(self) -> Group:
        table = Table(expand=True, header_style="bold")
        columns = [("#", 2), ("Role", 7), ("Model", 11), ("Result", 13), ("Kernel ms", 9),
                   ("vs eager", 8), ("vs compile", 10), ("% peak", 6), ("Cost", 6)]  # fmt: skip
        for col, width in columns:
            justify = "left" if col in ("Role", "Model", "Result") else "right"
            table.add_column(col, justify=justify, min_width=width, no_wrap=True)
        table.add_column("Plan", ratio=1, overflow="ellipsis", no_wrap=True)
        for a in self.rows:
            r, t = a.result, a.result.timing
            pct = r.pct_peak_flops if r.pct_peak_flops is not None else r.pct_peak_bandwidth
            table.add_row(
                str(a.n), a.role, a.reply.model.removeprefix("muse-"),
                Text(str(r.status) + (f" {','.join(map(str, r.cheat_flags))}" if r.cheat_flags
                                      else ""), style=STATUS_STYLE.get(r.status, "red")),
                f"{t.kernel_ms:.3f}" if t else "", f"{t.speedup_eager:.2f}×" if t else "",
                f"{t.speedup_compile:.2f}×" if t else "", f"{pct:.0f}%" if pct is not None else "",
                f"${a.reply.cost_usd:.3f}",
                a.reply.response.plan_summary if a.reply.response else "(reply broke the schema)",
            )  # fmt: skip
        spent = sum(a.reply.cost_usd for a in self.rows)
        elapsed = time.monotonic() - self.phase_started
        header = Text.assemble(
            (f"{self.task.id} ", "bold"), (self.task.name.replace("_", " "), ""),
            (f"   setup {self.setup}   spent ${spent:.2f}", "dim"),
        )  # fmt: skip
        status = Text(f"▶ {self.phase} ({elapsed:.0f} s)", style="cyan")
        return Group(Panel(header), table, status)


@dataclass
class _ViewWriter:
    writer: object
    view: LiveView

    def write(self, role: Role, messages, tags) -> ModelReply:
        self.view.set_phase(f"attempt {tags['attempt']}: {role} is writing a kernel")
        return self.writer.write(role, messages, tags)


@dataclass
class _ViewEvaluator:
    evaluator: object
    view: LiveView

    def evaluate(self, task: Task, code: str) -> HarnessResult:
        self.view.set_phase("evaluating on an H100: compile, checks, hidden shapes, timing")
        return self.evaluator.evaluate(task, code)


@app.command()
def demo(
    task_id: str = typer.Argument(..., help="Task to solve, e.g. L1-19"),
    setup: str = typer.Option(None, help="spark_only, wonder_only or wonder_spark"),
    max_attempts: int = typer.Option(None, help="Override [budget].max_attempts"),
) -> None:
    """Solve one task end to end, showing each attempt live."""
    from triton_kernel_agent.db import ResultsDB, Run
    from triton_kernel_agent.prompts import task_prompt
    from triton_kernel_agent.sandbox import SandboxRunner
    from triton_kernel_agent.setups import DBRecorder
    from triton_kernel_agent.tasks import get_task, load_tasks

    cfg = load_config()
    setup = setup or cfg.models.setup
    _check_setup(setup)
    task = get_task(task_id)
    budget = _budget(cfg)
    if max_attempts:
        budget = Budget(max_attempts, budget.max_usd)
    run_id = f"demo-{task.id}-{datetime.now(UTC):%Y%m%d-%H%M%S}"
    inputs, output = _shapes().get(task.id, (None, None))
    view = LiveView(task, setup)

    with ResultsDB(cfg.paths.db) as db, SandboxRunner.from_config() as sandbox:
        db.upsert_tasks(load_tasks(include_canaries=True))
        db.insert_run(Run(run_id, {"setup": setup, "demo": task.id}, datetime.now(UTC),
                          _git_sha()))  # fmt: skip
        recorder = DBRecorder(db, run_id, setup)

        def on_attempt(record: AttemptRecord) -> None:
            recorder(record)
            view.rows.append(record)

        with Live(view.render(), console=console, refresh_per_second=4,
                  get_renderable=view.render):  # fmt: skip
            outcome = run_task(
                task, task_prompt(task, inputs, output), _ViewWriter(_writer(cfg, setup), view),
                _ViewEvaluator(sandbox, view), budget, on_attempt=on_attempt,
            )  # fmt: skip
            view.set_phase(f"done: {outcome.stop_reason}")
    console.print()

    best = outcome.best
    if best:
        t = best.result.timing
        speeds = f"{t.speedup_eager:.2f}× eager, {t.speedup_compile:.2f}× torch.compile"
        console.print(f"[bold green]Best:[/] attempt {best.n}, {t.kernel_ms:.3f} ms, {speeds}")
    else:
        console.print("[bold red]No correct kernel.[/]")
    console.print(f"Stopped: {outcome.stop_reason}; spent ${outcome.cost_usd:.2f}; run {run_id}")


@app.command()
def run(
    run_id: str = typer.Option(None, help="Resume this run, or name a new one"),
    setup: str = typer.Option(None, help="spark_only, wonder_only or wonder_spark"),
    split: str = typer.Option(None, help="dev or held_out (default: both)"),
    tasks: str = typer.Option(None, help="Comma-separated task ids (default: all scored)"),
    workers: int = typer.Option(4, help="Tasks in parallel (one H100 sandbox each)"),
    per_minute: int = typer.Option(30, help="Model call starts per minute"),
) -> None:
    """Run (or resume) the agent on many tasks."""
    from triton_kernel_agent.batch import RateLimitedWriter, RateLimiter, run_batch
    from triton_kernel_agent.db import ResultsDB, Run
    from triton_kernel_agent.sandbox import SandboxRunner
    from triton_kernel_agent.tasks import get_task, load_tasks

    cfg = load_config()
    setup = setup or cfg.models.setup
    _check_setup(setup)
    chosen = [get_task(t) for t in tasks.split(",")] if tasks else load_tasks(split=split)
    run_id = run_id or f"{setup}-{datetime.now(UTC):%Y%m%d-%H%M%S}"
    writer = RateLimitedWriter(_writer(cfg, setup), RateLimiter(workers, per_minute))
    console.print(f"run [bold]{run_id}[/]: {len(chosen)} tasks, setup {setup}, {workers} workers")

    def show(a: AttemptRecord) -> None:
        r = a.result
        extra = f" {r.timing.speedup_eager:.2f}× eager" if r.timing else ""
        console.print(
            f"{r.task_id:>6} #{a.n:<2} {a.role:7} {r.status}{extra}  ${a.reply.cost_usd:.3f}"
        )

    started = datetime.now(UTC)
    with ResultsDB(cfg.paths.db) as db:
        run_obj = Run(run_id, {"setup": setup, "split": split, "tasks": tasks}, started, _git_sha())
        outcomes = run_batch(
            db, run_obj, setup, chosen, writer, SandboxRunner.from_config, _budget(cfg),
            workers=workers, shapes=_shapes(), on_attempt=show,
        )  # fmt: skip
    solved = sum(1 for o in outcomes.values() if o.best)
    spent = sum(o.cost_usd for o in outcomes.values())
    console.print(f"[bold]{solved}/{len(outcomes)}[/] solved; ${spent:.2f} on the API this session")


@app.command()
def report(out: str = typer.Option("site/index.html", help="Where to write the page")) -> None:
    """Build the static results page from the database."""
    from triton_kernel_agent.report import main as build

    build(["--out", out])


if __name__ == "__main__":
    app()
