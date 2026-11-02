"""Results database (B12): tasks, runs, attempts and evals in one DuckDB file.

Tables follow the tech spec's data model. Primary keys make a re-inserted attempt an error,
which is what lets the batch runner (B17) resume without duplicating work. evals holds one
HarnessResult per attempt, with speedups stored as columns for SQL (B18).
"""

import json
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from triton_kernel_agent.harness_result import HarnessResult, Timing
from triton_kernel_agent.tasks import Task

SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    task_id VARCHAR PRIMARY KEY,
    level INTEGER NOT NULL,
    op_name VARCHAR NOT NULL,
    task_group VARCHAR NOT NULL,
    split VARCHAR,                 -- dev / held_out; NULL for canaries
    role VARCHAR NOT NULL,         -- scored / canary
    in_fast_p BOOLEAN NOT NULL,
    dtype VARCHAR NOT NULL,
    bytes BIGINT,
    flops BIGINT
);
CREATE TABLE IF NOT EXISTS runs (
    run_id VARCHAR PRIMARY KEY,
    config VARCHAR NOT NULL,       -- JSON
    started_at TIMESTAMP NOT NULL, -- UTC
    git_sha VARCHAR NOT NULL
);
CREATE TABLE IF NOT EXISTS attempts (
    attempt_id VARCHAR PRIMARY KEY,
    run_id VARCHAR NOT NULL REFERENCES runs (run_id),
    task_id VARCHAR NOT NULL REFERENCES tasks (task_id),
    n INTEGER NOT NULL,            -- 1-based attempt number within (run, task)
    setup VARCHAR NOT NULL,        -- spark_only / wonder_only / wonder_spark (B16)
    model VARCHAR NOT NULL,
    effort VARCHAR NOT NULL,
    input_tokens INTEGER NOT NULL,
    cached_input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    cost_usd DOUBLE NOT NULL,
    code_hash VARCHAR NOT NULL,
    code VARCHAR NOT NULL,         -- kept for audits (Y6) and the report (B19)
    plan_summary VARCHAR NOT NULL,
    role VARCHAR NOT NULL,         -- planner / fixer (the agent loop resumes from it)
    UNIQUE (run_id, task_id, n)
);
CREATE TABLE IF NOT EXISTS evals (
    attempt_id VARCHAR PRIMARY KEY REFERENCES attempts (attempt_id),
    task_id VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    error_excerpt VARCHAR NOT NULL,
    max_err DOUBLE,
    failing_shape_class VARCHAR,
    failing_shape VARCHAR,         -- JSON; never shown to the agent
    cheat_flags VARCHAR[] NOT NULL,
    kernel_ms DOUBLE,
    eager_ms DOUBLE,
    compile_ms DOUBLE,
    speedup_eager DOUBLE,
    speedup_compile DOUBLE,
    pct_peak_bandwidth DOUBLE,
    pct_peak_flops DOUBLE,
    gpu_seconds DOUBLE NOT NULL,
    harness_version VARCHAR NOT NULL,
    extra VARCHAR NOT NULL         -- JSON
);
"""


@dataclass(frozen=True)
class Run:
    run_id: str
    config: dict[str, Any]
    started_at: datetime  # timezone-aware; stored as UTC
    git_sha: str


@dataclass(frozen=True)
class Attempt:
    attempt_id: str
    run_id: str
    task_id: str
    n: int
    setup: str
    model: str
    effort: str
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    cost_usd: float
    code_hash: str
    code: str
    plan_summary: str = ""
    role: str = "planner"


@dataclass
class ResultsDB:
    path: Path | str = ":memory:"
    _con: Any = field(init=False, repr=False)
    # One DuckDB connection is not thread-safe; the batch runner (B17) shares one, so every
    # call takes this lock.
    _lock: Any = field(init=False, repr=False, default_factory=threading.RLock)

    def __post_init__(self) -> None:
        import duckdb

        if str(self.path) != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._con = duckdb.connect(str(self.path))
        self._con.execute(SCHEMA)

    def close(self) -> None:
        self._con.close()

    def __enter__(self) -> "ResultsDB":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def sql(self, query: str, params: list[Any] | None = None) -> list[tuple]:
        """Run a query and fetch all rows (for metrics and ad-hoc analysis)."""
        with self._lock:
            return self._con.execute(query, params or []).fetchall()

    # --- writes ---

    def upsert_tasks(self, tasks: list[Task]) -> None:
        with self._lock:
            rows = [
                (
                    t.id,
                    t.level,
                    t.name,
                    t.group,
                    t.split,
                    t.role,
                    t.in_fast_p,
                    t.dtype,
                    t.bytes,
                    t.flops,
                )
                for t in tasks
            ]
            self._con.executemany(
                "INSERT OR REPLACE INTO tasks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows
            )

    def insert_run(self, run: Run) -> None:
        with self._lock:
            started = run.started_at.astimezone(UTC).replace(tzinfo=None)
            self._con.execute(
                "INSERT INTO runs VALUES (?, ?, ?, ?)",
                [run.run_id, json.dumps(run.config, sort_keys=True), started, run.git_sha],
            )

    def insert_attempt(self, attempt: Attempt) -> None:
        with self._lock:
            values = list(attempt.__dict__.values())
            self._con.execute(
                f"INSERT INTO attempts VALUES ({', '.join('?' * len(values))})", values
            )

    def insert_eval(self, attempt_id: str, result: HarnessResult) -> None:
        with self._lock:
            t = result.timing
            self._con.execute(
                f"INSERT INTO evals VALUES ({', '.join('?' * 18)})",
                [
                    attempt_id,
                    result.task_id,
                    str(result.status),
                    result.error_excerpt,
                    result.max_err,
                    str(result.failing_shape_class) if result.failing_shape_class else None,
                    json.dumps(result.failing_shape) if result.failing_shape is not None else None,
                    [str(f) for f in result.cheat_flags],
                    t.kernel_ms if t else None,
                    t.eager_ms if t else None,
                    t.compile_ms if t else None,
                    t.speedup_eager if t else None,
                    t.speedup_compile if t else None,
                    result.pct_peak_bandwidth,
                    result.pct_peak_flops,
                    result.gpu_seconds,
                    result.harness_version,
                    json.dumps(result.extra, sort_keys=True, default=str),
                ],
            )

        # --- reads ---

    def get_run(self, run_id: str) -> Run | None:
        row = self._one("SELECT * FROM runs WHERE run_id = ?", run_id)
        if row is None:
            return None
        return Run(row[0], json.loads(row[1]), row[2].replace(tzinfo=UTC), row[3])

    def get_attempt(self, attempt_id: str) -> Attempt | None:
        row = self._one("SELECT * FROM attempts WHERE attempt_id = ?", attempt_id)
        return Attempt(*row) if row else None

    def get_eval(self, attempt_id: str) -> HarnessResult | None:
        row = self._one("SELECT * FROM evals WHERE attempt_id = ?", attempt_id)
        if row is None:
            return None
        (_, task_id, status, excerpt, max_err, shape_class, shape, flags, kernel_ms,
         eager_ms, compile_ms, _, _, pct_bw, pct_flops, gpu_s, version, extra) = row  # fmt: skip
        timing = None
        if kernel_ms is not None:
            timing = Timing(kernel_ms=kernel_ms, eager_ms=eager_ms, compile_ms=compile_ms)
        return HarnessResult(
            task_id=task_id,
            status=status,
            error_excerpt=excerpt,
            max_err=max_err,
            failing_shape_class=shape_class,
            failing_shape=json.loads(shape) if shape is not None else None,
            cheat_flags=tuple(flags),
            timing=timing,
            pct_peak_bandwidth=pct_bw,
            pct_peak_flops=pct_flops,
            gpu_seconds=gpu_s,
            harness_version=version,
            extra=json.loads(extra),
        )

    def record_attempt(self, attempt: Attempt, result: HarnessResult) -> None:
        """Insert an attempt and its eval together: a kill never leaves half a record."""
        with self._lock:
            self._con.execute("BEGIN TRANSACTION")
            try:
                self.insert_attempt(attempt)
                self.insert_eval(attempt.attempt_id, result)
                self._con.execute("COMMIT")
            except BaseException:
                self._con.execute("ROLLBACK")
                raise

    def task_attempts(self, run_id: str, task_id: str) -> list[tuple[Attempt, HarnessResult]]:
        """A task's recorded attempts in order, each with its eval (for resuming)."""
        rows = self.sql(
            "SELECT attempt_id FROM attempts WHERE run_id = ? AND task_id = ? ORDER BY n",
            [run_id, task_id],
        )
        return [(self.get_attempt(aid), self.get_eval(aid)) for (aid,) in rows]

    def attempt_ids(self, run_id: str) -> set[str]:
        """Attempts already recorded for a run (B17 skips these when resuming)."""
        return {
            r[0] for r in self.sql("SELECT attempt_id FROM attempts WHERE run_id = ?", [run_id])
        }

    def _one(self, query: str, key: str) -> tuple | None:
        with self._lock:
            return self._con.execute(query, [key]).fetchone()
