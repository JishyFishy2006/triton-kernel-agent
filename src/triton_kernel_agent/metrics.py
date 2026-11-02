"""Metrics (B18): fast_p, correct within k attempts, speedup means, cost per correct kernel.

Computed from the results database for one run, over scored tasks (canaries never count),
optionally narrowed to a split (report on held_out) and a level (Y2: targets differ by level).
A task's result is its best correct attempt (highest speedup over eager).
- fast_p: share of tasks whose best kernel is correct and more than p times faster than eager
  (fast_0 = correct at all). Tasks with in_fast_p = false (L1-12) are left out (Y2).
- correct_within[k]: share of tasks with a correct kernel by attempt k.
- speedups: median and geometric mean of best speedups over eager and torch.compile, over
  correct tasks counted in fast_p.
- cost per correct kernel: (API dollars + GPU seconds x price) / tasks solved.
"""

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from statistics import median

from triton_kernel_agent.db import ResultsDB

FAST_P = (0.0, 1.0, 2.0)


@dataclass(frozen=True)
class TaskResult:
    task_id: str
    level: int
    in_fast_p: bool
    attempts: int
    first_correct: int | None  # attempt number of the first correct kernel
    best_speedup_eager: float | None
    best_speedup_compile: float | None
    api_usd: float
    gpu_seconds: float


@dataclass(frozen=True)
class Metrics:
    run_id: str
    tasks: int
    solved: int
    fast_p: dict[float, float]
    fast_p_compile: dict[float, float]
    correct_within: dict[int, float]
    median_speedup_eager: float | None
    geomean_speedup_eager: float | None
    median_speedup_compile: float | None
    geomean_speedup_compile: float | None
    api_usd: float
    gpu_usd: float
    cost_per_correct: float | None
    cheats: dict[str, int] = field(default_factory=dict)  # cheat flag -> attempts flagged
    per_task: list[TaskResult] = field(default_factory=list)


def _geomean(values: list[float]) -> float | None:
    return math.exp(sum(math.log(v) for v in values) / len(values)) if values else None


def task_results(
    db: ResultsDB, run_id: str, split: str | None = None, level: int | None = None
) -> list[TaskResult]:
    query = """
        SELECT t.task_id, t.level, t.in_fast_p, a.n, a.cost_usd, e.status,
               e.speedup_eager, e.speedup_compile, e.gpu_seconds
        FROM attempts a
        JOIN tasks t USING (task_id)
        JOIN evals e USING (attempt_id)
        WHERE a.run_id = ? AND t.role = 'scored'
    """
    params: list = [run_id]
    if split is not None:
        query += " AND t.split = ?"
        params.append(split)
    if level is not None:
        query += " AND t.level = ?"
        params.append(level)
    by_task: dict[str, list[tuple]] = defaultdict(list)
    for row in db.sql(query, params):
        by_task[row[0]].append(row)

    results = []
    for task_id, rows in sorted(by_task.items()):
        correct = [r for r in rows if r[5] == "correct"]
        best = max(correct, key=lambda r: r[6]) if correct else None
        results.append(
            TaskResult(
                task_id=task_id,
                level=rows[0][1],
                in_fast_p=rows[0][2],
                attempts=len(rows),
                first_correct=min(r[3] for r in correct) if correct else None,
                best_speedup_eager=best[6] if best else None,
                best_speedup_compile=best[7] if best else None,
                api_usd=sum(r[4] for r in rows),
                gpu_seconds=sum(r[8] for r in rows),
            )
        )
    return results


def compute_metrics(
    db: ResultsDB,
    run_id: str,
    *,
    split: str | None = None,
    level: int | None = None,
    gpu_usd_per_hour: float,
    max_k: int = 10,
) -> Metrics:
    results = task_results(db, run_id, split, level)
    n = len(results)
    counted = [r for r in results if r.in_fast_p]
    solved = [r for r in results if r.first_correct is not None]
    fast = [r for r in counted if r.best_speedup_eager is not None]

    def share(hits: int, total: int) -> float:
        return hits / total if total else 0.0

    api_usd = sum(r.api_usd for r in results)
    gpu_usd = sum(r.gpu_seconds for r in results) * gpu_usd_per_hour / 3600
    eager = [r.best_speedup_eager for r in fast]
    compile_ = [r.best_speedup_compile for r in fast]

    cheat_query = """
        SELECT flag, count(*) FROM (
            SELECT unnest(e.cheat_flags) AS flag
            FROM evals e JOIN attempts a USING (attempt_id)
            WHERE a.run_id = ?)
        GROUP BY flag
    """
    return Metrics(
        run_id=run_id,
        tasks=n,
        solved=len(solved),
        fast_p={p: share(sum(1 for s in eager if s > p), len(counted)) for p in FAST_P},
        fast_p_compile={p: share(sum(1 for s in compile_ if s > p), len(counted)) for p in FAST_P},
        correct_within={
            k: share(sum(1 for r in solved if r.first_correct <= k), n) for k in range(1, max_k + 1)
        },
        median_speedup_eager=median(eager) if eager else None,
        geomean_speedup_eager=_geomean(eager),
        median_speedup_compile=median(compile_) if compile_ else None,
        geomean_speedup_compile=_geomean(compile_),
        api_usd=api_usd,
        gpu_usd=gpu_usd,
        cost_per_correct=(api_usd + gpu_usd) / len(solved) if solved else None,
        cheats=dict(Counter(dict(db.sql(cheat_query, [run_id])))),
        per_task=results,
    )
