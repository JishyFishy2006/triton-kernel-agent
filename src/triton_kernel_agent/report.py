"""Static leaderboard and report page (B19), built from the results database.

    uv run python -m triton_kernel_agent.report [--db data/results.duckdb] [--out site/index.html]

The human baseline row (Y3) comes from data/baselines.json, written by scripts/run_baselines.py.

One self-contained HTML file (no scripts, no external assets), so it can be opened locally or
published as is. It renders from any database, including a fresh empty one.
"""

import argparse
import html
import json
from datetime import UTC, datetime
from pathlib import Path

from triton_kernel_agent.config import load_config
from triton_kernel_agent.db import ResultsDB
from triton_kernel_agent.metrics import FAST_P, Metrics, compute_metrics

SCOPES = [("Held-out", "held_out", None), ("Held-out, Level 1", "held_out", 1),
          ("Held-out, Level 2", "held_out", 2), ("Dev", "dev", None)]  # fmt: skip
CURVE_COLORS = ["#2a6fdb", "#d9822b", "#2e9e5b", "#b5468c", "#7a5af8"]

CSS = """
:root { --bg: #ffffff; --fg: #1d1f23; --muted: #626873; --line: #e3e5e8; --head: #f5f6f8;
        --good: #1f7a45; --bad: #b42318; }
@media (prefers-color-scheme: dark) {
  :root { --bg: #15171a; --fg: #e6e7e9; --muted: #9aa0a8; --line: #2c3036; --head: #1d2024;
          --good: #5fc48a; --bad: #f2867c; }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--fg);
       font: 15px/1.5 system-ui, -apple-system, Segoe UI, sans-serif; }
main { max-width: 1100px; margin: 0 auto; padding: 24px 16px 64px; }
h1 { font-size: 26px; margin: 0 0 4px; } h2 { font-size: 19px; margin: 36px 0 8px; }
.muted { color: var(--muted); }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
th, td { border-bottom: 1px solid var(--line); padding: 6px 10px; text-align: right;
         white-space: nowrap; }
th { background: var(--head); font-weight: 600; }
th:first-child, td:first-child { text-align: left; }
.good { color: var(--good); } .bad { color: var(--bad); }
svg text { fill: var(--muted); font-size: 11px; }
svg .grid { stroke: var(--line); }
.empty { padding: 32px; border: 1px dashed var(--line); border-radius: 8px; text-align: center; }
"""


def _e(value: object) -> str:
    return html.escape(str(value))


def _pct(x: float | None) -> str:
    return "–" if x is None else f"{100 * x:.0f}%"


def _x(x: float | None) -> str:
    return "–" if x is None else f"{x:.2f}×"


def _usd(x: float | None) -> str:
    return "–" if x is None else f"${x:.2f}"


def _runs(db: ResultsDB) -> list[tuple[str, str, str]]:
    """(run_id, setup, started_at) for every run, newest first."""
    rows = db.sql("SELECT run_id, config, started_at FROM runs ORDER BY started_at DESC")
    return [(r[0], json.loads(r[1]).get("setup", "?"), str(r[2])[:16]) for r in rows]


def _leaderboard(rows: list[tuple[str, str, Metrics]]) -> str:
    head = ("<tr><th>Run</th><th>Setup</th><th>Tasks</th><th>Solved</th>"
            + "".join(f"<th>fast<sub>{p:g}</sub></th>" for p in FAST_P)
            + "<th>fast<sub>1</sub> vs compile</th><th>Correct ≤1</th><th>≤3</th>"
            "<th>Median speedup</th><th>Geomean</th><th>Cost / correct</th>"
            "<th>Total</th></tr>")  # fmt: skip
    body = []
    for run_id, setup, m in rows:
        fast1 = m.fast_p.get(1.0, 0.0)
        body.append(
            f"<tr><td>{_e(run_id)}</td><td>{_e(setup)}</td><td>{m.tasks}</td><td>{m.solved}</td>"
            + "".join(f"<td>{_pct(m.fast_p[p])}</td>" for p in FAST_P[:1])
            + f"<td class='{'good' if fast1 else ''}'>{_pct(fast1)}</td>"
            + "".join(f"<td>{_pct(m.fast_p[p])}</td>" for p in FAST_P[2:])
            + f"<td>{_pct(m.fast_p_compile.get(1.0))}</td>"
            f"<td>{_pct(m.correct_within.get(1))}</td><td>{_pct(m.correct_within.get(3))}</td>"
            f"<td>{_x(m.median_speedup_eager)}</td><td>{_x(m.geomean_speedup_eager)}</td>"
            f"<td>{_usd(m.cost_per_correct)}</td><td>{_usd(m.api_usd + m.gpu_usd)}</td></tr>"
        )
    return f"<div class='scroll'><table>{head}{''.join(body)}</table></div>"


def _curves(rows: list[tuple[str, str, Metrics]]) -> str:
    """Correct within k attempts, one line per run, as inline SVG."""
    w, h, pad = 640, 220, 36
    ks = range(1, 11)

    def xy(k: int, share: float) -> tuple[float, float]:
        return pad + (k - 1) * (w - 2 * pad) / 9, h - pad - share * (h - 2 * pad)

    parts = [f"<svg viewBox='0 0 {w} {h}' width='100%' role='img' "
             "aria-label='Share of tasks correct within k attempts'>"]  # fmt: skip
    for share in (0, 0.25, 0.5, 0.75, 1):
        _, y = xy(1, share)
        parts.append(f"<line class='grid' x1='{pad}' x2='{w - pad}' y1='{y}' y2='{y}'/>"
                     f"<text x='4' y='{y + 4}'>{share:.0%}</text>")  # fmt: skip
    for k in ks:
        x, _ = xy(k, 0)
        parts.append(f"<text x='{x - 3}' y='{h - 12}'>{k}</text>")
    for i, (run_id, setup, m) in enumerate(rows):
        color = CURVE_COLORS[i % len(CURVE_COLORS)]
        points = " ".join(f"{x:.1f},{y:.1f}" for x, y in (xy(k, m.correct_within[k]) for k in ks))
        parts.append(f"<polyline fill='none' stroke='{color}' stroke-width='2' points='{points}'/>")
        parts.append(f"<text x='{w - pad - 150}' y='{pad + 14 * i}' style='fill:{color}'>"
                     f"{_e(setup)} ({_e(run_id)})</text>")  # fmt: skip
    parts.append("</svg>")
    return "".join(parts)


def _per_task(db: ResultsDB, rows: list[tuple[str, str, Metrics]]) -> str:
    tasks = db.sql("SELECT task_id, op_name, task_group, split FROM tasks WHERE role = 'scored' "
                   "ORDER BY split, task_group, task_id")  # fmt: skip
    best = {(run_id, r.task_id): r for run_id, _, m in rows for r in m.per_task}
    head = (
        "<tr><th>Task</th><th>Op</th><th>Group</th><th>Split</th>"
        + "".join(
            f"<th>{_e(setup)}<br><span class='muted'>{_e(run_id)}</span></th>"
            for run_id, setup, _ in rows
        )
        + "</tr>"
    )
    body = []
    for task_id, op, group, split in tasks:
        cells = []
        for run_id, _, _ in rows:
            r = best.get((run_id, task_id))
            if r is None:
                cells.append("<td class='muted'>–</td>")
            elif r.best_speedup_eager is None:
                cells.append(f"<td class='bad'>✗ <span class='muted'>({r.attempts})</span></td>")
            else:
                cls = "good" if r.best_speedup_eager > 1 else ""
                cells.append(f"<td class='{cls}'>{_x(r.best_speedup_eager)} "
                             f"<span class='muted'>@{r.first_correct}</span></td>")  # fmt: skip
        body.append(f"<tr><td>{_e(task_id)}</td><td>{_e(op)}</td><td>{_e(group)}</td>"
                    f"<td>{_e(split)}</td>{''.join(cells)}</tr>")  # fmt: skip
    return f"<div class='scroll'><table>{head}{''.join(body)}</table></div>"


def _cheats(db: ResultsDB) -> str:
    rows = db.sql("""
        SELECT flag, count(*) AS n, count(DISTINCT a.task_id) AS tasks
        FROM (SELECT attempt_id, unnest(cheat_flags) AS flag FROM evals) f
        JOIN attempts a USING (attempt_id)
        GROUP BY flag ORDER BY n DESC
    """)  # fmt: skip
    if not rows:
        return "<p class='muted'>No attempt has been flagged yet.</p>"
    body = "".join(f"<tr><td>{_e(f)}</td><td>{n}</td><td>{t}</td></tr>" for f, n, t in rows)
    return ("<div class='scroll'><table><tr><th>Check</th><th>Attempts caught</th>"
            f"<th>Tasks</th></tr>{body}</table></div>")  # fmt: skip


def _human(db: ResultsDB, baselines: dict, rows: list[tuple[str, str, Metrics]]) -> str:
    """The human baseline next to each run's best kernel on the same tasks."""
    ops = dict(db.sql("SELECT task_id, op_name FROM tasks"))
    best = {(run_id, r.task_id): r for run_id, _, m in rows for r in m.per_task}
    head = ("<tr><th>Task</th><th>Op</th><th>Human ms</th><th>vs eager</th><th>vs compile</th>"
            "<th>% peak</th>"
            + "".join(f"<th>{_e(setup)}<br><span class='muted'>{_e(run_id)}</span></th>"
                      for run_id, setup, _ in rows)
            + "</tr>")  # fmt: skip
    body = []
    for task_id, result in sorted(baselines["results"].items()):
        timing = result.get("timing")
        if result["status"] != "correct" or not timing:
            human = f"<td colspan='4' class='bad'>{_e(result['status'])}</td>"
        else:
            eager = timing["eager_ms"] / timing["kernel_ms"]
            compile_ = timing["compile_ms"] / timing["kernel_ms"]
            peak = result.get("pct_peak_flops") or result.get("pct_peak_bandwidth")
            human = (
                f"<td>{timing['kernel_ms']:.3f}</td><td>{_x(eager)}</td>"
                f"<td>{_x(compile_)}</td><td>{'–' if peak is None else f'{peak:.0f}%'}</td>"
            )
        cells = []
        for run_id, _, _ in rows:
            r = best.get((run_id, task_id))
            if r is None or r.best_speedup_eager is None:
                cells.append(f"<td class='muted'>{'–' if r is None else '✗'}</td>")
            else:
                cells.append(f"<td>{_x(r.best_speedup_eager)} <span class='muted'>"
                             f"({_x(r.best_speedup_compile)} compile)</span></td>")  # fmt: skip
        body.append(f"<tr><td>{_e(task_id)}</td><td>{_e(ops.get(task_id, ''))}</td>{human}"
                    f"{''.join(cells)}</tr>")  # fmt: skip
    return (f"<h2>Human baseline: {_e(baselines['label'])}</h2>"
            "<p class='muted'>Run through the same harness as the agent. Run columns show the "
            "agent's best speedup over eager on the same task.</p>"
            f"<div class='scroll'><table>{head}{''.join(body)}</table></div>")  # fmt: skip


def render(db: ResultsDB, gpu_usd_per_hour: float, baselines: dict | None = None) -> str:
    runs = _runs(db)
    generated = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    sections = []
    if not runs:
        sections.append("<div class='empty'><p>No runs yet.</p><p class='muted'>Results "
                        "appear here after the first batch run.</p></div>")  # fmt: skip
    else:
        for title, split, level in SCOPES:
            rows = [(run_id, setup, compute_metrics(db, run_id, split=split, level=level,
                                                    gpu_usd_per_hour=gpu_usd_per_hour))
                    for run_id, setup, _ in runs]  # fmt: skip
            if all(m.tasks == 0 for _, _, m in rows):
                continue
            sections.append(f"<h2>{_e(title)}</h2>{_leaderboard(rows)}")
            if split == "held_out" and level is None:
                sections.append(f"<h2>Correct within k attempts (held-out)</h2>{_curves(rows)}")
        all_rows = [(run_id, setup, compute_metrics(db, run_id, gpu_usd_per_hour=gpu_usd_per_hour))
                    for run_id, setup, _ in runs]  # fmt: skip
        sections.append("<h2>Best kernel per task</h2>"
                        "<p class='muted'>Speedup over PyTorch eager of the best correct kernel; "
                        "@n is the attempt that first got it right; ✗ (n) means no correct "
                        "kernel in n attempts.</p>" + _per_task(db, all_rows))  # fmt: skip
    if baselines and baselines.get("results"):
        sections.append(_human(db, baselines, all_rows if runs else []))
    sections.append(f"<h2>Cheats caught</h2>{_cheats(db)}")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Triton Kernel Agent results</title><style>{CSS}</style></head>
<body><main>
<h1>Triton Kernel Agent results</h1>
<p class="muted">KernelBench Level 1–2 subset on an H100, strict fp32. fast<sub>p</sub>: share
of tasks correct and more than p× faster than PyTorch eager. Generated {generated}.</p>
{"".join(sections)}
</main></body></html>
"""


def main(argv: list[str] | None = None) -> Path:
    cfg = load_config()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", type=Path, default=cfg.paths.db)
    parser.add_argument("--out", type=Path, default=Path("site/index.html"))
    parser.add_argument("--baselines", type=Path, default=cfg.paths.db.parent / "baselines.json")
    args = parser.parse_args(argv)
    baselines = json.loads(args.baselines.read_text()) if args.baselines.exists() else None
    with ResultsDB(args.db) as db:
        page = render(db, cfg.modal.gpu_usd_per_hour, baselines)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(page)
    print(f"wrote {args.out}")
    return args.out


if __name__ == "__main__":
    main()
