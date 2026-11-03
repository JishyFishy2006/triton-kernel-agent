"""Run candidate fixtures (tests/fixtures/candidates) through the whole harness on an H100.

Each fixture goes through harness.evaluate() exactly as an agent attempt does: static scan,
an unprivileged worker, hidden cases, runtime checks, clean baselines, timing (Y4 isolation).
Headers say what must happen:
    # task: L1-19
    # expect: <static flags>          (tests/test_static_check.py, offline)
    # expect-status: runtime_cheat    (one status, or several separated by " or ")
    # expect-flags: input_insensitive (optional: flags the result must include)
    # expect-max-speedup: 1.1         (optional: a trick must not buy a speedup over eager)
    # red-team: skip-static           (optional: skip the scan, to test the layers behind it)
Fixtures without expect-status are not run here.

Run with: uv run modal run src/triton_kernel_agent/modal_fixtures.py [--only ok_relu,...]
"""

import json
import sys
from pathlib import Path

import modal

from triton_kernel_agent.modal_tasks import GPU, image

# Only meaningful locally: in the container this file sits at /root/triton_kernel_agent/.
FIXTURES = (
    Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "candidates"
    if modal.is_local()
    else None
)

app = modal.App("tka-fixtures")


def fixture_header(text: str, key: str) -> str | None:
    for line in text.splitlines():
        if line.startswith(f"# {key}:"):
            return line.split(":", 1)[1].strip()
    return None


@app.function(gpu=GPU, image=image, timeout=900, max_containers=10)
def run_fixture(name: str, request: dict) -> dict:
    from triton_kernel_agent.harness import EvalRequest, evaluate

    result = evaluate(EvalRequest(**request))
    return {"fixture": name, "result": json.loads(result.to_json())}


def fixture_args(
    names: set[str] | None = None, baselines: dict[str, dict] | None = None
) -> list[tuple[str, dict]]:
    from triton_kernel_agent.tasks import get_task

    args = []
    for path in sorted(FIXTURES.glob("*.py")):
        text = path.read_text()
        if (names and path.stem not in names) or fixture_header(text, "expect-status") is None:
            continue
        task = get_task(fixture_header(text, "task"))
        request = {
            "task_id": task.id, "task_source": task.source, "candidate_source": text,
            "atol": task.atol, "rtol": task.rtol, "bytes": task.bytes or 0,
            "flops": task.flops or 0, "input_domain": task.input_domain,
            "baselines": (baselines or {}).get(task.id),
            "skip_static_scan": fixture_header(text, "red-team") == "skip-static",
        }  # fmt: skip
        args.append((path.stem, request))
    return args


def run_all(names: set[str] | None = None) -> list[dict]:
    """Every fixture; the first one per task times the baselines, the rest reuse them."""
    args = fixture_args(names)
    first: dict[str, tuple[str, dict]] = {}
    for name, request in args:
        if name.startswith("ok_"):
            first.setdefault(request["task_id"], (name, request))
    for name, request in args:
        first.setdefault(request["task_id"], (name, request))
    leaders = list(first.values())
    out = list(run_fixture.starmap(leaders, order_outputs=True))
    baselines = {}
    for r in out:
        bench = r["result"]["extra"].get("benchmark") or {}
        if bench.get("eager_runs_ms"):
            baselines[r["result"]["task_id"]] = {
                "eager_runs_ms": bench["eager_runs_ms"],
                "compile_runs_ms": bench["compile_runs_ms"],
            }
    done = {name for name, _ in leaders}
    rest = [(n, r) for n, r in fixture_args(names, baselines) if n not in done]
    out += list(run_fixture.starmap(rest, order_outputs=True))
    return out


def problems(results: list[dict]) -> list[str]:
    """Fixtures whose result differs from their header."""
    found = []
    for r in results:
        text = (FIXTURES / f"{r['fixture']}.py").read_text()
        want = {s.strip() for s in fixture_header(text, "expect-status").split(" or ")}
        flags = fixture_header(text, "expect-flags")
        want_flags = {f.strip() for f in flags.split(",")} if flags else set()
        status, got_flags = r["result"]["status"], set(r["result"]["cheat_flags"])
        timing, max_speedup = r["result"].get("timing"), fixture_header(text, "expect-max-speedup")
        if timing and max_speedup and timing["eager_ms"] / timing["kernel_ms"] > float(max_speedup):
            found.append(f"{r['fixture']}: {timing['eager_ms'] / timing['kernel_ms']:.2f}x eager, "
                         f"more than {max_speedup}x")  # fmt: skip
        if status not in want or not want_flags <= got_flags:
            found.append(f"{r['fixture']}: {status} {sorted(got_flags)}, expected "
                         f"{' or '.join(sorted(want))} {sorted(want_flags)}")  # fmt: skip
    return found


@app.local_entrypoint()
def main(only: str = "", out: str = "") -> None:
    results = run_all(set(only.split(",")) if only else None)
    if out:
        Path(out).write_text(json.dumps(results, indent=1, default=str))
    for r in sorted(results, key=lambda r: r["fixture"]):
        res = r["result"]
        timing = res.get("timing") or {}
        speed = f"{timing['eager_ms'] / timing['kernel_ms']:.2f}x" if timing else ""
        print(f"{r['fixture']:34} {res['status']:14} {','.join(res['cheat_flags']):28} {speed}")
    bad = problems(results)
    print("\n".join(bad) or "every fixture behaved as its header says", file=sys.stderr)
