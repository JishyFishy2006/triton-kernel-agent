"""Run candidate fixtures (tests/fixtures/candidates) through the harness checks on an H100.

For each fixture: the static scan (B7), correctness on every hidden case of its task (B5, B6),
then the runtime checks (B8) with a HarnessGuard taken before the candidate's code runs.

Run with: uv run modal run src/triton_kernel_agent/modal_fixtures.py [--only ok_relu,rt_...]
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
def run_fixture(name: str, source: str, task_source: str, input_domain: str) -> dict:
    import dataclasses
    import traceback

    import torch

    from triton_kernel_agent.correctness import Case, run_case
    from triton_kernel_agent.hidden_shapes import build_case, generate
    from triton_kernel_agent.reference import INIT_SEED
    from triton_kernel_agent.runtime_check import HarnessGuard, check_runtime, load_candidate
    from triton_kernel_agent.static_check import scan

    result: dict = {"fixture": name, "static": [f.value for f in scan(source).flags]}
    guard = HarnessGuard()
    try:
        model_new = load_candidate(source, name).ModelNew  # the candidate's import

        def candidate(init):
            torch.manual_seed(INIT_SEED)
            return model_new(*init).cuda()

        cases = generate(name, task_source, input_domain=input_domain)
        checks = []
        for hidden in cases:
            ref, init, inputs = build_case(task_source, hidden, "cuda")
            check = run_case(ref, candidate(init), Case(inputs, hidden.shape_class), 1e-4, 1e-4)
            checks.append({"shape_class": hidden.shape_class, "status": check.status,
                           "message": check.message})  # fmt: skip
        result["correctness"] = checks
        result["correct"] = all(c["status"] is None for c in checks)

        first = cases[0]
        ref, init, inputs = build_case(task_source, first, "cuda")
        _, _, fresh = build_case(
            task_source, dataclasses.replace(first, seed=first.seed + 1), "cuda"
        )
        report = check_runtime(ref, candidate(init), inputs, fresh, source, guard)
        result["runtime"] = [f.value for f in report.flags]
        result["runtime_messages"] = [f.message for f in report.findings]
        result["gpu_kernels"] = sorted(set(report.gpu_kernels))
        result["aten_ops"] = sorted(set(report.aten_ops))
    except Exception:
        result["error"] = traceback.format_exc()[-1500:]
    finally:
        guard.restore()  # containers are reused: undo whatever the candidate patched
    return result


def fixture_args(names: set[str] | None = None) -> list[tuple[str, str, str, str]]:
    from triton_kernel_agent.tasks import get_task

    args = []
    for path in sorted(FIXTURES.glob("*.py")):
        text = path.read_text()
        if names and path.stem not in names:
            continue
        if fixture_header(text, "expect-runtime") is None:
            continue  # static-only fixture
        task = get_task(fixture_header(text, "task"))
        args.append((path.stem, text, task.source, task.input_domain))
    return args


@app.local_entrypoint()
def main(only: str = "", out: str = "") -> None:
    args = fixture_args(set(only.split(",")) if only else None)
    results = list(run_fixture.starmap(args, order_outputs=True))
    if out:
        Path(out).write_text(json.dumps(results, indent=1, default=str))
    for r in results:
        if "error" in r:
            print(f"{r['fixture']:24} ERROR {r['error'][-300:]}")
            continue
        print(f"{r['fixture']:24} correct={r['correct']!s:5} runtime={r['runtime']} "
              f"kernels={r['gpu_kernels'][:3]}")  # fmt: skip
    print(f"{len(results)} fixtures", file=sys.stderr)
