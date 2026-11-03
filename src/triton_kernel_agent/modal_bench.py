"""B9 acceptance on an H100, through the whole isolated harness (Y4): identical kernels time
the same, slower kernels time slower, side-stream work is timed, and a kernel that changes
after the checks is caught.

Run with: uv run modal run src/triton_kernel_agent/modal_bench.py
"""

import json
from pathlib import Path

import modal

from triton_kernel_agent.modal_tasks import GPU, image

FIXTURES = (
    Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "candidates"
    if modal.is_local()
    else None
)

app = modal.App("tka-bench")


@app.function(gpu=GPU, image=image, timeout=1800)
def bench_acceptance(task: dict, fixtures: dict[str, str]) -> dict:
    from triton_kernel_agent.harness import EvalRequest, evaluate

    def run(source: str, baselines=None, skip_static=False) -> dict:
        req = EvalRequest(**task, candidate_source=source, baselines=baselines,
                          skip_static_scan=skip_static)  # fmt: skip
        return json.loads(evaluate(req).to_json())

    # The same kernel launched twice: about twice the work, all of it Triton.
    launch = "        relu_kernel[grid](x, out, n, BLOCK=1024)\n"
    slow = fixtures["ok_relu"].replace(launch, launch * 2)
    assert slow != fixtures["ok_relu"]
    out = {"first": run(fixtures["ok_relu"])}
    bench = out["first"]["extra"]["benchmark"]
    cached = {"eager_runs_ms": bench["eager_runs_ms"], "compile_runs_ms": bench["compile_runs_ms"]}
    out["second"] = run(fixtures["ok_relu"], cached)
    out["slow"] = run(slow, cached)
    # Side streams are a static cheat; skip the scan to time one anyway.
    out["side_stream"] = run(fixtures["cheat_side_stream"], cached, skip_static=True)
    out["lazy"] = run(fixtures["rt_lazy_after_checks"], cached)
    return out


@app.local_entrypoint()
def main() -> None:
    print(json.dumps(acceptance(), indent=1, default=str))


def acceptance() -> dict:
    from triton_kernel_agent.tasks import get_task

    t = get_task("L1-19")
    task = {
        "task_id": t.id, "task_source": t.source, "atol": t.atol, "rtol": t.rtol,
        "bytes": t.bytes or 0, "flops": t.flops or 0, "input_domain": t.input_domain,
    }  # fmt: skip
    names = ("ok_relu", "cheat_side_stream", "rt_lazy_after_checks")
    fixtures = {n: (FIXTURES / f"{n}.py").read_text() for n in names}
    return bench_acceptance.remote(task, fixtures)
