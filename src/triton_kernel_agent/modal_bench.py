"""B9 acceptance on an H100: identical kernels time the same, slower kernels time slower,
side-stream work is timed, and a kernel that changes after the checks is caught.

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
# Extra GPU time added to the slow kernel: ~1 ms at H100 clocks.
SLEEP_CYCLES = 2_000_000

app = modal.App("tka-bench")


@app.function(gpu=GPU, image=image, timeout=1800)
def bench_acceptance(task_source: str, fixtures: dict[str, str]) -> dict:
    import torch
    from triton.testing import do_bench

    from triton_kernel_agent.benchmark import benchmark
    from triton_kernel_agent.harness_result import ShapeClass
    from triton_kernel_agent.hidden_shapes import build_case, generate
    from triton_kernel_agent.reference import INIT_SEED
    from triton_kernel_agent.runtime_check import load_candidate

    large = next(c for c in generate("L1-19", task_source) if c.shape_class == ShapeClass.LARGE)
    ref, init, inputs = build_case(task_source, large, "cuda")

    def candidate(name: str, source: str):
        torch.manual_seed(INIT_SEED)
        return load_candidate(source, name).ModelNew(*init).cuda()

    slow_source = fixtures["ok_relu"].replace(
        "        return out\n", f"        torch.cuda._sleep({SLEEP_CYCLES})\n        return out\n"
    )
    out: dict = {"input_shapes": [list(x.shape) for x in inputs]}
    first = benchmark(ref, candidate("ok_a", fixtures["ok_relu"]), inputs, 1e-4, 1e-4)
    out["first"] = {"timing": first.timing.__dict__, "extra": first.extra, "ok": first.consistent}
    for name, source in [
        ("second", fixtures["ok_relu"]),
        ("slow", slow_source),
        ("side_stream", fixtures["cheat_side_stream"]),
        ("lazy", fixtures["rt_lazy_after_checks"]),
    ]:
        model = candidate(name, source)
        result = benchmark(ref, model, inputs, 1e-4, 1e-4, compile_baseline=False)
        out[name] = {
            "kernel_ms": result.timing.kernel_ms,
            "runs": result.extra["kernel_runs_ms"],
            "consistent": result.consistent,
            "post_check": result.post_check.status,
        }
        if name == "side_stream":
            # What a timer without the device-wide sync would have reported.
            out[name]["unsynced_ms"] = do_bench(lambda m=model: m(*inputs), return_mode="median")
    return out


@app.local_entrypoint()
def main() -> None:
    from triton_kernel_agent.tasks import get_task

    names = ("ok_relu", "cheat_side_stream", "rt_lazy_after_checks")
    fixtures = {n: (FIXTURES / f"{n}.py").read_text() for n in names}
    print(
        json.dumps(
            bench_acceptance.remote(get_task("L1-19").source, fixtures), indent=1, default=str
        )
    )
