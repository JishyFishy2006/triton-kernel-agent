"""Roofline checks on an H100 (B10): the analytic numbers, and opt-in Nsight Compute.

- copy_bandwidth: a plain Triton copy kernel timed with B9's bench(); B10's acceptance is that it
  reaches at least 60% of peak bandwidth by the analytic roofline.
- ncu_profile: run one candidate kernel under `ncu --clock-control none --section SpeedOfLight`
  (decisions/ncu_on_modal.md) and return the table. Diagnostic only: slow, and its numbers move
  with the boost clock.

Run with: uv run modal run src/triton_kernel_agent/modal_ncu.py
"""

import csv
import json
from pathlib import Path

import modal

from triton_kernel_agent.modal_tasks import GPU, TORCH_VERSION, image

FIXTURES = (
    Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "candidates"
    if modal.is_local()
    else None
)
NCU_IMAGE = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.11")
    .run_commands(
        "apt-get update", "apt-get install -y --no-install-recommends cuda-nsight-compute-12-8"
    )
    .pip_install(f"torch=={TORCH_VERSION}", "numpy")
    .add_local_python_source("triton_kernel_agent")
)
NCU = "/usr/local/cuda-12.8/bin/ncu"
# Runs inside the ncu container: builds the task's large hidden case and calls the candidate
# twice (the first launch compiles; ncu skips it).
RUNNER = """
import sys
import torch
from triton_kernel_agent.harness_result import ShapeClass
from triton_kernel_agent.hidden_shapes import build_case, generate
from triton_kernel_agent.runtime_check import load_candidate

task_source, cand_source = open(sys.argv[1]).read(), open(sys.argv[2]).read()
case = next(c for c in generate("ncu", task_source) if c.shape_class == ShapeClass.LARGE)
_, init, inputs = build_case(task_source, case, "cuda")
model = load_candidate(cand_source).ModelNew(*init).cuda()
for _ in range(2):
    model(*inputs)
torch.cuda.synchronize()
"""

app = modal.App("tka-roofline")


@app.function(gpu=GPU, image=image, timeout=600)
def copy_bandwidth(cand_source: str, nbytes_in: int = 2**30) -> dict:
    import torch
    from triton.testing import do_bench

    from triton_kernel_agent.roofline import peak_for, roofline
    from triton_kernel_agent.runtime_check import load_candidate

    model = load_candidate(cand_source, "copy").ModelNew()
    x = torch.rand(nbytes_in // 4, device="cuda")
    # Our own copy kernel (B10's acceptance), so timing it in-process is fine.
    kernel_ms = min(do_bench(lambda: model(x), return_mode="median") for _ in range(3))
    r = roofline(nbytes=2 * nbytes_in, flops=0, kernel_ms=kernel_ms,
                 peak=peak_for(torch.cuda.get_device_name()))  # fmt: skip
    return {"kernel_ms": kernel_ms, **r.as_dict()}


@app.function(gpu=GPU, image=NCU_IMAGE, timeout=900)
def ncu_profile(task_source: str, cand_source: str) -> dict:
    import subprocess

    from triton_kernel_agent.runtime_check import triton_kernel_names

    for name, text in (("task.py", task_source), ("cand.py", cand_source), ("run.py", RUNNER)):
        Path("/tmp", name).write_text(text)
    names = "|".join(sorted(triton_kernel_names(cand_source)))
    cmd = [
        NCU, "--clock-control", "none", "--target-processes", "all", "--csv",
        "--section", "SpeedOfLight", "--kernel-name", f"regex:^({names})",
        "--launch-skip", "1", "--launch-count", "1",
        "python", "/tmp/run.py", "/tmp/task.py", "/tmp/cand.py",
    ]  # fmt: skip
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd="/tmp")
    rows = [line for line in proc.stdout.splitlines() if line.startswith('"')]
    metrics = {
        f"{r['Metric Name']} ({r['Metric Unit']})": r["Metric Value"]
        for r in csv.DictReader(rows)
        if r.get("Metric Name")
    }
    return {"returncode": proc.returncode, "metrics": metrics, "stderr": proc.stderr[-2000:]}


@app.local_entrypoint()
def main() -> None:
    from triton_kernel_agent.tasks import get_task

    copy = (FIXTURES / "ok_copy.py").read_text()
    print("copy:", json.dumps(copy_bandwidth.remote(copy), indent=1))
    relu = (FIXTURES / "ok_relu.py").read_text()
    print("ncu:", json.dumps(ncu_profile.remote(get_task("L1-19").source, relu), indent=1))
