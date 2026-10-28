"""Run every task's reference on a Modal H100 and record its metadata in tasks/metadata.json.

Run with: uv run modal run src/triton_kernel_agent/modal_tasks.py
Sources are read locally and sent as strings; the container never sees the repo or any secret.
"""

import json
import sys
import tomllib
from dataclasses import asdict

import modal

if modal.is_local():
    from triton_kernel_agent.config import load_config

    # config.toml exists only on the local machine; the container never re-reads it.
    GPU = load_config().modal.gpu
else:
    GPU = None

TORCH_VERSION = "2.14.1"

app = modal.App("tka-tasks")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(f"torch=={TORCH_VERSION}", "numpy")
    .add_local_python_source("triton_kernel_agent")
)


@app.function(gpu=GPU, image=image, timeout=600, max_containers=10)
def measure(task_id: str, source: str) -> dict:
    import gc

    import torch

    from triton_kernel_agent.reference import run_reference

    try:
        run = run_reference(source, device="cuda", task_id=task_id)
    finally:
        # Containers are reused across tasks; some inputs are several GB.
        gc.collect()
        torch.cuda.empty_cache()
    return {
        "task_id": task_id,
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        **asdict(run),
    }


@app.local_entrypoint()
def main(only: str = "") -> None:
    from triton_kernel_agent.tasks import MANIFEST_PATH, METADATA_PATH, load_tasks

    tasks = load_tasks(include_canaries=True)
    if only:
        tasks = [t for t in tasks if t.id in set(only.split(","))]
    print(f"measuring {len(tasks)} tasks", file=sys.stderr)

    results = list(measure.starmap([(t.id, t.source) for t in tasks], order_outputs=True))
    bad = {r["task_id"]: r["off_device_ops"] for r in results if r["off_device_ops"]}
    if bad:
        raise SystemExit(f"reference ops ran off the GPU: {bad}")

    existing = json.loads(METADATA_PATH.read_text())["tasks"] if METADATA_PATH.exists() else {}
    with MANIFEST_PATH.open("rb") as f:
        commit = tomllib.load(f)["kernelbench_commit"]
    metadata = {
        "kernelbench_commit": commit,
        "measured_on": {"gpu": results[0]["gpu"], "torch": results[0]["torch"]},
        "tasks": existing
        | {
            r["task_id"]: {
                k: r[k] for k in ("input_shapes", "output_shape", "output_dtype", "bytes", "flops")
            }
            for r in results
        },
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=1) + "\n")
    for r in results:
        print(f"{r['task_id']:>6}  {r['bytes'] / 1e9:8.3f} GB  {r['flops'] / 1e9:10.1f} GFLOP")
    print(f"wrote {METADATA_PATH}", file=sys.stderr)
