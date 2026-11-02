"""L9 probe: can Nsight Compute (ncu) read GPU performance counters on a Modal H100?

Result on 2026-10-29 (driver 610.57.04, ncu 2025.1.1): yes, with --clock-control none.
RmProfilingAdminOnly is 0, so counters are not admin-only; ncu's default clock locking fails
("Failed to lock GPU clock frequencies") because the container cannot control host clocks.
See decisions/ncu_on_modal.md.

Run with: uv run modal run scripts/l9_ncu_probe.py
"""

from pathlib import Path

import modal

app = modal.App("tka-l9-ncu-probe")
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.11")
    .run_commands(
        "apt-get update",
        "apt-get install -y --no-install-recommends cuda-nsight-compute-12-8 || true",
    )
    .pip_install("torch==2.14.1", "numpy")
)

WORKLOAD = """
import torch
x = torch.randn(4096, 4096, device="cuda")
for _ in range(3):
    y = torch.relu(x)
torch.cuda.synchronize()
print("workload done")
"""


@app.function(gpu="H100!", image=image, timeout=900)
def probe() -> dict:
    import glob
    import subprocess

    def sh(cmd: str) -> str:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return f"$ {cmd}\n[exit {p.returncode}]\n{(p.stdout + p.stderr)[-4000:]}"

    ncu = (
        glob.glob("/usr/local/cuda*/bin/ncu") + glob.glob("/opt/nvidia/nsight-compute/*/ncu") + [""]
    )[0]
    Path("/tmp/workload.py").write_text(WORKLOAD)
    return {
        "gpu": sh("nvidia-smi --query-gpu=name,driver_version --format=csv,noheader"),
        "admin_only": sh("grep -i RmProfilingAdminOnly /proc/driver/nvidia/params"),
        "ncu_path": ncu or "ncu not found",
        "ncu_version": sh(f"{ncu} --version") if ncu else "",
        "whoami": sh("id"),
        "ncu_run": sh(
            f"{ncu} --clock-control none --target-processes all --launch-count 1 "
            "--metrics dram__throughput.avg.pct_of_peak_sustained_elapsed,"
            "sm__throughput.avg.pct_of_peak_sustained_elapsed "
            "python /tmp/workload.py"
        )
        if ncu
        else "",
        "ncu_speed_of_light": sh(
            f"{ncu} --clock-control none --target-processes all --launch-count 1 "
            "--section SpeedOfLight python /tmp/workload.py"
        )
        if ncu
        else "",
    }


@app.local_entrypoint()
def main() -> None:
    for key, value in probe.remote().items():
        print(f"===== {key}\n{value}\n")
