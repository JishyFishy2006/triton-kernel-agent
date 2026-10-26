"""Hello-GPU on Modal: report which GPU and CUDA version a container actually gets.

Run with: uv run modal run src/triton_kernel_agent/modal_hello.py
"""

import subprocess
from dataclasses import asdict

import modal

from triton_kernel_agent.gpu_info import SMI_QUERY, parse_nvidia_smi

if modal.is_local():
    from triton_kernel_agent.config import load_config

    # config.toml exists only on the local machine; the container never re-reads it.
    GPU = load_config().modal.gpu
else:
    GPU = None

app = modal.App("tka-hello-gpu")
image = modal.Image.debian_slim(python_version="3.11").add_local_python_source(
    "triton_kernel_agent"
)


@app.function(gpu=GPU, image=image, timeout=120)
def hello_gpu() -> dict:
    header = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=True).stdout
    query = subprocess.run(
        ["nvidia-smi", *SMI_QUERY], capture_output=True, text=True, check=True
    ).stdout
    return asdict(parse_nvidia_smi(header, query))


@app.local_entrypoint()
def main() -> None:
    for key, value in hello_gpu.remote().items():
        print(f"{key:>15}: {value}")
