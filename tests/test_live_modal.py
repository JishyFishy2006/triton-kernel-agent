"""Live Modal run on an H100. Costs a few cents.

Run with: uv run pytest -m live
"""

import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.live


def _modal_authenticated() -> bool:
    return (Path.home() / ".modal.toml").exists() or "MODAL_TOKEN_ID" in os.environ


@pytest.mark.skipif(not _modal_authenticated(), reason="Modal not set up; run `modal setup`")
def test_hello_gpu_returns_h100_name_and_cuda_version():
    from triton_kernel_agent.modal_hello import app, hello_gpu

    with app.run():
        info = hello_gpu.remote()

    print(f"\nmodal hello-gpu: {info}")
    assert "H100" in info["name"]
    assert info["form_factor"] == "SXM"
    assert info["cuda_version"]
