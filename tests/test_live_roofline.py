"""Live B10 acceptance on a Modal H100: a plain copy kernel reports at least 60% of peak
bandwidth. Also checks the opt-in Nsight Compute profile runs (decisions/ncu_on_modal.md).

Run with: uv run pytest -m live tests/test_live_roofline.py
"""

import os
from pathlib import Path

import pytest


def _modal_authenticated() -> bool:
    return (Path.home() / ".modal.toml").exists() or "MODAL_TOKEN_ID" in os.environ


pytestmark = [pytest.mark.live, pytest.mark.skipif(not _modal_authenticated(), reason="no Modal")]


def test_copy_kernel_reaches_60_percent_of_peak_bandwidth():
    from triton_kernel_agent.modal_ncu import FIXTURES, app, copy_bandwidth

    with app.run():
        result = copy_bandwidth.remote((FIXTURES / "ok_copy.py").read_text())
    print(f"\ncopy kernel: {result['pct_peak_bandwidth']:.1f}% of peak bandwidth")
    assert result["pct_peak_bandwidth"] >= 60
    assert result["bound"] == "memory"


def test_ncu_profiles_a_candidate_kernel():
    from triton_kernel_agent.modal_ncu import FIXTURES, app, ncu_profile
    from triton_kernel_agent.tasks import get_task

    with app.run():
        result = ncu_profile.remote(get_task("L1-19").source, (FIXTURES / "ok_relu.py").read_text())
    assert result["returncode"] == 0, result["stderr"]
    assert float(result["metrics"]["DRAM Throughput (%)"]) > 50
