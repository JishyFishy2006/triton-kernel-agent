"""Live B4 acceptance test: every task's reference runs on a Modal H100. Costs well under $1.

Run with: uv run pytest -m live tests/test_live_tasks.py
"""

import json
import os
from pathlib import Path

import pytest

from triton_kernel_agent.tasks import METADATA_PATH, load_tasks

pytestmark = pytest.mark.live


def _modal_authenticated() -> bool:
    return (Path.home() / ".modal.toml").exists() or "MODAL_TOKEN_ID" in os.environ


@pytest.mark.skipif(not _modal_authenticated(), reason="Modal not set up; run `modal setup`")
def test_every_reference_op_runs_on_the_gpu_and_matches_metadata():
    from triton_kernel_agent.modal_tasks import app, measure

    tasks = load_tasks(include_canaries=True)
    with app.run():
        results = list(measure.starmap([(t.id, t.source) for t in tasks], order_outputs=True))

    recorded = json.loads(METADATA_PATH.read_text())["tasks"]
    assert len(results) == 45
    for r in results:
        assert "H100" in r["gpu"]
        assert r["off_device_ops"] == [], r["task_id"]
        assert (r["bytes"], r["flops"]) == (
            recorded[r["task_id"]]["bytes"],
            recorded[r["task_id"]]["flops"],
        ), r["task_id"]
