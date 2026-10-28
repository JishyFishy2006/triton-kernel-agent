"""Live B5 + B6 check on a Modal H100: every task's hidden cases, three candidates each.

No false rejections (a copy of the reference and torch.compile pass everywhere), and TF32 is
caught on nearly every case where it changes the output. Takes about a minute.

Run with: uv run pytest -m live tests/test_live_check.py
"""

import os
from pathlib import Path

import pytest

from triton_kernel_agent.tasks import load_tasks

pytestmark = pytest.mark.live


def _modal_authenticated() -> bool:
    return (Path.home() / ".modal.toml").exists() or "MODAL_TOKEN_ID" in os.environ


@pytest.mark.skipif(not _modal_authenticated(), reason="Modal not set up; run `modal setup`")
def test_hidden_cases_on_gpu_have_no_false_rejections_and_catch_tf32():
    from triton_kernel_agent.modal_check import app, check_task

    tasks = load_tasks(include_canaries=True)
    with app.run():
        args = [(t.id, t.source, t.input_domain, t.atol, t.rtol) for t in tasks]
        results = list(check_task.starmap(args, order_outputs=True))

    false_rejections = [
        (r["task_id"], c["shape_class"], name, c[name]["message"])
        for r in results
        for c in r["cases"]
        for name in ("copy", "compile")
        if name in c and c[name]["status"] is not None
    ]
    assert false_rejections == []
    assert all(len(r["cases"]) >= 5 for r in results)

    changed = [
        c["tf32"]
        for r in results
        for c in r["cases"]
        if c["tf32"].get("ref_err") is not None and c["tf32"]["max_err"] != c["tf32"]["ref_err"]
    ]
    caught = sum(1 for c in changed if c["status"] is not None)
    print(f"\nTF32 caught on {caught} of {len(changed)} cases where it changes the output")
    assert caught >= 0.9 * len(changed)
