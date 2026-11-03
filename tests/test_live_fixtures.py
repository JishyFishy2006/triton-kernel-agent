"""Live acceptance on Modal H100s: every fixture through the whole isolated harness (B7, B8,
B9, Y4). Each fixture's header says what must happen (see modal_fixtures.py). Real Triton
kernels (ok_*) must be correct; every red-team cheat (y4_*) must fail, or gain nothing.

Run with: uv run pytest -m live tests/test_live_fixtures.py   (about ten minutes)
"""

import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.live


def _modal_authenticated() -> bool:
    return (Path.home() / ".modal.toml").exists() or "MODAL_TOKEN_ID" in os.environ


@pytest.fixture(scope="module")
def results():
    from triton_kernel_agent.modal_fixtures import app, run_all

    if not _modal_authenticated():
        pytest.skip("Modal not set up; run `modal setup`")
    with app.run():
        return run_all()


def test_every_fixture_behaves_as_its_header_says(results):
    from triton_kernel_agent.modal_fixtures import problems

    assert problems(results) == []


def test_roadmap_fixtures(results):
    by_name = {r["fixture"]: r["result"] for r in results}
    assert "input_insensitive" in by_name["rt_cache_by_shape"]["cheat_flags"]
    assert "aten_compute" in by_name["rt_hidden_cublas"]["cheat_flags"]
    for name in ("ok_relu", "ok_matmul", "ok_softmax", "ok_gemm_add_relu"):
        assert by_name[name]["status"] == "correct", name


def test_no_cheat_is_correct_and_faster_than_eager(results):
    for r in results:
        res = r["result"]
        if r["fixture"].startswith(("y4_", "rt_", "cheat_")) and res["status"] == "correct":
            t = res["timing"]
            assert t["eager_ms"] / t["kernel_ms"] < 1.1, r["fixture"]
