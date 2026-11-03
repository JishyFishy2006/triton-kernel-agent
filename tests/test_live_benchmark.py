"""Live B9 acceptance on a Modal H100. Roadmap: two identical kernels measure within 5%; a
kernel with an added sleep measures slower. Plus L4 #2 (side stream) and a kernel that changes
after the checks. Takes about a minute.

Run with: uv run pytest -m live tests/test_live_benchmark.py
"""

import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.live


def _modal_authenticated() -> bool:
    return (Path.home() / ".modal.toml").exists() or "MODAL_TOKEN_ID" in os.environ


@pytest.fixture(scope="module")
def result():
    from triton_kernel_agent.modal_bench import acceptance, app

    if not _modal_authenticated():
        pytest.skip("Modal not set up; run `modal setup`")
    with app.run():
        return acceptance()


def ms(r):
    return r["timing"]["kernel_ms"]


def test_identical_kernels_measure_within_5_percent(result):
    first, second = result["first"], result["second"]
    assert first["status"] == second["status"] == "correct"
    assert abs(ms(second) / ms(first) - 1) < 0.05, (ms(first), ms(second))
    assert second["extra"]["benchmark"]["baselines_cached"]


def test_a_kernel_with_an_added_sleep_measures_slower(result):
    assert result["slow"]["status"] == "correct"
    assert ms(result["slow"]) > 1.5 * ms(result["first"])


def test_side_stream_kernels_are_caught_or_fully_timed(result):
    # Its stream does not wait for the new input values written between timed batches, so the
    # checked output of a batch comes out wrong (seen on an H100). Were it correct, the worker's
    # device-wide sync would put its work inside the parent's clock.
    side = result["side_stream"]
    if side["status"] == "correct":
        assert ms(side) > 0.8 * ms(result["first"])
    else:
        assert side["cheat_flags"] == ["changed_after_checks"]


def test_a_kernel_that_changes_after_the_checks_is_caught(result):
    assert result["lazy"]["status"] == "runtime_cheat"
    # Its skipped calls return the last output: right values, impossibly fast.
    assert set(result["lazy"]["cheat_flags"]) <= {"changed_after_checks", "impossible_speed"}


def test_baselines_are_timed(result):
    timing, bench = result["first"]["timing"], result["first"]["extra"]["benchmark"]
    assert timing["eager_ms"] > 0 and timing["compile_ms"] > 0
    assert len(bench["kernel_runs_ms"]) == 5 and bench["calls_per_batch"] >= 1
