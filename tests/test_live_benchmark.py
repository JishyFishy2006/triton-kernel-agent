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
    from triton_kernel_agent.modal_bench import FIXTURES, app, bench_acceptance
    from triton_kernel_agent.tasks import get_task

    if not _modal_authenticated():
        pytest.skip("Modal not set up; run `modal setup`")
    names = ("ok_relu", "cheat_side_stream", "rt_lazy_after_checks")
    fixtures = {n: (FIXTURES / f"{n}.py").read_text() for n in names}
    with app.run():
        return bench_acceptance.remote(get_task("L1-19").source, fixtures)


def test_identical_kernels_measure_within_5_percent(result):
    first, second = result["first"]["timing"]["kernel_ms"], result["second"]["kernel_ms"]
    assert abs(second / first - 1) < 0.05, (first, second)
    assert result["first"]["ok"] and result["second"]["consistent"]


def test_a_kernel_with_an_added_sleep_measures_slower(result):
    assert result["slow"]["kernel_ms"] > 1.5 * result["first"]["timing"]["kernel_ms"]


def test_side_stream_work_is_timed(result):
    side = result["side_stream"]
    # Timed with the device-wide sync it costs about what the same kernel costs normally;
    # a timer on the current stream alone would have reported a fraction of that.
    assert side["kernel_ms"] > 0.5 * result["first"]["timing"]["kernel_ms"]
    assert side["unsynced_ms"] < 0.5 * side["kernel_ms"]


def test_a_kernel_that_changes_after_the_checks_is_caught(result):
    assert result["lazy"]["consistent"] is False


def test_baselines_are_timed(result):
    timing = result["first"]["timing"]
    assert timing["eager_ms"] > 0 and timing["compile_ms"] > 0
    assert result["first"]["extra"]["repeats"] == 3
