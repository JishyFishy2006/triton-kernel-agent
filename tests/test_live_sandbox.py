"""Live B11 acceptance in one Modal Sandbox on an H100. Roadmap: an infinite loop is killed at
the timeout; an illegal memory access is reported and the next eval still runs. Plus: no
network, no secrets, a printed fake result is ignored, and real kernels come out correct.

Run with: uv run pytest -m live tests/test_live_sandbox.py   (a few minutes)
"""

import os
from pathlib import Path

import pytest

from triton_kernel_agent.harness_result import Status
from triton_kernel_agent.tasks import get_task

FIXTURES = Path(__file__).parent / "fixtures" / "candidates"


def _modal_authenticated() -> bool:
    return (Path.home() / ".modal.toml").exists() or "MODAL_TOKEN_ID" in os.environ


pytestmark = [pytest.mark.live, pytest.mark.skipif(not _modal_authenticated(), reason="no Modal")]


@pytest.fixture(scope="module")
def runner():
    from triton_kernel_agent.sandbox import SandboxRunner

    with SandboxRunner(eval_timeout_s=240) as r:
        yield r


def fixture(name):
    return (FIXTURES / f"{name}.py").read_text()


def test_no_network_and_no_secrets(runner):
    code, out, err, _ = runner.exec(
        "python", "-c",
        "import urllib.request; urllib.request.urlopen('https://example.com', timeout=10)",
    )  # fmt: skip
    assert code != 0, out
    _, env, _, _ = runner.exec("env")
    assert "META_API_KEY" not in env and "MODAL_TOKEN" not in env


def test_infinite_loop_is_killed_at_the_timeout(runner):
    runner.eval_timeout_s = 60
    try:
        result = runner.evaluate(get_task("L1-19"), fixture("sb_infinite_loop"))
    finally:
        runner.eval_timeout_s = 240
    assert result.status == Status.TIMEOUT
    assert 55 < result.gpu_seconds < 120


def test_illegal_memory_access_is_reported_and_the_next_eval_runs(runner):
    task = get_task("L1-19")
    bad = runner.evaluate(task, fixture("sb_illegal_access"))
    assert bad.status == Status.RUNTIME_ERROR
    assert "illegal memory access" in bad.error_excerpt.lower()

    good = runner.evaluate(task, fixture("ok_relu"))
    assert good.status == Status.CORRECT, good.error_excerpt
    assert good.timing.kernel_ms > 0 and good.pct_peak_bandwidth > 50


def test_a_printed_fake_result_is_ignored(runner):
    result = runner.evaluate(get_task("L1-19"), fixture("sb_fake_result"))
    assert result.status == Status.MISMATCH


def test_full_pipeline_on_a_fused_gemm(runner):
    task = get_task("L2-76")
    first = runner.evaluate(task, fixture("ok_gemm_add_relu"))
    assert first.status == Status.CORRECT, first.error_excerpt
    assert first.pct_peak_flops is not None
    print(f"\nL2-76: {first.timing.kernel_ms:.2f} ms, {first.timing.speedup_eager:.2f}x eager, "
          f"{first.pct_peak_flops:.0f}% of fp32 peak, {first.gpu_seconds:.0f} s")  # fmt: skip

    # A second attempt reuses the eager/compile timings and caught cheats are reported.
    cached = runner.evaluate(task, fixture("cheat_submodule_call"))
    assert cached.status == Status.STATIC_CHEAT
    second = runner.evaluate(task, fixture("ok_gemm_add_relu"))
    assert second.extra["benchmark"]["baselines_cached"] is True
    assert second.timing.eager_ms == first.timing.eager_ms


def test_honest_triton_matmul_passes_at_full_size(runner):
    # 4096^3 in fp32 with in-order accumulation over K: must not be rejected for rounding.
    result = runner.evaluate(get_task("L1-1"), fixture("ok_matmul"))
    assert result.status == Status.CORRECT, result.error_excerpt
