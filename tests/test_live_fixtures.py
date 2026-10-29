"""Live B8 acceptance on a Modal H100, plus the B7 real-kernel claim.

Roadmap: a fixture that caches outputs fails; one returning a view of the input fails; one
calling cuBLAS through a wrapper fails. Every fixture's header says what should happen; the
real Triton kernels (ok_*) must be correct on every hidden case and clean at runtime.

Run with: uv run pytest -m live tests/test_live_fixtures.py
"""

import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.live


def _modal_authenticated() -> bool:
    return (Path.home() / ".modal.toml").exists() or "MODAL_TOKEN_ID" in os.environ


@pytest.fixture(scope="module")
def results():
    from triton_kernel_agent.modal_fixtures import app, fixture_args, run_fixture

    if not _modal_authenticated():
        pytest.skip("Modal not set up; run `modal setup`")
    args = fixture_args()
    with app.run():
        out = list(run_fixture.starmap(args, order_outputs=True))
    return {r["fixture"]: (r, text) for r, (_, text, _, _) in zip(out, args, strict=True)}


def test_every_fixture_behaves_as_its_header_says(results):
    from triton_kernel_agent.modal_fixtures import fixture_header

    problems = []
    for name, (r, text) in results.items():
        if "error" in r:
            problems.append(f"{name}: {r['error'][-300:]}")
            continue
        want_runtime = fixture_header(text, "expect-runtime")
        want = set() if want_runtime == "none" else {f.strip() for f in want_runtime.split(",")}
        got = set(r["runtime"])
        if (want == set() and got) or not want <= got:
            problems.append(f"{name}: runtime {sorted(got)}, expected {sorted(want)}")
        want_correct = fixture_header(text, "expect-correct") == "yes"
        if r["correct"] != want_correct:
            problems.append(f"{name}: correct={r['correct']}, expected {want_correct}")
    assert problems == []


def test_roadmap_fixtures(results):
    assert "input_insensitive" in results["rt_cache_by_shape"][0]["runtime"]
    assert "output_aliases_input" in results["rt_view_of_input"][0]["runtime"]
    assert "aten_compute" in results["rt_hidden_cublas"][0]["runtime"]
    for name in ("ok_relu", "ok_matmul", "ok_softmax", "ok_gemm_add_relu"):
        assert results[name][0]["correct"] and results[name][0]["runtime"] == [], name
