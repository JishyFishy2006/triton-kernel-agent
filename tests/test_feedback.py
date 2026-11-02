"""B15 acceptance: snapshot tests for each category; a test asserts no hidden shape appears in
any hint. Regenerate snapshots after an intended wording change with:
    UPDATE_SNAPSHOTS=1 uv run pytest tests/test_feedback.py
"""

import os
import re
from pathlib import Path

import pytest

from triton_kernel_agent.feedback import CHEAT_HINTS, SHAPE_CLASS_HINTS, format_feedback
from triton_kernel_agent.harness_result import (
    CheatFlag,
    HarnessResult,
    ShapeClass,
    Status,
    Timing,
)

SNAPSHOTS = Path(__file__).parent / "snapshots" / "feedback"
HIDDEN = [[1237, 2913], [2913, 4099]]  # distinctive dims, so a leak is easy to spot

CASES = {
    "correct_fast": HarnessResult(
        "L2-76", Status.CORRECT, max_err=2e-6,
        timing=Timing(kernel_ms=2.41, eager_ms=2.77, compile_ms=2.74),
        pct_peak_bandwidth=12.4, pct_peak_flops=85.3, extra={"roofline": {"bound": "compute"}},
    ),
    "correct_slow": HarnessResult(
        "L1-19", Status.CORRECT, max_err=0.0,
        timing=Timing(kernel_ms=5.10, eager_ms=4.30, compile_ms=4.31),
        pct_peak_bandwidth=75.9, extra={"roofline": {"bound": "memory"}},
    ),
    "compile_error": HarnessResult(
        "L1-1", Status.COMPILE_ERROR,
        error_excerpt="triton.compiler.errors.CompilationError: at 12:8: tl.dot needs 2-D blocks",
    ),
    "static_cheat": HarnessResult(
        "L1-1", Status.STATIC_CHEAT,
        cheat_flags=[CheatFlag.TORCH_COMPUTE_CALL, CheatFlag.TF32_DOT],
        error_excerpt="line 30: [torch_compute_call] calls torch.matmul outside a Triton kernel\n"
        'line 14: [tf32_dot] tl.dot needs input_precision="ieee" (strict fp32)',
    ),
    "timeout": HarnessResult("L1-19", Status.TIMEOUT, error_excerpt="killed after 300 s"),
    "runtime_error_hidden": HarnessResult(
        "L1-8", Status.RUNTIME_ERROR, extra={"case": "hidden"},
        error_excerpt="RuntimeError: CUDA error: an illegal memory access (block 1237 of 2913)",
    ),
    "runtime_error_visible": HarnessResult(
        "L1-8", Status.RUNTIME_ERROR, extra={"case": "visible"},
        error_excerpt='File "candidate.py", line 42: IndexError: index 8205 is out of bounds',
    ),
    "wrong_shape": HarnessResult(
        "L1-8", Status.WRONG_SHAPE, failing_shape_class=ShapeClass.SIZE_ONE,
        failing_shape=[[1, 2913], [2913, 4099]],
        error_excerpt="output is 1-D torch.float32, expected 2-D torch.float32",
    ),
    "nan_inf": HarnessResult(
        "L1-23", Status.NAN_INF, max_err=float("inf"),
        failing_shape_class=ShapeClass.INPUT_DISTRIBUTION, failing_shape=HIDDEN,
        error_excerpt="output has NaN or Inf where the reference is finite",
    ),
    "mismatch_non_pow2": HarnessResult(
        "L1-8", Status.MISMATCH, max_err=0.0412, failing_shape_class=ShapeClass.NON_POW2,
        failing_shape=HIDDEN,
        # Hypothetical leak: an excerpt that slipped through with real sizes in it.
        error_excerpt="max abs error 0.0412 vs fp64; row 1237 of 2913 differs",
    ),
    "mismatch_large": HarnessResult(
        "L1-7", Status.MISMATCH, max_err=3.5, failing_shape_class=ShapeClass.LARGE,
        failing_shape=HIDDEN, error_excerpt="max abs error 3.5 vs fp64",
    ),
    "runtime_cheat": HarnessResult(
        "L1-19", Status.RUNTIME_CHEAT,
        cheat_flags=[CheatFlag.INPUT_INSENSITIVE, CheatFlag.OUTPUT_ALIASES_INPUT],
    ),
    "changed_after_checks": HarnessResult(
        "L1-19", Status.RUNTIME_CHEAT, cheat_flags=[CheatFlag.CHANGED_AFTER_CHECKS],
    ),
    "harness_error": HarnessResult(
        "L1-19", Status.HARNESS_ERROR,
        error_excerpt="Traceback ... /root/triton_kernel_agent/harness.py line 88 KeyError",
    ),
}  # fmt: skip


@pytest.mark.parametrize("name", sorted(CASES))
def test_snapshot(name):
    text = format_feedback(CASES[name])
    path = SNAPSHOTS / f"{name}.txt"
    if os.environ.get("UPDATE_SNAPSHOTS"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n")
    assert text + "\n" == path.read_text()


def test_every_status_has_a_snapshot():
    assert {r.status for r in CASES.values()} == set(Status)


def numbers(text: str) -> set[str]:
    return set(re.findall(r"\d+", text))


@pytest.mark.parametrize("name", sorted(CASES))
def test_no_hidden_shape_appears_in_any_hint(name):
    result = CASES[name]
    text = format_feedback(result)
    # Multi-digit dims identify a shape; "size 1" or "2-D" say nothing a shape class doesn't.
    hidden = {str(d) for shape in (result.failing_shape or []) for d in shape if d >= 10}
    assert not (numbers(text) & hidden), text
    assert "1237" not in text and "2913" not in text and "4099" not in text


def test_visible_failures_may_show_their_shapes():
    result = HarnessResult(
        "L1-8", Status.MISMATCH, max_err=0.5, failing_shape_class=ShapeClass.VISIBLE,
        failing_shape=[[8205, 2949]], error_excerpt="output is [8205, 5921] torch.float32",
    )  # fmt: skip
    assert "8205" in format_feedback(result)


def test_visible_crashes_keep_line_numbers():
    assert "line 42" in format_feedback(CASES["runtime_error_visible"])


def test_harness_errors_show_nothing_internal():
    text = format_feedback(CASES["harness_error"])
    assert "Traceback" not in text and "harness.py" not in text


def test_every_flag_and_shape_class_has_a_hint():
    assert set(CHEAT_HINTS) == set(CheatFlag)
    assert set(SHAPE_CLASS_HINTS) == set(ShapeClass)
