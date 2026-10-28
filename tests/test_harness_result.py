import json
import math

import pytest

from triton_kernel_agent.harness_result import (
    ERROR_EXCERPT_CHARS,
    HARNESS_VERSION,
    CheatFlag,
    HarnessResult,
    ShapeClass,
    Status,
    Timing,
)

CORRECT = HarnessResult(
    task_id="L1-19",
    status=Status.CORRECT,
    max_err=0.0,
    timing=Timing(kernel_ms=4.0, eager_ms=4.3, compile_ms=4.4),
    pct_peak_bandwidth=96.1,
    gpu_seconds=12.5,
)
MISMATCH = HarnessResult(
    task_id="L2-76",
    status="mismatch",
    max_err=0.031,
    failing_shape_class="non_pow2",
    failing_shape=[[1023, 8191]],
    error_excerpt="max abs error 0.031 > tolerance",
)
CHEAT = HarnessResult(
    task_id="L1-1",
    status=Status.RUNTIME_CHEAT,
    cheat_flags=["output_aliases_input", "aten_compute"],
)


@pytest.mark.parametrize("result", [CORRECT, MISMATCH, CHEAT])
def test_json_round_trip_is_identical(result):
    text = result.to_json()
    assert HarnessResult.from_json(text) == result
    assert HarnessResult.from_json(text).to_json() == text


def test_json_uses_plain_strings():
    data = json.loads(MISMATCH.to_json())
    assert data["status"] == "mismatch"
    assert data["failing_shape_class"] == "non_pow2"
    assert data["harness_version"] == HARNESS_VERSION
    assert json.loads(CHEAT.to_json())["cheat_flags"] == ["aten_compute", "output_aliases_input"]


def test_nan_and_inf_errors_survive_the_round_trip():
    result = HarnessResult(
        task_id="L1-1", status=Status.NAN_INF, max_err=math.inf, failing_shape_class="large"
    )
    assert HarnessResult.from_json(result.to_json()).max_err == math.inf


def test_strings_are_coerced_to_enums():
    assert MISMATCH.status is Status.MISMATCH
    assert MISMATCH.failing_shape_class is ShapeClass.NON_POW2
    assert CHEAT.cheat_flags == (CheatFlag.ATEN_COMPUTE, CheatFlag.OUTPUT_ALIASES_INPUT)


def test_speedups_and_flags():
    assert CORRECT.timing.speedup_eager == pytest.approx(1.075)
    assert CORRECT.timing.speedup_compile == pytest.approx(1.1)
    assert CORRECT.is_correct
    assert not MISMATCH.is_correct
    assert Status.STATIC_CHEAT.is_cheat and not Status.MISMATCH.is_cheat


def test_long_errors_keep_the_end():
    result = HarnessResult(
        task_id="L1-1", status=Status.COMPILE_ERROR, error_excerpt="x" * 5000 + "SyntaxError"
    )
    assert len(result.error_excerpt) == ERROR_EXCERPT_CHARS
    assert result.error_excerpt.endswith("SyntaxError")


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"status": "correct"}, "timing"),
        ({"status": "mismatch", "timing": Timing(1, 1, 1)}, "timing"),
        ({"status": "runtime_error", "pct_peak_bandwidth": 50.0}, "peak"),
        ({"status": "static_cheat"}, "cheat flags"),
        ({"status": "mismatch", "cheat_flags": ["tf32_dot"]}, "cheat flags"),
        ({"status": "static_cheat", "cheat_flags": ["aten_compute"]}, "static flags"),
        ({"status": "runtime_cheat", "cheat_flags": ["tf32_dot"]}, "runtime flag"),
        ({"status": "mismatch"}, "shape class"),
        ({"status": "timeout", "failing_shape_class": "large"}, "no failing shape"),
    ],
)
def test_inconsistent_results_are_rejected(kwargs, match):
    with pytest.raises(ValueError, match=match):
        HarnessResult(task_id="L1-1", **kwargs)


def test_unknown_values_are_rejected():
    with pytest.raises(ValueError):
        HarnessResult(task_id="L1-1", status="passed")
    with pytest.raises(ValueError):
        HarnessResult(task_id="L1-1", status="static_cheat", cheat_flags=["sneaky"])


def test_fields_cover_the_evals_table():
    # Tech spec, Data model: evals(attempt_id, status, max_err, failing_shape_class,
    # speedup_eager, speedup_compile, pct_peak, cheat_flags, gpu_seconds). attempt_id is
    # added by the orchestrator; speedups come from timing; pct_peak is split in two.
    data = CORRECT.to_dict()
    for column in ("status", "max_err", "failing_shape_class", "cheat_flags", "gpu_seconds"):
        assert column in data
    assert {"pct_peak_bandwidth", "pct_peak_flops", "timing"} <= data.keys()
