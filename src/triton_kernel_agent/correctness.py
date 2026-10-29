"""Correctness checker (B5): compare a candidate model with the reference on a list of cases.

Rules decided in Y2 (decisions/tasks_chosen.md):
- Strict IEEE fp32: the reference and the candidate both run with TF32 off (decision 1).
- Tolerance is relative to an fp64 run of the reference (decision 2). The candidate passes when
  its worst error against fp64 is at most REF_ERR_MULTIPLE times the fp32 reference's own worst
  error, or at most FLOOR_ULPS ulps of the output's largest magnitude (the floor for ops the
  reference computes exactly). The floor scales with the output, unlike a fixed atol: on an H100
  a fixed 1e-4 floor let TF32 through on L2-86, L2-99 and L2-64 at 100x the reference's error.
  If the fp64 run is impossible (out of memory), fall back to allclose against the fp32
  reference at the task's atol/rtol, as KernelBench does.
- Models stay in training mode, as KernelBench runs them; nothing here calls .eval().

Runs where torch is installed (the Modal container, or tests on CPU); torch is imported lazily.
"""

import copy
import re
import traceback
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from triton_kernel_agent.harness_result import ERROR_EXCERPT_CHARS, ShapeClass, Status

# How much worse than the fp32 reference (both measured against fp64) a candidate may be.
# An honest Triton GEMM that accumulates over K = 8192 in order lands at 9.6x cuBLAS's error
# (L2-76 on an H100), so 8x rejected it; torch.compile stays within 2.6x. TF32 lands at a
# median 190x: 16x still catches it on 141 of the 149 hidden cases where it changes the output.
REF_ERR_MULTIPLE = 16.0
# Floor for outputs the reference computes (nearly) exactly: this many ulps of max |output|.
# 16 ulps let TF32 through on L2-45 and L2-56; torch.compile uses at most 21% of 4 ulps.
FLOOR_ULPS = 4
# Elements per chunk when computing errors in float64, so multi-GB outputs fit in memory.
_CHUNK = 1 << 26


@dataclass(frozen=True)
class Case:
    inputs: Sequence[Any]
    shape_class: ShapeClass
    # Hidden cases never put shapes or sizes in messages; the agent may see the message.
    hidden: bool = True


@dataclass(frozen=True)
class CaseCheck:
    status: Status | None  # None means the case passed
    max_err: float  # worst |candidate - truth| (truth is fp64 when available)
    allowed_err: float  # the bound max_err was held to (nan for the fixed and exact rules)
    ref_err: float | None  # worst |fp32 reference - fp64 reference|; None without fp64
    rule: str  # "fp64_relative", "fixed" or "exact"
    message: str = ""

    @property
    def passed(self) -> bool:
        return self.status is None


@dataclass(frozen=True)
class CorrectnessResult:
    status: Status | None  # None means every case passed
    max_err: float | None  # worst error over all cases checked
    failing_shape_class: ShapeClass | None = None
    failing_shape: list[list[int]] | None = None
    error_excerpt: str = ""
    cases: list[CaseCheck] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.status is None

    def harness_fields(self) -> dict[str, Any]:
        """Keyword arguments for HarnessResult, for a failed check (B11 adds the rest)."""
        # HarnessResult carries a failing shape only for output problems, not crashes.
        output_problem = self.status in (Status.WRONG_SHAPE, Status.NAN_INF, Status.MISMATCH)
        return {
            "status": self.status,
            "max_err": self.max_err,
            "failing_shape_class": self.failing_shape_class if output_problem else None,
            "failing_shape": self.failing_shape if output_problem else None,
            "error_excerpt": self.error_excerpt,
            "extra": {"correctness": [c.__dict__ | {"status": c.status} for c in self.cases]},
        }


@contextmanager
def strict_fp32() -> Iterator[None]:
    """Turn TF32 off for matmul and cuDNN, then restore the previous settings.

    Uses only the legacy flags: torch raises if legacy and new (fp32_precision) APIs are mixed.
    """
    import torch

    matmul, cudnn = torch.backends.cuda.matmul, torch.backends.cudnn
    saved = (matmul.allow_tf32, cudnn.allow_tf32, torch.get_float32_matmul_precision())
    matmul.allow_tf32 = False
    cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    try:
        yield
    finally:
        matmul.allow_tf32, cudnn.allow_tf32 = saved[:2]
        torch.set_float32_matmul_precision(saved[2])


def tf32_enabled() -> bool:
    """True if anything turned TF32 back on. Raises if legacy and new APIs were mixed."""
    import torch

    return (
        torch.backends.cuda.matmul.allow_tf32
        or torch.backends.cudnn.allow_tf32
        or torch.get_float32_matmul_precision() != "highest"
    )


def _redact(text: str) -> str:
    # Hidden-case messages must not leak sizes: drop every multi-digit number.
    return re.sub(r"\d{2,}", "<n>", text)


def _excerpt(text: str, hidden: bool) -> str:
    text = _redact(text) if hidden else text
    return text[-ERROR_EXCERPT_CHARS:]


def _chunks(t) -> Iterator[Any]:
    flat = t.reshape(-1)
    for i in range(0, flat.numel(), _CHUNK):
        yield flat[i : i + _CHUNK]


def _max_abs(a, b) -> float:
    """max |a - b| over positions where b is finite, in float64, chunked."""
    import torch

    worst = 0.0
    for x, y in zip(_chunks(a), _chunks(b), strict=True):
        d = (x.double() - y.double()).abs()[torch.isfinite(y)]
        if d.numel():
            worst = max(worst, d.max().item())
    return worst


def _max_magnitude(t) -> float:
    """max |t| over finite elements, chunked."""
    import torch

    worst = 0.0
    for x in _chunks(t):
        x = x[torch.isfinite(x)].abs()
        if x.numel():
            worst = max(worst, x.max().item())
    return worst


def _nonfinite_problem(candidate, truth) -> Status | None:
    """Where truth is finite the candidate must be too; where it isn't they must agree."""
    import torch

    for c, t in zip(_chunks(candidate), _chunks(truth), strict=True):
        t_finite = torch.isfinite(t)
        if not torch.isfinite(c[t_finite]).all():
            return Status.NAN_INF
        c_bad, t_bad = c[~t_finite], t[~t_finite]
        same = (c_bad == t_bad) | (torch.isnan(c_bad) & torch.isnan(t_bad))
        if not same.all():
            return Status.MISMATCH
    return None


def _describe(t, hidden: bool) -> str:
    return f"{t.dim()}-D {t.dtype}" if hidden else f"{list(t.shape)} {t.dtype}"


def compare(candidate, ref32, ref64=None, atol=1e-4, rtol=1e-4, hidden=True) -> CaseCheck:
    """Check one output tensor. ref64 is the reference run in float64, or None."""
    import torch

    if candidate.shape != ref32.shape or candidate.dtype != ref32.dtype:
        msg = f"output is {_describe(candidate, hidden)}, expected {_describe(ref32, hidden)}"
        return CaseCheck(Status.WRONG_SHAPE, float("nan"), float("nan"), None, "exact", msg)

    if not ref32.is_floating_point():
        wrong = int((candidate != ref32).sum().item())
        err = _max_abs(candidate, ref32) if wrong else 0.0
        if not wrong:
            return CaseCheck(None, 0.0, 0.0, None, "exact")
        msg = "integer output differs from the reference"
        if not hidden:
            msg = f"{wrong} of {ref32.numel()} integer elements differ from the reference"
        return CaseCheck(Status.MISMATCH, err, 0.0, None, "exact", msg)

    truth = ref64 if ref64 is not None else ref32
    if (status := _nonfinite_problem(candidate, truth)) is not None:
        what = "NaN or Inf where the reference is finite"
        if status == Status.MISMATCH:
            what = "finite values where the reference is NaN or Inf"
        return CaseCheck(status, float("inf"), float("nan"), None, "exact", f"output has {what}")

    err = _max_abs(candidate, truth)
    if ref64 is not None:
        ref_err = _max_abs(ref32, ref64)
        floor = FLOOR_ULPS * torch.finfo(ref32.dtype).eps * _max_magnitude(ref64)
        allowed = max(REF_ERR_MULTIPLE * ref_err, floor)
        if err <= allowed:
            return CaseCheck(None, err, allowed, ref_err, "fp64_relative")
        msg = (
            f"max abs error {err:.3g} vs fp64; allowed {allowed:.3g} ({REF_ERR_MULTIPLE:g}x "
            f"the fp32 reference's own error {ref_err:.3g}, floor {floor:.3g})"
        )
        return CaseCheck(Status.MISMATCH, err, allowed, ref_err, "fp64_relative", msg)

    ok = all(
        torch.allclose(c.double(), t.double(), atol=atol, rtol=rtol, equal_nan=True)
        for c, t in zip(_chunks(candidate), _chunks(ref32), strict=True)
    )
    msg = "" if ok else f"max abs error {err:.3g} outside atol={atol:g}, rtol={rtol:g}"
    return CaseCheck(None if ok else Status.MISMATCH, err, float("nan"), None, "fixed", msg)


def _leaves(output) -> list[Any]:
    import torch
    from torch.utils._pytree import tree_leaves

    return [t for t in tree_leaves(output) if isinstance(t, torch.Tensor)]


def _clone(inputs: Sequence[Any], dtype=None) -> list[Any]:
    import torch

    out = []
    for x in inputs:
        if isinstance(x, torch.Tensor):
            x = x.clone()
            if dtype is not None and x.is_floating_point():
                x = x.to(dtype)
        out.append(x)
    return out


def _reference_fp64(ref_model, inputs: Sequence[Any]) -> list[Any] | None:
    import torch

    try:
        model64 = copy.deepcopy(ref_model).double()
        with torch.no_grad():
            return _leaves(model64(*_clone(inputs, torch.float64)))
    except torch.OutOfMemoryError:
        return None
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def poison_free_memory(min_bytes: int = 1 << 30) -> None:
    """Fill the CUDA caching allocator's free memory with NaN (L4 #5).

    A candidate that returns torch.empty(...) without writing it could otherwise get back a
    freed buffer that still holds a correct answer. Best effort: it covers one block of at least
    min_bytes, which the allocator splits for later requests. No-op without CUDA.
    """
    import torch

    if not torch.cuda.is_available():
        return
    torch.cuda.empty_cache()
    free, _ = torch.cuda.mem_get_info()
    nbytes = min(max(min_bytes, 0), int(free * 0.5))
    if nbytes >= 4:
        torch.empty(nbytes // 4, dtype=torch.float32, device="cuda").fill_(float("nan"))
    torch.cuda.synchronize()


def run_case(ref_model, cand_model, case: Case, atol: float, rtol: float) -> CaseCheck:
    """Run the candidate, then the reference (fp32 and fp64), on fresh copies of the inputs.

    The candidate runs first, into NaN-poisoned memory, so no reference output exists yet for it
    to find or inherit (L4 #1 and #5).
    """
    import torch

    poison_free_memory()
    with strict_fp32(), torch.no_grad():
        try:
            cand = _leaves(cand_model(*_clone(case.inputs)))
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            msg = _excerpt(traceback.format_exc(), case.hidden)
            return CaseCheck(Status.RUNTIME_ERROR, float("nan"), float("nan"), None, "exact", msg)
        # Re-assert strict fp32: a candidate that turned TF32 on in forward would otherwise
        # drag the reference into TF32 too, and the two would agree (seen on an H100).
        with strict_fp32():
            ref32 = _leaves(ref_model(*_clone(case.inputs)))
    ref64 = _reference_fp64(ref_model, case.inputs)

    if len(cand) != len(ref32):
        msg = f"returned {len(cand)} tensors, expected {len(ref32)}"
        return CaseCheck(Status.WRONG_SHAPE, float("nan"), float("nan"), None, "exact", msg)

    worst: CaseCheck | None = None
    for i, (c, r32) in enumerate(zip(cand, ref32, strict=True)):
        r64 = ref64[i] if ref64 is not None and r32.is_floating_point() else None
        check = compare(c, r32, r64, atol, rtol, case.hidden)
        if not check.passed:
            return check
        if worst is None or check.max_err > worst.max_err:
            worst = check
    return worst


def check(ref_model, cand_model, cases: Sequence[Case], atol: float, rtol: float):
    """Run cases in order and stop at the first failure."""
    import torch

    checks: list[CaseCheck] = []
    for case in cases:
        result = run_case(ref_model, cand_model, case, atol, rtol)
        checks.append(result)
        if not result.passed:
            shapes = [list(x.shape) for x in case.inputs if isinstance(x, torch.Tensor)]
            finite = [c.max_err for c in checks if c.max_err == c.max_err]
            return CorrectnessResult(
                status=result.status,
                max_err=max(finite) if finite else None,
                failing_shape_class=case.shape_class,
                failing_shape=shapes,
                error_excerpt=result.message,
                cases=checks,
            )
    return CorrectnessResult(None, max((c.max_err for c in checks), default=None), cases=checks)
