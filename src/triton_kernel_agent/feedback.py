"""Feedback formatter (B15): turn a HarnessResult into the hint the model sees next.

Never leaks hidden test cases: the exact failing shape is never included, only its class
(non-power-of-2, size 1, ...), and as a last line of defence every number in the hint that
equals a dimension of the hidden failing shape is scrubbed. Error text from hidden cases is
already redacted by the correctness checker (B5). Harness errors show nothing internal.
"""

import re

from triton_kernel_agent.harness_result import CheatFlag, HarnessResult, ShapeClass, Status

SHAPE_CLASS_HINTS = {
    ShapeClass.VISIBLE: "the visible inputs from get_inputs()",
    ShapeClass.NON_POW2: (
        "hidden inputs whose sizes are not powers of 2 and not multiples of your block sizes: "
        "mask every load and store, and handle partial tiles at every edge"
    ),
    ShapeClass.SIZE_ONE: (
        "hidden inputs with a dimension of size 1: check reductions, strides and grid sizes "
        "when a dimension collapses"
    ),
    ShapeClass.BATCH: (
        "hidden inputs with a different batch size: nothing may assume the visible batch size"
    ),
    ShapeClass.LARGE: (
        "a large hidden input: watch for int32 overflow in pointer offsets (cast to int64) "
        "and for grid limits"
    ),
    ShapeClass.INPUT_DISTRIBUTION: (
        "hidden inputs from a different distribution (negative, wider and shifted values): "
        "do not assume positive or small inputs; keep exp/log/softmax numerically stable"
    ),
}

CHEAT_HINTS = {
    CheatFlag.TORCH_COMPUTE_CALL: "PyTorch computes part of the result outside your Triton "
    "kernels. Do all math in @triton.jit kernels; the wrapper may only allocate, reshape "
    "and launch.",
    CheatFlag.REFERENCE_IMPORT: "An import is not on the allowlist (torch, torch.nn, "
    "torch.nn.functional, triton, triton.language, math, typing, functools, dataclasses, "
    "itertools, collections, numbers, enum).",
    CheatFlag.FILE_OR_NETWORK: "Files, processes and the network are off limits.",
    CheatFlag.BACKEND_FLAGS: "torch.backends and TF32 settings must not be touched.",
    CheatFlag.TF32_DOT: 'Every tl.dot must pass input_precision="ieee" (strict fp32).',
    CheatFlag.INTROSPECTION: "Introspection (gc, inspect, sys, frames, __globals__) is not "
    "allowed.",
    CheatFlag.MONKEYPATCH: "Changing other modules at runtime (attribute assignment, setattr, "
    "exec, eval) is not allowed.",
    CheatFlag.SIDE_STREAM: "Launch on the current CUDA stream only; no streams or CUDA graphs.",
    CheatFlag.HARDCODED_DATA: "Large literal data tables are not allowed: compute the result.",
    CheatFlag.ATEN_COMPUTE: "PyTorch or cuBLAS/cuDNN kernels ran during forward. Only your own "
    "Triton kernels may compute.",
    CheatFlag.INPUT_INSENSITIVE: "The output did not change when the input values changed: "
    "do not cache results between calls.",
    CheatFlag.OUTPUT_ALIASES_INPUT: "The output shares memory with an input: allocate a new "
    "output tensor and never write into the inputs.",
    CheatFlag.HARNESS_TAMPERED: "Timers, checks or TF32 settings were changed while your code ran.",
    CheatFlag.CHANGED_AFTER_CHECKS: "The output was wrong after benchmarking: the kernel must "
    "compute the same correct result on every call.",
}


def _scrub(text: str, result: HarnessResult) -> str:
    """Remove hidden sizes: every multi-digit number equal to a dimension of the hidden failing
    shape, and every multi-digit number at all after a crash on a hidden case (no shape is
    recorded for crashes). Single digits (size 1, 2-D) say nothing about a shape."""
    if result.extra.get("case") == "hidden" and result.status in (
        Status.RUNTIME_ERROR,
        Status.TIMEOUT,
    ):
        return re.sub(r"\d{2,}", "<n>", text)
    if not result.failing_shape or result.failing_shape_class == ShapeClass.VISIBLE:
        return text
    dims = {str(d) for shape in result.failing_shape for d in shape if d >= 10}
    return re.sub(r"\d+", lambda m: "<n>" if m.group(0) in dims else m.group(0), text)


def _excerpt(result: HarnessResult) -> str:
    text = result.error_excerpt.strip()
    return f"\n\nDetails:\n{text}" if text else ""


def _flags(result: HarnessResult) -> str:
    return "\n".join(f"- {CHEAT_HINTS[f]}" for f in result.cheat_flags)


def _correct(result: HarnessResult) -> str:
    t = result.timing
    lines = [
        f"Correct on every test. Kernel {t.kernel_ms:.3f} ms vs PyTorch eager {t.eager_ms:.3f} ms "
        f"({t.speedup_eager:.2f}x) and torch.compile {t.compile_ms:.3f} ms "
        f"({t.speedup_compile:.2f}x)."
    ]
    roof = result.extra.get("roofline") or {}
    if result.pct_peak_bandwidth is not None:
        peak = f"It reaches {result.pct_peak_bandwidth:.0f}% of peak memory bandwidth"
        if result.pct_peak_flops is not None:
            peak += f" and {result.pct_peak_flops:.0f}% of fp32 peak FLOP rate"
        if roof.get("bound"):
            peak += f"; the task is {roof['bound']}-bound"
        lines.append(peak + ".")
    if t.speedup_eager <= 1:
        lines.append(
            "Make it faster than eager without breaking correctness: fuse operations to avoid "
            "extra passes over memory, pick block sizes (or @triton.autotune) for the bound "
            "resource, and keep loads coalesced."
        )
    return "\n".join(lines)


def format_feedback(result: HarnessResult) -> str:
    s = result.status
    if s == Status.CORRECT:
        text = _correct(result)
    elif s == Status.COMPILE_ERROR:
        text = "Your code failed to compile or import. Fix the error below." + _excerpt(result)
    elif s == Status.STATIC_CHEAT:
        text = (
            "Rejected before running: the code breaks these rules.\n"
            + _flags(result)
            + _excerpt(result)
        )
    elif s == Status.TIMEOUT:
        text = (
            "The evaluation did not finish within its time limit: look for an infinite loop, a "
            "kernel that never terminates, or a launch that is far too slow."
        )
    elif s == Status.RUNTIME_ERROR:
        text = (
            "Your code crashed while running. Common causes: out-of-bounds loads or stores "
            "(mask them), wrong strides in pointer arithmetic, or a grid that does not match the "
            "data." + _excerpt(result)
        )
    elif s in (Status.WRONG_SHAPE, Status.NAN_INF, Status.MISMATCH):
        what = {
            Status.WRONG_SHAPE: "The output has the wrong shape or dtype",
            Status.NAN_INF: "The output has NaN or Inf where the reference is finite",
            Status.MISMATCH: "The output values are wrong",
        }[s]
        if s == Status.MISMATCH and result.max_err is not None:
            what += f" (max abs error {result.max_err:.3g})"
        cls = SHAPE_CLASS_HINTS.get(result.failing_shape_class, "one of the test inputs")
        text = f"{what} on {cls}." + _excerpt(result)
    elif s == Status.RUNTIME_CHEAT:
        text = "Rejected while running: these checks failed.\n" + _flags(result)
    else:  # HARNESS_ERROR: our problem; show nothing internal
        text = (
            "The evaluation failed for reasons unrelated to your code. Resubmit the same approach."
        )
    return _scrub(text, result)
