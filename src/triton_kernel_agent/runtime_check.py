"""Runtime anti-cheat (B8): watch the candidate while it runs.

- aten_compute: every aten op the candidate's forward dispatches is recorded; anything beyond
  allocation, views and copies is PyTorch doing the math. The profiler also lists GPU kernels:
  any that isn't one of the candidate's @triton.jit kernels (or a fill/copy) is flagged, which
  catches cuBLAS/cuDNN reached around aten (ctypes, cupy, a wrapper).
- input_insensitive (L4 #4): new input values are written into the SAME buffers and the
  candidate runs again. If the reference's output changes and the candidate's doesn't, it is
  caching (by shape, by pointer, or by call).
- output_aliases_input (L4 #5): an output shares memory with an input (identity, in-place).
- harness_tampered (L4 #3): timers, checks and TF32 settings are snapshotted before the
  candidate's code is even imported; any change afterwards is flagged.

Runs where torch is installed; torch is imported lazily.
"""

import ast
import importlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from triton_kernel_agent.correctness import (
    _clone,
    _leaves,
    poison_free_memory,
    strict_fp32,
    tf32_enabled,
)
from triton_kernel_agent.harness_result import CheatFlag

# aten ops a Triton wrapper may use: allocation, views, copies, metadata. Not math.
ALLOWED_ATEN = frozenset(
    {
        "empty", "empty_like", "empty_strided", "new_empty", "new_empty_strided", "zeros",
        "zeros_like", "new_zeros", "ones", "ones_like", "full", "full_like", "new_full",
        "arange", "fill_", "zero_", "copy_", "clone", "contiguous", "_to_copy", "to", "detach",
        "alias", "lift_fresh", "lift_fresh_copy", "view", "_unsafe_view", "reshape",
        "as_strided", "permute", "transpose", "t", "expand", "unsqueeze", "squeeze", "select",
        "slice", "narrow", "split", "split_with_sizes", "chunk", "unbind", "cat", "stack",
        "flatten", "unflatten", "view_as", "expand_as", "resize_", "set_", "_reshape_alias",
        "_local_scalar_dense", "is_same_size", "sym_size", "sym_stride", "sym_numel",
        "sym_storage_offset", "result_type",
    }
)  # fmt: skip
# Non-Triton GPU kernels that only move or fill memory (allocations, copies, cat).
ALLOWED_KERNEL_SUBSTRINGS = ("fill", "copy", "memcpy", "memset", "catarray", "cat_")

# What the harness relies on to time and check. Replacing any of these is tampering.
GUARDED = (
    ("time", "perf_counter"),
    ("time", "perf_counter_ns"),
    ("time", "time"),
    ("time", "monotonic"),
    ("torch", "allclose"),
    ("torch", "equal"),
    ("torch", "isfinite"),
    ("torch.cuda", "synchronize"),
    ("torch.cuda", "Event"),
    ("torch.cuda", "current_stream"),
    ("torch.cuda.Event", "record"),
    ("torch.cuda.Event", "elapsed_time"),
    ("torch.Tensor", "__eq__"),
    ("torch.Tensor", "abs"),
    ("torch.Tensor", "max"),
    ("torch.profiler", "profile"),
    ("triton.testing", "do_bench"),
    ("triton_kernel_agent.correctness", "compare"),
    ("triton_kernel_agent.correctness", "run_case"),
    ("triton_kernel_agent.correctness", "check"),
    ("triton_kernel_agent.correctness", "strict_fp32"),
    ("triton_kernel_agent.runtime_check", "check_runtime"),
)


@dataclass(frozen=True)
class RuntimeFinding:
    flag: CheatFlag
    message: str


@dataclass(frozen=True)
class RuntimeReport:
    findings: tuple[RuntimeFinding, ...]
    aten_ops: tuple[str, ...] = ()  # every aten op forward dispatched
    gpu_kernels: tuple[str, ...] = ()  # every GPU kernel forward launched
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def flags(self) -> tuple[CheatFlag, ...]:
        return tuple(sorted({f.flag for f in self.findings}))

    @property
    def clean(self) -> bool:
        return not self.findings


class HarnessGuard:
    """Snapshot of GUARDED objects. Take it before the candidate's source is executed."""

    def __init__(self) -> None:
        self.saved: dict[tuple[str, str], tuple[Any, Any]] = {}
        for module, attr in GUARDED:
            owner = _resolve(module)
            if owner is not None and hasattr(owner, attr):
                self.saved[(module, attr)] = (owner, getattr(owner, attr))
        self.tf32 = _tf32_state()

    def changed(self) -> list[str]:
        changed = [
            f"{module}.{attr}"
            for (module, attr), (owner, original) in self.saved.items()
            if getattr(owner, attr, None) is not original
        ]
        if _tf32_state() != self.tf32:
            changed.append("TF32 settings")
        return changed

    def restore(self) -> None:
        """Put everything back (tests run many candidates in one process; B11 uses fresh ones)."""
        for (_, attr), (owner, original) in self.saved.items():
            setattr(owner, attr, original)


def _resolve(dotted: str) -> Any:
    try:
        return importlib.import_module(dotted)
    except ImportError:
        module, _, attr = dotted.rpartition(".")
        try:
            return getattr(importlib.import_module(module), attr)
        except (ImportError, AttributeError):
            return None


def _tf32_state() -> Any:
    try:
        return tf32_enabled()
    except RuntimeError as e:  # mixing the legacy and new TF32 APIs makes torch raise
        return f"error: {e}"


def load_candidate(source: str, name: str = "candidate") -> Any:
    """Import candidate source as a real module (Triton's @jit reads source from the file).

    Written to a fresh temporary directory; the caller takes a HarnessGuard first.
    """
    import importlib.util
    import sys
    import tempfile
    from pathlib import Path

    path = Path(tempfile.mkdtemp(prefix="tka_")) / f"{name}.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses and pickling look modules up by name
    spec.loader.exec_module(module)
    return module


def triton_kernel_names(source: str) -> set[str]:
    """Names of the candidate's @triton.jit functions: the kernels it may launch."""
    names = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.FunctionDef):
            for dec in node.decorator_list:
                target = dec.func if isinstance(dec, ast.Call) else dec
                if isinstance(target, ast.Attribute) and target.attr in ("jit", "autotune"):
                    names.add(node.name)
    return names


def _same_memory(a, b) -> bool:
    if a.numel() == 0 or b.numel() == 0:
        return False
    sa, sb = a.untyped_storage(), b.untyped_storage()
    a0, b0 = sa.data_ptr(), sb.data_ptr()
    return a0 < b0 + sb.nbytes() and b0 < a0 + sa.nbytes()


def _run_audited(cand_model, inputs: Sequence[Any]):
    """Run forward once, recording aten ops and GPU kernels."""
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode

    class AtenAudit(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.ops: list[str] = []

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            self.ops.append(func.overloadpacket.__name__)
            return func(*args, **(kwargs or {}))

    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(activities=activities) as prof:
        with AtenAudit() as audit:
            outputs = _leaves(cand_model(*inputs))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    kernels = [
        e.name
        for e in prof.events()
        if getattr(e, "device_type", None) == torch.autograd.DeviceType.CUDA
    ]
    return outputs, audit.ops, kernels


def check_runtime(
    ref_model,
    cand_model,
    inputs: Sequence[Any],
    fresh_inputs: Sequence[Any],
    source: str,
    guard: HarnessGuard | None = None,
) -> RuntimeReport:
    """Run the candidate under audit, then again on new values written into the same buffers.

    inputs and fresh_inputs are two samples of the same shapes (e.g. two hidden cases' seeds).
    """
    import torch

    findings: list[RuntimeFinding] = []
    allowed_kernels = triton_kernel_names(source)

    def flag(f: CheatFlag, message: str) -> None:
        findings.append(RuntimeFinding(f, message))

    poison_free_memory()
    buffers = _clone(inputs)
    with strict_fp32(), torch.no_grad():
        out1, ops, kernels = _run_audited(cand_model, buffers)
        out1_values = [o.clone() for o in out1]

        if bad := sorted({op for op in ops if op not in ALLOWED_ATEN}):
            flag(CheatFlag.ATEN_COMPUTE, f"PyTorch ops in forward: {', '.join(bad)}")
        foreign = sorted(
            {
                k
                for k in kernels
                if not any(k == n or k.startswith(n + "_") for n in allowed_kernels)
                and not any(s in k.lower() for s in ALLOWED_KERNEL_SUBSTRINGS)
            }
        )
        if foreign:
            flag(CheatFlag.ATEN_COMPUTE, f"non-Triton GPU kernels: {', '.join(foreign)[:500]}")

        tensor_inputs = [x for x in buffers if isinstance(x, torch.Tensor)]
        if any(_same_memory(o, x) for o in out1 for x in tensor_inputs):
            flag(CheatFlag.OUTPUT_ALIASES_INPUT, "an output shares memory with an input")

        # Same buffers, new values: a cache keyed on shape, pointer or call count goes stale.
        ref1 = [r.clone() for r in _leaves(ref_model(*_clone(inputs)))]
        for buf, new in zip(buffers, fresh_inputs, strict=True):
            if isinstance(buf, torch.Tensor):
                buf.copy_(new)
        out2 = _leaves(cand_model(*buffers))
        ref2 = _leaves(ref_model(*_clone(fresh_inputs)))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ref_changed = any(not torch.equal(a, b) for a, b in zip(ref1, ref2, strict=True))
        cand_same = len(out1_values) == len(out2) and all(
            a.shape == b.shape and torch.equal(a, b) for a, b in zip(out1_values, out2, strict=True)
        )
        if ref_changed and cand_same:
            flag(
                CheatFlag.INPUT_INSENSITIVE,
                "output did not change when new values were written into the same inputs",
            )

        # strict_fp32() restores the flags on exit, so a change made in forward is seen here.
        changed = [] if _tf32_state() is False else ["TF32 settings (during forward)"]
    # The guard sees what the candidate's import changed (module-level monkeypatching).
    changed += guard.changed() if guard is not None else []
    if changed:
        flag(CheatFlag.HARNESS_TAMPERED, f"changed by the candidate: {', '.join(changed)}")

    return RuntimeReport(
        tuple(findings),
        aten_ops=tuple(ops),
        gpu_kernels=tuple(kernels),
        extra={"allowed_kernels": sorted(allowed_kernels)},
    )
