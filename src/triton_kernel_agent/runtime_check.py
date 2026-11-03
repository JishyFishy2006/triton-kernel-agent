"""Runtime anti-cheat (B8), judged in the trusted parent from what the worker reports.

The candidate runs in a worker process (worker.py, Y4 isolation), so everything it reports
about itself is a claim. The checks that decide a verdict use only what the parent sees:
- input_insensitive (L4 #4): new values are written into the SAME buffers and the candidate
  runs again. Its output is compared with the reference's for those values; a wrong answer
  equal to the previous output is caching (by shape, pointer, version or a sample of values).
- outputs of timed calls, and physically impossible speed: benchmark.py.
Best effort, from the worker's own report (a determined candidate can fake these, so they are
backed by the static scan and the checks above):
- aten_compute: aten ops beyond allocation, views and copies, or GPU kernels that are not the
  candidate's @triton.jit kernels (cuBLAS/cuDNN reached around aten).
- output_aliases_input (L4 #5): an output shares memory with an input.
"""

import ast
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

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


def audit_findings(ops: Sequence[str], kernels: Sequence[str], source: str) -> list[RuntimeFinding]:
    """aten_compute findings from one audited forward's ops and GPU kernel names."""
    findings = []
    if bad := sorted({op for op in ops if op not in ALLOWED_ATEN}):
        findings.append(
            RuntimeFinding(CheatFlag.ATEN_COMPUTE, f"PyTorch ops in forward: {', '.join(bad)}")
        )
    allowed_kernels = triton_kernel_names(source)
    foreign = sorted(
        {
            k
            for k in kernels
            if not any(k == n or k.startswith(n + "_") for n in allowed_kernels)
            and not any(s in k.lower() for s in ALLOWED_KERNEL_SUBSTRINGS)
        }
    )
    if foreign:
        findings.append(
            RuntimeFinding(
                CheatFlag.ATEN_COMPUTE, f"non-Triton GPU kernels: {', '.join(foreign)[:500]}"
            )
        )
    return findings


def unchanged(before: Sequence[Any], after: Sequence[Any]) -> bool:
    """True if two output lists are bitwise equal (shapes included)."""
    import torch

    return len(before) == len(after) and all(
        a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b)
        for a, b in zip(before, after, strict=True)
    )


def load_candidate(source: str, name: str = "candidate") -> Any:
    """Import candidate source as a real module (Triton's @jit reads source from the file).

    Written to a fresh temporary directory. Only for tools that run a kernel that already
    passed (modal_ncu.py); the harness never imports candidate code (worker.py does).
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
