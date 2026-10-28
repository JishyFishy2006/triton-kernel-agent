"""HarnessResult: the one JSON result every harness eval returns (tech spec, "Harness").

Every harness module (B5-B10) reports through this type, and B11 joins them, so it is written
first and changed only with a HARNESS_VERSION bump. One result per candidate eval; it maps
one-to-one onto a row of the evals table (B12). The full field reference is in
docs/harness_result.md.
"""

import json
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

# Results are only compared within one harness version (Y5 tags harness-v1). v0 is the
# version under construction; bump on every change to fields or categories from v1 on.
HARNESS_VERSION = "v0"
# Error text sent back is cut to this many characters, keeping the end (where tracebacks point).
ERROR_EXCERPT_CHARS = 2000


class Status(StrEnum):
    """Outcome category, in pipeline order: the first stage that fails sets it."""

    COMPILE_ERROR = "compile_error"  # syntax, import or Triton compile error
    STATIC_CHEAT = "static_cheat"  # AST scan flagged it (B7)
    RUNTIME_ERROR = "runtime_error"  # raised, or illegal memory access, while running
    TIMEOUT = "timeout"  # killed at the wall-clock limit (B11)
    WRONG_SHAPE = "wrong_shape"  # output shape or dtype differs from the reference (B5)
    NAN_INF = "nan_inf"  # output has NaN or Inf where the reference does not (B5)
    MISMATCH = "mismatch"  # values outside tolerance (B5)
    RUNTIME_CHEAT = "runtime_cheat"  # profiler, input-sensitivity or aliasing check failed (B8)
    CORRECT = "correct"  # passed every check and was benchmarked (B9, B10)
    HARNESS_ERROR = "harness_error"  # our bug, not the candidate's; never shown to the agent

    @property
    def is_cheat(self) -> bool:
        return self in (Status.STATIC_CHEAT, Status.RUNTIME_CHEAT)


class CheatFlag(StrEnum):
    # Static (B7): found in the source before it runs.
    TORCH_COMPUTE_CALL = "torch_compute_call"  # torch.matmul, F.softmax, x @ w, ... outside kernels
    REFERENCE_IMPORT = "reference_import"  # imports the reference, or anything off the allowlist
    FILE_OR_NETWORK = "file_or_network"  # open(), sockets, subprocess, ...
    BACKEND_FLAGS = "backend_flags"  # touches torch.backends (TF32 and friends)
    TF32_DOT = "tf32_dot"  # tl.dot without input_precision="ieee" (Y2 decision 1)
    INTROSPECTION = "introspection"  # gc, inspect, frames, __globals__: hunting the reference
    MONKEYPATCH = "monkeypatch"  # assigns module attributes, setattr, exec/eval, builtins
    SIDE_STREAM = "side_stream"  # launches on another CUDA stream or a CUDA graph
    HARDCODED_DATA = "hardcoded_data"  # large literal arrays or blobs: precomputed answers
    # Runtime (B8): seen while it runs.
    ATEN_COMPUTE = "aten_compute"  # aten compute ops, or cuBLAS/cuDNN kernels, in forward
    INPUT_INSENSITIVE = "input_insensitive"  # output did not change when the input did
    OUTPUT_ALIASES_INPUT = "output_aliases_input"  # output shares memory with an input
    HARNESS_TAMPERED = "harness_tampered"  # timers, checks or TF32 flags changed by the run

    @property
    def is_static(self) -> bool:
        return self in _STATIC_FLAGS


_STATIC_FLAGS = frozenset(
    {
        CheatFlag.TORCH_COMPUTE_CALL,
        CheatFlag.REFERENCE_IMPORT,
        CheatFlag.FILE_OR_NETWORK,
        CheatFlag.BACKEND_FLAGS,
        CheatFlag.TF32_DOT,
        CheatFlag.INTROSPECTION,
        CheatFlag.MONKEYPATCH,
        CheatFlag.SIDE_STREAM,
        CheatFlag.HARDCODED_DATA,
    }
)


class ShapeClass(StrEnum):
    """Which kind of test case failed. The agent sees this, never the exact hidden shape."""

    VISIBLE = "visible"  # the task's own get_inputs() shapes
    NON_POW2 = "non_pow2"
    SIZE_ONE = "size_one"
    LARGE = "large"
    BATCH = "batch"  # batch size varied (training-mode BatchNorm, Y2)
    INPUT_DISTRIBUTION = "input_distribution"  # same shape, randn or shifted inputs (Y2)


@dataclass(frozen=True)
class Timing:
    """Median times in ms from do_bench (B9); all three on the same inputs."""

    kernel_ms: float
    eager_ms: float
    compile_ms: float

    @property
    def speedup_eager(self) -> float:
        return self.eager_ms / self.kernel_ms

    @property
    def speedup_compile(self) -> float:
        return self.compile_ms / self.kernel_ms


@dataclass(frozen=True)
class HarnessResult:
    task_id: str
    status: Status
    # Last ERROR_EXCERPT_CHARS of the error or traceback; empty when there is none.
    # On hidden cases it must not contain shapes or indices: B15 may show it to the agent.
    error_excerpt: str = ""
    # Largest |candidate - reference| over all cases checked (B5); None if it never ran.
    max_err: float | None = None
    failing_shape_class: ShapeClass | None = None
    # The exact failing shape, for our logs only. B15 must never put it in a hint.
    failing_shape: list[list[int]] | None = None
    cheat_flags: tuple[CheatFlag, ...] = ()
    timing: Timing | None = None  # only for CORRECT
    pct_peak_bandwidth: float | None = None  # B10, from the task's bytes
    pct_peak_flops: float | None = None  # B10, from the task's FLOPs; None if FLOPs are 0
    gpu_seconds: float = 0.0  # wall time on the GPU container, for cost
    harness_version: str = HARNESS_VERSION
    extra: dict[str, Any] = field(default_factory=dict)  # per-module detail, not shown to agent

    def __post_init__(self) -> None:
        # Coerce so results built from JSON or by hand compare equal.
        object.__setattr__(self, "status", Status(self.status))
        if self.failing_shape_class is not None:
            object.__setattr__(self, "failing_shape_class", ShapeClass(self.failing_shape_class))
        flags = tuple(sorted({CheatFlag(f) for f in self.cheat_flags}))
        object.__setattr__(self, "cheat_flags", flags)
        if isinstance(self.timing, dict):
            object.__setattr__(self, "timing", Timing(**self.timing))
        if len(self.error_excerpt) > ERROR_EXCERPT_CHARS:
            object.__setattr__(self, "error_excerpt", self.error_excerpt[-ERROR_EXCERPT_CHARS:])
        self._check()

    def _check(self) -> None:
        s = self.status
        if s.is_cheat != bool(self.cheat_flags):
            raise ValueError(f"{s} needs cheat flags exactly when it is a cheat status")
        if s == Status.STATIC_CHEAT and not all(f.is_static for f in self.cheat_flags):
            raise ValueError("static_cheat may only carry static flags")
        if s == Status.RUNTIME_CHEAT and all(f.is_static for f in self.cheat_flags):
            raise ValueError("runtime_cheat needs at least one runtime flag")
        if (s == Status.CORRECT) != (self.timing is not None):
            raise ValueError("timing is required for correct results and only for them")
        perf = (self.pct_peak_bandwidth, self.pct_peak_flops)
        if s != Status.CORRECT and any(p is not None for p in perf):
            raise ValueError("percent of peak is only reported for correct results")
        failed_a_case = s in (Status.WRONG_SHAPE, Status.NAN_INF, Status.MISMATCH)
        if failed_a_case and self.failing_shape_class is None:
            raise ValueError(f"{s} must say which shape class failed")
        if not failed_a_case and (self.failing_shape_class or self.failing_shape):
            raise ValueError(f"{s} has no failing shape")

    @property
    def is_correct(self) -> bool:
        return self.status == Status.CORRECT

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HarnessResult":
        return cls(**data)

    @classmethod
    def from_json(cls, text: str) -> "HarnessResult":
        return cls.from_dict(json.loads(text))
