"""The harness pipeline (tech spec, "Harness"): one candidate in, one HarnessResult out.

Runs inside the Modal sandbox (sandbox.py), one fresh process per eval:
    python -m triton_kernel_agent.harness  < request JSON on stdin
and prints the result on a line prefixed with the request's nonce, so nothing the candidate
prints can pose as the result.

Stages, in order; the first that fails sets the status:
1. compile:     parse the source (syntax error -> compile_error)
2. static:      AST scan (B7) -> static_cheat
3. import:      HarnessGuard first, then import the candidate (errors -> compile_error)
4. correctness: the visible case, then every hidden case (B5, B6)
5. runtime:     aten/kernel audit, same-buffer rerun, aliasing, tampering (B8) -> runtime_cheat
6. benchmark:   candidate vs eager vs torch.compile, re-checked after timing (B9)
7. roofline:    % of peak from the task's bytes and FLOPs (B10)
"""

import dataclasses
import json
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Any

from triton_kernel_agent.harness_result import (
    ERROR_EXCERPT_CHARS,
    CheatFlag,
    HarnessResult,
    ShapeClass,
    Status,
)

RESULT_PREFIX = "HARNESS_RESULT"


@dataclass(frozen=True)
class EvalRequest:
    task_id: str
    task_source: str
    candidate_source: str
    atol: float
    rtol: float
    bytes: int
    flops: int
    input_domain: str = "any"
    nonce: str = ""
    # {"eager_runs_ms", "compile_runs_ms"} from an earlier eval of this task (B9).
    baselines: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvalRequest":
        return cls(**data)


@dataclass
class _Clock:
    start: float = field(default_factory=time.monotonic)

    def seconds(self) -> float:
        return time.monotonic() - self.start


def _result(req: EvalRequest, clock: _Clock, status: Status, **kwargs: Any) -> HarnessResult:
    return HarnessResult(task_id=req.task_id, status=status, gpu_seconds=clock.seconds(), **kwargs)


def _tail(text: str) -> str:
    return text[-ERROR_EXCERPT_CHARS:]


def evaluate(req: EvalRequest) -> HarnessResult:
    clock = _Clock()
    try:
        return _evaluate(req, clock)
    except SystemExit:  # the candidate called exit(): its failure, not the harness's
        excerpt = _tail(traceback.format_exc())
        return _result(req, clock, Status.RUNTIME_ERROR, error_excerpt=excerpt)
    except Exception:  # a bug in the harness itself; never shown to the agent
        excerpt = _tail(traceback.format_exc())
        return _result(req, clock, Status.HARNESS_ERROR, error_excerpt=excerpt)


def _evaluate(req: EvalRequest, clock: _Clock) -> HarnessResult:
    from triton_kernel_agent.static_check import scan

    # 1-2. compile and static scan: nothing of the candidate's has run yet.
    try:
        report = scan(req.candidate_source)
    except SyntaxError as e:
        return _result(req, clock, Status.COMPILE_ERROR, error_excerpt=f"SyntaxError: {e}")
    if not report.clean:
        return _result(
            req, clock, Status.STATIC_CHEAT, cheat_flags=report.flags,
            error_excerpt=_tail(report.summary()),
        )  # fmt: skip

    import torch

    from triton_kernel_agent.correctness import Case, check
    from triton_kernel_agent.hidden_shapes import HiddenCase, build_case, generate
    from triton_kernel_agent.reference import INIT_SEED, INPUT_SEED
    from triton_kernel_agent.runtime_check import HarnessGuard, check_runtime, load_candidate

    # 3. import, with the harness snapshotted first (L4 #3).
    guard = HarnessGuard()
    try:
        model_new = load_candidate(req.candidate_source, "candidate").ModelNew
    except Exception:
        return _result(
            req, clock, Status.COMPILE_ERROR, error_excerpt=_tail(traceback.format_exc())
        )

    def candidate(init: list[Any]):
        torch.manual_seed(INIT_SEED)
        return model_new(*init).cuda()

    # 4. correctness: the visible case first (shapes may be shown), then the hidden ones.
    visible = HiddenCase(ShapeClass.VISIBLE, {}, "default", INPUT_SEED)
    hidden = generate(req.task_id, req.task_source, input_domain=req.input_domain)
    ref, init, inputs = build_case(req.task_source, visible, "cuda")
    try:
        cand = candidate(init)
    except Exception:
        return _result(
            req, clock, Status.RUNTIME_ERROR, error_excerpt=_tail(traceback.format_exc())
        )
    visible_check = check(ref, cand, [Case(inputs, ShapeClass.VISIBLE, hidden=False)],
                          req.atol, req.rtol)  # fmt: skip
    if not visible_check.passed:
        return _failed(req, clock, visible_check)
    max_err = visible_check.max_err
    for case in hidden:
        h_ref, h_init, h_inputs = build_case(req.task_source, case, "cuda")
        result = check(h_ref, candidate(h_init), [Case(h_inputs, case.shape_class)],
                       req.atol, req.rtol)  # fmt: skip
        if not result.passed:
            return _failed(req, clock, result)
        max_err = max(max_err, result.max_err)
        del h_ref, h_inputs
        torch.cuda.empty_cache()

    # 5. runtime checks on the first hidden case, refilled with a second sample of it.
    first = hidden[0]
    r_ref, r_init, r_inputs = build_case(req.task_source, first, "cuda")
    second = dataclasses.replace(first, seed=first.seed + 1)
    _, _, fresh = build_case(req.task_source, second, "cuda")
    runtime = check_runtime(r_ref, candidate(r_init), r_inputs, fresh, req.candidate_source, guard)
    if not runtime.clean:
        return _result(
            req, clock, Status.RUNTIME_CHEAT, max_err=max_err, cheat_flags=runtime.flags,
            error_excerpt=_tail("\n".join(f.message for f in runtime.findings)),
            extra={"gpu_kernels": sorted(set(runtime.gpu_kernels))},
        )  # fmt: skip
    del r_ref, r_inputs, fresh

    # 6-7. benchmark on the visible inputs (fresh candidate instance), then roofline.
    from triton_kernel_agent.benchmark import benchmark
    from triton_kernel_agent.roofline import peak_for, roofline

    bench = benchmark(ref, candidate(init), inputs, req.atol, req.rtol, baselines=req.baselines)
    if not bench.consistent:
        return _result(
            req, clock, Status.RUNTIME_CHEAT, max_err=max_err,
            cheat_flags=[CheatFlag.CHANGED_AFTER_CHECKS],
            error_excerpt=_tail(bench.post_check.message), extra={"benchmark": bench.extra},
        )  # fmt: skip
    roof = roofline(req.bytes, req.flops, bench.timing.kernel_ms,
                    peak_for(torch.cuda.get_device_name()))  # fmt: skip
    return _result(
        req, clock, Status.CORRECT, max_err=max_err, timing=bench.timing,
        pct_peak_bandwidth=roof.pct_peak_bandwidth, pct_peak_flops=roof.pct_peak_flops,
        extra={"benchmark": bench.extra, "roofline": roof.as_dict()},
    )  # fmt: skip


def _failed(req: EvalRequest, clock: _Clock, result) -> HarnessResult:
    fields = result.harness_fields()
    # Triton compile errors surface on the first launch; report them as compile errors.
    if fields["status"] == Status.RUNTIME_ERROR and "CompilationError" in fields["error_excerpt"]:
        fields["status"] = Status.COMPILE_ERROR
    return _result(req, clock, **fields)


def main() -> None:
    request = EvalRequest.from_dict(json.loads(sys.stdin.read()))
    result = evaluate(request)
    sys.stdout.write(f"\n{RESULT_PREFIX}:{request.nonce}:{result.to_json()}\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
