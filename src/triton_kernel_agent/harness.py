"""The harness pipeline (tech spec, "Harness"): one candidate in, one HarnessResult out.

Runs inside the Modal sandbox (sandbox.py), one fresh process per eval:
    python -m triton_kernel_agent.harness  < request JSON on stdin
and prints the result on a line prefixed with the request's nonce.

This process is the trusted parent: it never imports candidate code (Y4 isolation). The
candidate runs in worker processes (worker.py, isolation.py) as an unprivileged user, with no
access to this process, its stdout, the request (and its nonce), the reference model, or the
harness files. The parent builds every input, computes every reference output, compares the
outputs it reads back from files, and times calls with its own clock. Y4's red team broke the
previous single-process design 26 ways, all through that shared process.

Stages, in order; the first that fails sets the status:
1. compile:     parse the source (syntax error -> compile_error)
2. static:      AST scan (B7) -> static_cheat
3. import:      the worker imports the candidate (errors -> compile_error)
4. correctness: the visible case, then every hidden case (B5, B6)
5. runtime:     same-buffer rerun, aten/kernel audit, aliasing (B8) -> runtime_cheat
6. benchmark:   eager and torch.compile in a clean worker, then the candidate in a fresh one;
                its last timed output re-checked, its speed checked against physics (B9)
7. roofline:    % of peak from the task's bytes and FLOPs (B10)
"""

import dataclasses
import json
import shutil
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from triton_kernel_agent.harness_result import (
    ERROR_EXCERPT_CHARS,
    CheatFlag,
    HarnessResult,
    ShapeClass,
    Status,
)

RESULT_PREFIX = "HARNESS_RESULT"
# Progress lines, so a runner whose harness process died knows whether a hidden case was
# running (and must then redact the crash text, which can contain sizes). Only this process
# writes to its stdout: workers write to their own log files.
STAGE_PREFIX = "HARNESS_STAGE"


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
    # {"eager_runs_ms", "compile_runs_ms"} from an earlier eval of this task (B9). They come
    # from clean reference workers only, so a candidate cannot poison them (Y4 B7).
    baselines: dict[str, Any] | None = None
    device: str = "cuda"  # tests run the whole pipeline on "cpu"
    # Red-team mode: skip the static scan to show the layers behind it hold on their own (Y4).
    # Only validation scripts set it; the agent's evals never do.
    skip_static_scan: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvalRequest":
        return cls(**data)


@dataclass
class _Clock:
    start: float = field(default_factory=time.monotonic)

    def seconds(self) -> float:
        return time.monotonic() - self.start


class _Stop(Exception):
    """Ends the pipeline with this result."""

    def __init__(self, result: HarnessResult):
        self.result = result


def _result(req: EvalRequest, clock: _Clock, status: Status, **kwargs: Any) -> HarnessResult:
    return HarnessResult(task_id=req.task_id, status=status, gpu_seconds=clock.seconds(), **kwargs)


def _tail(text: str) -> str:
    return text[-ERROR_EXCERPT_CHARS:]


def evaluate(req: EvalRequest) -> HarnessResult:
    clock = _Clock()
    try:
        return _evaluate(req, clock)
    except _Stop as stop:
        return stop.result
    except Exception:  # a bug in the harness itself; never shown to the agent
        excerpt = _tail(traceback.format_exc())
        return _result(req, clock, Status.HARNESS_ERROR, error_excerpt=excerpt)


def _stage(req: EvalRequest, name: str) -> None:
    """Announce the stage about to run: 'visible' or 'hidden' while candidate code runs."""
    sys.stdout.write(f"\n{STAGE_PREFIX}:{req.nonce}:{name}\n")
    sys.stdout.flush()


def _evaluate(req: EvalRequest, clock: _Clock) -> HarnessResult:
    from triton_kernel_agent.static_check import scan

    # 1-2. compile and static scan: nothing of the candidate's has run yet.
    try:
        report = scan(req.candidate_source)
    except SyntaxError as e:
        return _result(req, clock, Status.COMPILE_ERROR, error_excerpt=f"SyntaxError: {e}")
    if not report.clean and not req.skip_static_scan:
        return _result(
            req, clock, Status.STATIC_CHEAT, cheat_flags=report.flags,
            error_excerpt=_tail(report.summary()),
        )  # fmt: skip

    from triton_kernel_agent.isolation import WORKER_UID, clean_user, eval_dir, lock_down_harness

    lock_down_harness()
    root = eval_dir()
    try:
        return _Pipeline(req, clock, root).run()
    finally:
        shutil.rmtree(root, ignore_errors=True)
        if WORKER_UID is not None:
            clean_user(WORKER_UID)


@dataclass
class _Case:
    shape_class: ShapeClass
    hidden: bool
    ref: Any  # reference model (parent only)
    init: list[Any]
    inputs: list[Any]


class _Pipeline:
    def __init__(self, req: EvalRequest, clock: _Clock, root: Path):
        self.req, self.clock, self.root = req, clock, root
        self.device = req.device
        self.max_err = 0.0
        # Set once a hidden case has run: from then on nothing the worker says is passed on
        # (it can stash hidden sizes and raise them later, in any case or stage).
        self.hidden_seen = False
        # Two visible-shape input sets and the candidate's checked outputs for them; timed
        # batches alternate between them (benchmark.time_worker).
        self.timed_sets: list[tuple[Path, list[dict], list[Any]]] = []
        self.candidate_path = root / "candidate.py"
        self.candidate_path.write_text(req.candidate_source)
        self.candidate_path.chmod(0o644)

    def stop(self, status: Status, **kwargs: Any) -> _Stop:
        if "max_err" not in kwargs and self.max_err:
            kwargs["max_err"] = self.max_err
        return _Stop(_result(self.req, self.clock, status, **kwargs))

    # --- the stages ---

    def run(self) -> HarnessResult:
        from triton_kernel_agent.hidden_shapes import HiddenCase, build_case, generate
        from triton_kernel_agent.isolation import Worker, write_inputs
        from triton_kernel_agent.reference import INPUT_SEED

        req = self.req
        visible_case = HiddenCase(ShapeClass.VISIBLE, {}, "default", INPUT_SEED)
        hidden = generate(req.task_id, req.task_source, input_domain=req.input_domain)

        def case(c: HiddenCase) -> _Case:
            ref, init, inputs = build_case(req.task_source, c, self.device)
            return _Case(c.shape_class, c.shape_class != ShapeClass.VISIBLE, ref, init, inputs)

        visible = case(visible_case)
        visible_b = case(dataclasses.replace(visible_case, seed=INPUT_SEED + 1))
        cached = req.baselines if req.baselines and req.baselines.get("eager_runs_ms") else None

        # 3-5. correctness and runtime checks, in one worker.
        with Worker(self.root, "check") as worker:
            self.load(worker)
            _stage(req, "visible")
            for name, c in (("a", visible), ("b", visible_b)):
                outputs = self.check_case(worker, c)
                directory = self.root / f"visible_{name}"
                directory.mkdir()
                self.timed_sets.append((directory, write_inputs(c.inputs, directory), outputs))
            _stage(req, "hidden")
            self.hidden_seen = True
            for c in hidden:
                self.check_case(worker, case(c))
            first = case(hidden[0])
            fresh = case(dataclasses.replace(hidden[0], seed=hidden[0].seed + 1))
            runtime = self.runtime_checks(worker, first, fresh)
            _stage(req, "visible")
            if cached:  # no clean worker has to run, so this one can be timed right away
                return self.finish(self.time_candidate(worker, visible), cached, runtime)

        # 6. baselines in workers that never see candidate code, while none is running: the
        # candidate's worker and everything its user started were killed on leaving the block.
        directory, files, _ = self.timed_sets[0]
        baselines = self.time_baselines(visible, directory, files)

        # 6-7. the candidate, timed in a fresh worker; then its speed.
        with Worker(self.root, "time") as worker:
            self.load(worker)
            kernel = self.time_candidate(worker, visible)
        return self.finish(kernel, baselines, runtime)

    def references(self, c: _Case):
        from triton_kernel_agent.correctness import reference_outputs

        return reference_outputs(c.ref, c.inputs)

    def load(self, worker) -> None:
        answer = self.call(worker, {"cmd": "load", "path": str(self.candidate_path),
                                    "class": "ModelNew"})  # fmt: skip
        if not answer["ok"]:
            raise self.stop(Status.COMPILE_ERROR, error_excerpt=_tail(answer["message"]))

    def build(self, worker, c: _Case) -> None:
        from triton_kernel_agent.isolation import encode
        from triton_kernel_agent.reference import INIT_SEED

        command = {"cmd": "build", "init": encode(c.init), "seed": INIT_SEED,
                   "device": self.device}  # fmt: skip
        self.expect_ok(worker, command, c, "ModelNew(...)")

    def check_case(self, worker, c: _Case) -> list[Any]:
        """Build, run, and compare one case; stop the pipeline at the first failure."""
        from triton_kernel_agent.correctness import check_outputs

        ref32, ref64 = self.references(c)
        self.build(worker, c)
        outputs, _ = self.run_once(worker, c, reuse=False)
        check = check_outputs(outputs, ref32, ref64, self.req.atol, self.req.rtol, c.hidden)
        if not check.passed:
            self.fail_case(c, check)
        self.max_err = max(self.max_err, check.max_err)
        return outputs

    def run_once(self, worker, c: _Case, reuse: bool, audit: bool = False):
        """Send c's inputs (into the same buffers if reuse), run forward, read the outputs."""
        from triton_kernel_agent.isolation import BadOutput, read_output, write_inputs

        inbox = worker.inbox
        for f in inbox.iterdir():
            f.unlink()
        worker.clear_outbox()
        files = write_inputs(c.inputs, inbox)
        self.expect_ok(worker, {"cmd": "inputs", "files": files, "dir": str(inbox),
                                "reuse": reuse}, c, "loading inputs")  # fmt: skip
        answer = self.expect_ok(
            worker, {"cmd": "run", "out": str(worker.outbox), "audit": audit}, c, "forward"
        )
        try:
            metas = answer.get("outputs")
            if not isinstance(metas, list):
                raise BadOutput("no output list")
            outputs = [read_output(m, worker.outbox, self.device) for m in metas]
        except BadOutput as e:
            text = "output not readable" + ("" if self.secret(c) else f": {e}")
            raise self.stop(Status.WRONG_SHAPE, error_excerpt=text, extra=self.where(c)) from None
        finally:
            worker.clear_outbox()
        return outputs, answer

    def runtime_checks(self, worker, first: _Case, fresh: _Case):
        """Same buffers, new values; plus the worker's own audit (best effort)."""
        from triton_kernel_agent.correctness import check_outputs
        from triton_kernel_agent.runtime_check import RuntimeFinding, audit_findings, unchanged

        req = self.req
        ref32, ref64 = self.references(first)
        self.build(worker, first)
        out1, answer = self.run_once(worker, first, reuse=False, audit=True)
        check1 = check_outputs(out1, ref32, ref64, req.atol, req.rtol, hidden=True)
        if not check1.passed:  # same case it passed moments ago: behaves when not audited
            self.fail_case(first, check1)
        fresh_refs = self.references(fresh)
        out2, _ = self.run_once(worker, fresh, reuse=True)
        check2 = check_outputs(out2, *fresh_refs, req.atol, req.rtol, hidden=True)
        findings = audit_findings(answer.get("ops") or [], answer.get("kernels") or [],
                                  req.candidate_source)  # fmt: skip
        if not check2.passed:
            how = "did not change" if unchanged(out1, out2) else "was wrong"
            findings.append(RuntimeFinding(
                CheatFlag.INPUT_INSENSITIVE,
                f"output {how} when new values were written into the same input buffers",
            ))  # fmt: skip
        if answer.get("aliases_input"):
            findings.append(RuntimeFinding(CheatFlag.OUTPUT_ALIASES_INPUT,
                                           "an output shares memory with an input"))  # fmt: skip
        if findings:
            raise self.stop(
                Status.RUNTIME_CHEAT, cheat_flags=sorted({f.flag for f in findings}),
                error_excerpt=_tail("\n".join(f.message for f in findings)),
                extra={"gpu_kernels": sorted(set(answer.get("kernels") or []))},
            )  # fmt: skip
        return {"gpu_kernels": sorted(set(answer.get("kernels") or []))}

    def time_baselines(self, visible: _Case, directory: Path, files: list[dict]) -> dict:
        from triton_kernel_agent.benchmark import TimingFailed, time_worker
        from triton_kernel_agent.isolation import Worker, encode
        from triton_kernel_agent.reference import INIT_SEED

        task_path = self.root / "task.py"
        task_path.write_text(self.req.task_source)
        task_path.chmod(0o644)
        runs: dict[str, Any] = {}
        for name, compile_ in (("eager", False), ("compile", True)):
            with Worker(self.root, f"ref_{name}", strict_fp32=True) as worker:
                steps = [
                    {"cmd": "load", "path": str(task_path), "class": "Model"},
                    {"cmd": "build", "init": encode(visible.init), "seed": INIT_SEED,
                     "device": self.device, "compile": compile_},
                    {"cmd": "inputs", "files": files, "dir": str(directory)},
                ]  # fmt: skip
                for step in steps:
                    answer = worker.call(step)
                    if not answer["ok"]:
                        raise RuntimeError(f"reference {name} failed: {answer['message']}")
                try:
                    timing = time_worker(worker)
                except TimingFailed as e:
                    raise RuntimeError(f"reference {name} failed: {e}") from None
            runs[f"{name}_runs_ms"] = timing.runs_ms
            if compile_:
                runs["compile_s"] = timing.first_call_s
        return runs

    def time_candidate(self, worker, visible: _Case):
        """Time the candidate on the visible shapes; check every batch's last output."""
        from triton_kernel_agent.benchmark import TimingFailed, time_worker
        from triton_kernel_agent.correctness import check_outputs
        from triton_kernel_agent.isolation import BadOutput, WorkerError, read_output

        def load(index: int, reuse: bool) -> None:
            directory, files, _ = self.timed_sets[index]
            self.expect_ok(worker, {"cmd": "inputs", "files": files, "dir": str(directory),
                                    "reuse": reuse}, visible, "loading inputs")  # fmt: skip

        current = [0]

        def after_batch(_: int) -> None:
            """Outside the timed window: check the batch's last output, then new values."""
            answer = self.expect_ok(worker, {"cmd": "dump", "out": str(worker.outbox)}, visible,
                                    "returning the last timed output")  # fmt: skip
            try:
                outputs = [read_output(m, worker.outbox, self.device) for m in answer["outputs"]]
                # Against the candidate's own checked output for these values: same kernel,
                # so the task's fixed tolerance is plenty.
                check = check_outputs(outputs, self.timed_sets[current[0]][2], None,
                                      self.req.atol, self.req.rtol, hidden=True)  # fmt: skip
                why = check.message if not check.passed else None
            except (BadOutput, KeyError, TypeError):
                why = "its output was not readable"
            finally:
                worker.clear_outbox()
            if why is not None:
                raise self.stop(
                    Status.RUNTIME_CHEAT, cheat_flags=[CheatFlag.CHANGED_AFTER_CHECKS],
                    error_excerpt="the output of a timed call was wrong",
                    extra={"case": "visible", "stage": "timing"},
                )  # fmt: skip
            current[0] = 1 - current[0]
            load(current[0], reuse=True)  # new values, same buffers

        self.build(worker, visible)
        load(0, reuse=False)
        try:
            return time_worker(worker, after_batch)
        except TimingFailed as e:
            name = _error_name(e.answer.get("error"))
            text = f"{name} while timed (message withheld)"
            raise self.stop(Status.RUNTIME_ERROR, error_excerpt=text,
                            extra={"case": "visible", "stage": "timing"}) from None  # fmt: skip
        except WorkerError as e:
            status = Status.TIMEOUT if e.kind == "timeout" else Status.RUNTIME_ERROR
            what = "no answer while timed" if e.kind == "timeout" else "crashed while timed"
            raise self.stop(status, extra={"case": "visible", "stage": "timing"},
                            error_excerpt=f"the process {what}") from None  # fmt: skip

    def finish(self, kernel, baselines: dict, runtime: dict) -> HarnessResult:
        from statistics import median

        from triton_kernel_agent.benchmark import impossible_speed
        from triton_kernel_agent.harness_result import Timing

        req = self.req
        eager = baselines["eager_runs_ms"]
        compiled = baselines.get("compile_runs_ms") or eager
        timing = Timing(kernel_ms=kernel.median_ms, eager_ms=median(eager),
                        compile_ms=median(compiled))  # fmt: skip
        bench = {
            "kernel_runs_ms": kernel.runs_ms,
            "eager_runs_ms": eager,
            "compile_runs_ms": compiled,
            "baselines_cached": bool(req.baselines and req.baselines.get("eager_runs_ms")),
            "compile_s": baselines.get("compile_s", 0.0),
            "calls_per_batch": kernel.calls,
            "first_call_s": kernel.first_call_s,
        }
        extra = {"benchmark": bench, **runtime}
        peak = _peak(self.device)
        if peak is None:  # no peak numbers off the GPU: no roofline, no physics check
            return _result(req, self.clock, Status.CORRECT, max_err=self.max_err,
                           timing=timing, extra=extra)  # fmt: skip

        from triton_kernel_agent.roofline import roofline

        if why := impossible_speed(req.bytes, timing.kernel_ms, peak):
            # Timing belongs to correct results only; the measured runs stay in extra.
            raise self.stop(Status.RUNTIME_CHEAT, cheat_flags=[CheatFlag.IMPOSSIBLE_SPEED],
                            error_excerpt=why, extra=extra)  # fmt: skip
        roof = roofline(req.bytes, req.flops, timing.kernel_ms, peak)
        return _result(
            req, self.clock, Status.CORRECT, max_err=self.max_err, timing=timing,
            pct_peak_bandwidth=roof.pct_peak_bandwidth, pct_peak_flops=roof.pct_peak_flops,
            extra={**extra, "roofline": roof.as_dict()},
        )  # fmt: skip

    # --- talking to workers ---

    def secret(self, c: _Case | None = None) -> bool:
        """True if what the worker says now must not reach the agent (Y4 F2 and review)."""
        return self.hidden_seen or (c is not None and c.hidden)

    def where(self, c: _Case) -> dict[str, str]:
        return {"case": "hidden" if self.secret(c) else "visible"}

    def call(self, worker, command: dict, c: _Case | None = None) -> dict:
        """worker.call, with a dead or hung worker turned into the candidate's failure."""
        from triton_kernel_agent.isolation import WorkerError

        try:
            return worker.call(command)
        except WorkerError as e:
            extra = {"case": "hidden" if self.secret(c) else "visible"}
            if e.kind == "timeout":
                raise self.stop(Status.TIMEOUT, error_excerpt=str(e), extra=extra) from None
            # Exit codes and crash output are the candidate's to choose: withheld once secret.
            text = "the process crashed (output withheld)" if self.secret(c) else str(e)
            raise self.stop(Status.RUNTIME_ERROR, error_excerpt=_tail(text), extra=extra) from None

    def expect_ok(self, worker, command: dict, c: _Case, what: str) -> dict:
        """Send a command for case c; a candidate exception stops the pipeline."""
        answer = self.call(worker, command, c)
        if answer["ok"]:
            return answer
        status = Status.COMPILE_ERROR if answer.get("compile") else Status.RUNTIME_ERROR
        name = _error_name(answer.get("error"))
        if self.secret(c):  # messages (and custom exception names) can spell out sizes
            place = "on a hidden test case" if c.hidden else "after the hidden test cases"
            text = f"{name} in {what} {place} (message withheld)"
        else:
            text = str(answer.get("message", name))
        raise self.stop(status, error_excerpt=_tail(text), extra=self.where(c))

    def fail_case(self, c: _Case, check) -> None:
        import torch

        shapes = [list(x.shape) for x in c.inputs if isinstance(x, torch.Tensor)]
        output_problem = check.status in (Status.WRONG_SHAPE, Status.NAN_INF, Status.MISMATCH)
        if self.secret(c):
            # Error sizes and output ranks are the candidate's numbers to choose: withheld.
            message, max_err = f"{check.status} on a hidden test case (details withheld)", None
        else:
            finite = [e for e in (self.max_err, check.max_err) if e == e]
            message, max_err = check.message, max(finite) if finite else None
        raise self.stop(
            check.status, max_err=max_err,
            failing_shape_class=c.shape_class if output_problem else None,
            failing_shape=shapes if output_problem else None,
            error_excerpt=message, extra=self.where(c),
        )  # fmt: skip


# Exception names passed on to the agent; anything else (a custom class can be named after a
# hidden size) is reported as Exception.
ERROR_NAMES = frozenset(
    {
        "Exception", "RuntimeError", "ValueError", "TypeError", "IndexError", "KeyError",
        "AttributeError", "AssertionError", "ZeroDivisionError", "NotImplementedError",
        "OverflowError", "MemoryError", "OutOfMemoryError", "NameError", "UnboundLocalError",
        "ImportError", "ModuleNotFoundError", "SyntaxError", "RecursionError", "SystemExit",
        "CompilationError", "PTXASError", "OutOfResources", "UnsupportedLanguageConstruct",
    }
)  # fmt: skip


def _error_name(name: Any) -> str:
    return name if name in ERROR_NAMES else "Exception"


def _peak(device: str):
    """The GPU's peak numbers (roofline.GpuPeak), or None off the GPU."""
    if not device.startswith("cuda"):
        return None
    import torch

    from triton_kernel_agent.roofline import peak_for

    return peak_for(torch.cuda.get_device_name())


def main() -> None:
    request = EvalRequest.from_dict(json.loads(sys.stdin.read()))
    result = evaluate(request)
    sys.stdout.write(f"\n{RESULT_PREFIX}:{request.nonce}:{result.to_json()}\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
