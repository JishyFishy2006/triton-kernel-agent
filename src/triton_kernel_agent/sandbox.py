"""Sandbox runner (B11): candidate code string in, HarnessResult out.

Each eval runs harness.py in a fresh Python process inside a long-lived Modal Sandbox on an
H100 ("keep one warm container per worker"):
- no network (block_network=True) and no secrets: the Meta API key never leaves this machine;
- a hard wall-clock limit per eval (coreutils `timeout -s KILL`), reported as timeout;
- a crash (illegal memory access, segfault) kills only that process; the next eval starts clean;
- the result is read from the line tagged with a random nonce, so the candidate's own prints
  cannot pose as a result.
Eager and torch.compile timings are cached per task and passed to later evals (B9).
"""

import json
import secrets
import time
from dataclasses import asdict
from typing import Any

import modal

from triton_kernel_agent.correctness import _redact
from triton_kernel_agent.harness import RESULT_PREFIX, STAGE_PREFIX, EvalRequest
from triton_kernel_agent.harness_result import ERROR_EXCERPT_CHARS, HarnessResult, Status
from triton_kernel_agent.tasks import Task

APP_NAME = "tka-sandbox"
# Upper bound on one Sandbox's life; a batch run creates a new one when it expires.
SANDBOX_LIFETIME_S = 4 * 3600
# coreutils `timeout` exits 124 when it had to kill the command (137 if KILL reached it directly).
TIMEOUT_EXIT_CODES = (124, 137)
# Kills the worker user's processes and deletes its files (isolation.clean_user).
CLEAN_UP = "from triton_kernel_agent.isolation import NOBODY, clean_user; clean_user(NOBODY)"


def parse_output(stdout: str, nonce: str) -> HarnessResult | None:
    """The result line tagged with this eval's nonce, or None if the process died first."""
    tag = f"{RESULT_PREFIX}:{nonce}:"
    for line in reversed(stdout.splitlines()):
        if line.startswith(tag):
            return HarnessResult.from_json(line[len(tag) :])
    return None


def last_stage(stdout: str, nonce: str) -> str | None:
    """The last stage the harness announced ('visible', 'hidden'), or None."""
    tag = f"{STAGE_PREFIX}:{nonce}:"
    stages = [line[len(tag) :] for line in stdout.splitlines() if line.startswith(tag)]
    return stages[-1] if stages else None


def result_from_process(
    task_id: str, nonce: str, returncode: int, stdout: str, stderr: str, seconds: float,
    timeout_s: float,
) -> HarnessResult:  # fmt: skip
    """Turn one finished harness process into a HarnessResult."""
    parsed = parse_output(stdout, nonce)
    if parsed is not None:
        return parsed
    # Died without a result. If a hidden case was running, its sizes must not reach the agent.
    stage = last_stage(stdout, nonce)
    case = {"case": "hidden" if stage == "hidden" else "visible", "stage": stage}
    if returncode in TIMEOUT_EXIT_CODES or seconds >= timeout_s:
        msg = f"killed after {timeout_s:g} s wall-clock limit"
        return HarnessResult(task_id, Status.TIMEOUT, error_excerpt=msg, gpu_seconds=seconds,
                             extra=case)  # fmt: skip
    # A crash in the candidate (e.g. illegal memory access).
    tail = (stderr or stdout)[-ERROR_EXCERPT_CHARS:]
    if stage == "hidden":
        tail = _redact(tail)
    msg = f"process exited with code {returncode} before reporting a result\n{tail}"
    return HarnessResult(task_id, Status.RUNTIME_ERROR, error_excerpt=msg[-ERROR_EXCERPT_CHARS:],
                         gpu_seconds=seconds, extra=case)  # fmt: skip


class SandboxRunner:
    """One warm H100 Sandbox; call evaluate() as often as needed, then close()."""

    def __init__(self, gpu: str = "H100!", eval_timeout_s: float = 300):
        from triton_kernel_agent.modal_tasks import image

        self.gpu, self.eval_timeout_s = gpu, eval_timeout_s
        self.app = modal.App.lookup(APP_NAME, create_if_missing=True)
        self.image = image
        self.baselines: dict[str, dict[str, Any]] = {}
        self._sandbox: modal.Sandbox | None = None

    @classmethod
    def from_config(cls) -> "SandboxRunner":
        from triton_kernel_agent.config import load_config

        modal_cfg = load_config().modal
        return cls(gpu=modal_cfg.gpu, eval_timeout_s=modal_cfg.eval_timeout_s)

    def _get_sandbox(self) -> modal.Sandbox:
        if self._sandbox is None or self._sandbox.poll() is not None:
            self._sandbox = modal.Sandbox.create(
                app=self.app,
                image=self.image,
                gpu=self.gpu,
                timeout=SANDBOX_LIFETIME_S,
                block_network=True,  # generated code never talks to the outside world
            )
        return self._sandbox

    def exec(self, *cmd: str, stdin: str = "") -> tuple[int, str, str, float]:
        """Run a command in the sandbox: (returncode, stdout, stderr, seconds)."""
        start = time.monotonic()
        proc = self._get_sandbox().exec(*cmd)
        if stdin:
            proc.stdin.write(stdin.encode())
        proc.stdin.write_eof()
        proc.stdin.drain()
        stdout, stderr = proc.stdout.read(), proc.stderr.read()
        returncode = proc.wait()
        return returncode, stdout, stderr, time.monotonic() - start

    def evaluate(self, task: Task, code: str) -> HarnessResult:
        nonce = secrets.token_hex(16)
        request = EvalRequest(
            task_id=task.id,
            task_source=task.source,
            candidate_source=code,
            atol=task.atol,
            rtol=task.rtol,
            bytes=task.bytes or 0,
            flops=task.flops or 0,
            input_domain=task.input_domain,
            nonce=nonce,
            baselines=self.baselines.get(task.id),
        )
        limit = f"{self.eval_timeout_s:g}"
        returncode, stdout, stderr, seconds = self.exec(
            "timeout", "-s", "KILL", limit, "python", "-m", "triton_kernel_agent.harness",
            stdin=json.dumps(asdict(request)),
        )  # fmt: skip
        result = result_from_process(task.id, nonce, returncode, stdout, stderr, seconds,
                                     self.eval_timeout_s)  # fmt: skip
        if returncode in TIMEOUT_EXIT_CODES:  # the harness was killed and could not clean up
            self.exec("python", "-c", CLEAN_UP)
        self._remember_baselines(task.id, result)
        return result

    def _remember_baselines(self, task_id: str, result: HarnessResult) -> None:
        bench = result.extra.get("benchmark") or {}
        if bench.get("eager_runs_ms") and not bench.get("baselines_cached"):
            self.baselines[task_id] = {
                "eager_runs_ms": bench["eager_runs_ms"],
                "compile_runs_ms": bench.get("compile_runs_ms"),
            }

    def close(self) -> None:
        if self._sandbox is not None:
            self._sandbox.terminate()
            self._sandbox = None

    def __enter__(self) -> "SandboxRunner":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
