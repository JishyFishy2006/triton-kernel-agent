"""Batch runner (B17): many tasks in parallel, resumable, within the API's rate limits.

- Parallel: `workers` threads, each with its own evaluator (one warm sandbox per worker); the
  model client and the results database are shared (ResultsDB serialises access).
- Resumable: a run is identified by its run_id. On (re)start every task's recorded attempts
  are loaded and the agent loop continues from them; finished tasks make no calls. Attempts
  and their evals are written in one transaction with key (run, task, n), so a kill at any
  moment loses at most the attempt in flight and never duplicates one.
- Rate limits: RateLimitedWriter caps concurrent model calls and requests per minute; the SDK
  also retries 429s honouring Retry-After.
"""

import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from dataclasses import dataclass

from triton_kernel_agent.agent import (
    AttemptRecord,
    Budget,
    Evaluator,
    ModelReply,
    Role,
    TaskOutcome,
    Writer,
    run_task,
)
from triton_kernel_agent.db import ResultsDB, Run
from triton_kernel_agent.feedback import format_feedback
from triton_kernel_agent.prompts import KernelResponse, task_prompt
from triton_kernel_agent.setups import DBRecorder
from triton_kernel_agent.tasks import Task


class RateLimiter:
    """At most `max_concurrent` calls at once and `per_minute` call starts per 60 s."""

    def __init__(self, max_concurrent: int = 4, per_minute: int = 30, clock=time.monotonic,
                 sleep=time.sleep):  # fmt: skip
        self._slots = threading.BoundedSemaphore(max_concurrent)
        self._starts: deque[float] = deque()
        self._lock = threading.Lock()
        self.per_minute, self._clock, self._sleep = per_minute, clock, sleep

    def __enter__(self) -> "RateLimiter":
        self._slots.acquire()
        while True:
            with self._lock:
                now = self._clock()
                while self._starts and now - self._starts[0] >= 60:
                    self._starts.popleft()
                if len(self._starts) < self.per_minute:
                    self._starts.append(now)
                    return self
                wait_s = 60 - (now - self._starts[0])
            self._sleep(wait_s)

    def __exit__(self, *exc: object) -> None:
        self._slots.release()


@dataclass
class RateLimitedWriter:
    writer: Writer
    limiter: RateLimiter

    def write(self, role: Role, messages: list[dict[str, str]], tags: dict) -> ModelReply:
        with self.limiter:
            return self.writer.write(role, messages, tags)


def recorded_attempts(db: ResultsDB, run_id: str, task_id: str) -> list[AttemptRecord]:
    """Rebuild a task's AttemptRecords from the database, to resume the agent loop."""
    records = []
    for attempt, result in db.task_attempts(run_id, task_id):
        response = None
        if attempt.code:
            response = KernelResponse(attempt.plan_summary, attempt.code)
        reply = ModelReply(
            response, attempt.model, attempt.effort, attempt.cost_usd, attempt.input_tokens,
            attempt.cached_input_tokens, attempt.output_tokens,
        )  # fmt: skip
        records.append(
            AttemptRecord(attempt.n, attempt.role, reply, result, format_feedback(result))
        )
    return records


def run_batch(
    db: ResultsDB,
    run: Run,
    setup: str,
    tasks: Sequence[Task],
    writer: Writer,
    make_evaluator: Callable[[], Evaluator],
    budget: Budget,
    *,
    workers: int = 4,
    shapes: dict[str, tuple[list, list]] | None = None,
    on_attempt: Callable[[AttemptRecord], None] | None = None,
) -> dict[str, TaskOutcome]:
    """Run (or resume) every task. `shapes` maps task_id -> (input shapes, output shape) for
    the task prompt. Returns each task's outcome, including attempts from before a restart."""
    if db.get_run(run.run_id) is None:
        db.insert_run(run)
    db.upsert_tasks(list(tasks))
    recorder = DBRecorder(db, run.run_id, setup)
    local = threading.local()
    outcomes: dict[str, TaskOutcome] = {}

    def record(attempt: AttemptRecord) -> None:
        recorder(attempt)
        if on_attempt is not None:
            on_attempt(attempt)

    def one(task: Task) -> None:
        if not hasattr(local, "evaluator"):
            local.evaluator = make_evaluator()
        inputs, output = (shapes or {}).get(task.id, (None, None))
        outcomes[task.id] = run_task(
            task, task_prompt(task, inputs, output), writer, local.evaluator, budget,
            on_attempt=record, previous=recorded_attempts(db, run.run_id, task.id),
        )  # fmt: skip

    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        futures = [pool.submit(one, t) for t in tasks]
        done, _ = wait(futures, return_when=FIRST_EXCEPTION)
        for future in done:
            future.result()  # re-raise the first failure (e.g. Ctrl-C, budget exhausted)
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    return outcomes
