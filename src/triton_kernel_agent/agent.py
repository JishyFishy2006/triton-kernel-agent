"""Agent loop (B14): a small state machine with a hard budget per task.

Each attempt: a model writes {plan_summary, code}, the harness evaluates it, and the result
becomes feedback for the next attempt (tech spec, "Agent loop"):
- a compile error (or a reply that doesn't match the schema) goes to the fixer once, for a
  cheap minimal fix; anything else goes back to the planner with the feedback;
- stop as soon as a kernel is correct and faster than eager;
- a correct but slower kernel gets up to OPTIMIZE_PASSES more attempts aimed at speed;
- stop at budget.max_attempts attempts or once budget.max_usd is spent.
The best kernel is the fastest correct one. The model, the harness and where attempts are
recorded are all passed in, so tests use fakes and B16/B17 plug in the real ones.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol

from triton_kernel_agent.feedback import format_feedback
from triton_kernel_agent.harness_result import HarnessResult, Status
from triton_kernel_agent.prompts import (
    KernelResponse,
    Turn,
    fixer_messages,
    planner_messages,
)
from triton_kernel_agent.tasks import Task

Role = Literal["planner", "fixer"]
# Extra attempts spent trying to speed up a kernel that is correct but not faster than eager.
OPTIMIZE_PASSES = 2


@dataclass(frozen=True)
class Budget:
    max_attempts: int
    max_usd: float


@dataclass(frozen=True)
class ModelReply:
    """What one model call produced. response is None when the reply broke the schema."""

    response: KernelResponse | None
    model: str
    effort: str
    cost_usd: float
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    error: str = ""  # why the reply was rejected, if it was


class Writer(Protocol):
    def write(self, role: Role, messages: list[dict[str, str]], tags: dict) -> ModelReply: ...


class Evaluator(Protocol):
    def evaluate(self, task: Task, code: str) -> HarnessResult: ...


@dataclass(frozen=True)
class AttemptRecord:
    n: int  # 1-based
    role: Role
    reply: ModelReply
    result: HarnessResult
    feedback: str

    @property
    def speedup(self) -> float | None:
        return self.result.timing.speedup_eager if self.result.is_correct else None


@dataclass
class TaskOutcome:
    task_id: str
    attempts: list[AttemptRecord] = field(default_factory=list)
    stop_reason: str = ""

    @property
    def cost_usd(self) -> float:
        return sum(a.reply.cost_usd for a in self.attempts)

    @property
    def best(self) -> AttemptRecord | None:
        correct = [a for a in self.attempts if a.result.is_correct]
        return max(correct, key=lambda a: a.speedup) if correct else None


def _next_role(attempts: Sequence[AttemptRecord]) -> Role:
    if not attempts:
        return "planner"
    last = attempts[-1]
    if last.result.status == Status.COMPILE_ERROR and last.role != "fixer":
        return "fixer"
    return "planner"


def _stop_reason(attempts: Sequence[AttemptRecord], budget: Budget) -> str | None:
    if attempts:
        last = attempts[-1]
        if last.result.is_correct and last.speedup > 1:
            return "faster_than_eager"
        first_correct = next((a.n for a in attempts if a.result.is_correct), None)
        if first_correct is not None and last.n - first_correct >= OPTIMIZE_PASSES:
            return "correct_not_faster"
    if len(attempts) >= budget.max_attempts:
        return "max_attempts"
    if sum(a.reply.cost_usd for a in attempts) >= budget.max_usd:
        return "max_usd"
    return None


def _schema_failure(task: Task, reply: ModelReply) -> HarnessResult:
    msg = f"The reply did not match the required JSON format: {reply.error}"
    return HarnessResult(task.id, Status.COMPILE_ERROR, error_excerpt=msg)


def run_task(
    task: Task,
    task_text: str,
    writer: Writer,
    evaluator: Evaluator,
    budget: Budget,
    *,
    feedback: Callable[[HarnessResult], str] = format_feedback,
    on_attempt: Callable[[AttemptRecord], None] | None = None,
    previous: Sequence[AttemptRecord] = (),
) -> TaskOutcome:
    """Run the loop for one task. `previous` resumes a task whose attempts were recorded."""
    outcome = TaskOutcome(task.id, list(previous))
    while (reason := _stop_reason(outcome.attempts, budget)) is None:
        role = _next_role(outcome.attempts)
        history = [Turn(a.reply.response, a.feedback) for a in outcome.attempts if a.reply.response]
        if role == "fixer":
            last = outcome.attempts[-1]
            broken = last.reply.response or KernelResponse("", "")
            messages = fixer_messages(task_text, broken, last.result.error_excerpt)
        else:
            messages = planner_messages(task_text, history)

        n = len(outcome.attempts) + 1
        reply = writer.write(role, messages, {"task_id": task.id, "attempt": n, "role": role})
        if reply.response is None:
            result = _schema_failure(task, reply)
        else:
            result = evaluator.evaluate(task, reply.response.code)
        record = AttemptRecord(n, role, reply, result, feedback(result))
        outcome.attempts.append(record)
        if on_attempt is not None:
            on_attempt(record)
    outcome.stop_reason = reason
    return outcome
