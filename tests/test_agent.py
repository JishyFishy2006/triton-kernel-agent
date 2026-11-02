"""B14 acceptance: with a fake model scripted fail, fail, pass, the loop stops at attempt 3;
the budget cap stops it at 10."""

from dataclasses import dataclass, field

import pytest

from triton_kernel_agent.agent import (
    OPTIMIZE_PASSES,
    Budget,
    ModelReply,
    run_task,
)
from triton_kernel_agent.harness_result import HarnessResult, ShapeClass, Status, Timing
from triton_kernel_agent.prompts import KernelResponse
from triton_kernel_agent.tasks import get_task

TASK = get_task("L1-19")
BUDGET = Budget(max_attempts=10, max_usd=3.0)


def correct(speedup: float) -> HarnessResult:
    timing = Timing(kernel_ms=4.3 / speedup, eager_ms=4.3, compile_ms=4.3)
    return HarnessResult(TASK.id, Status.CORRECT, max_err=0.0, timing=timing)


FAIL = HarnessResult(TASK.id, Status.MISMATCH, max_err=0.5, failing_shape_class=ShapeClass.NON_POW2,
                     error_excerpt="max abs error 0.5")  # fmt: skip
COMPILE = HarnessResult(TASK.id, Status.COMPILE_ERROR, error_excerpt="NameError: name 'tl'")


@dataclass
class FakeWriter:
    """Returns code "<n>"; cost per call; None replies break the schema."""

    cost: float = 0.1
    broken: set[int] = field(default_factory=set)
    calls: list[tuple[str, list]] = field(default_factory=list)

    def write(self, role, messages, tags):
        self.calls.append((role, messages))
        n = len(self.calls)
        response = None if n in self.broken else KernelResponse(f"plan {n}", f"code {n}\n")
        return ModelReply(response, "fake", "low", self.cost, 10, 0, 20, error="bad json")


@dataclass
class ScriptedHarness:
    script: list[HarnessResult]
    evaluated: list[str] = field(default_factory=list)

    def evaluate(self, task, code):
        self.evaluated.append(code)
        return self.script[min(len(self.evaluated), len(self.script)) - 1]


def run(script, writer=None, budget=BUDGET, **kwargs):
    writer = writer or FakeWriter()
    harness = ScriptedHarness(script)
    return run_task(TASK, "TASK", writer, harness, budget, **kwargs), writer, harness


def test_fail_fail_pass_stops_at_attempt_3():
    outcome, writer, _ = run([FAIL, FAIL, correct(1.5)])
    assert len(outcome.attempts) == 3
    assert outcome.stop_reason == "faster_than_eager"
    assert outcome.best.n == 3 and outcome.best.speedup == pytest.approx(1.5)


def test_attempt_cap_stops_at_10():
    outcome, writer, _ = run([FAIL])
    assert len(outcome.attempts) == 10 and len(writer.calls) == 10
    assert outcome.stop_reason == "max_attempts" and outcome.best is None


def test_dollar_cap_stops_the_loop():
    outcome, _, _ = run([FAIL], writer=FakeWriter(cost=1.0))
    assert len(outcome.attempts) == 3 and outcome.stop_reason == "max_usd"
    assert outcome.cost_usd == 3.0


def test_compile_errors_go_to_the_fixer_once():
    outcome, writer, _ = run([COMPILE, COMPILE, COMPILE, correct(2.0)])
    assert [a.role for a in outcome.attempts] == ["planner", "fixer", "planner", "fixer"]
    fixer_prompt = writer.calls[1][1]
    assert "NameError: name 'tl'" in fixer_prompt[-1]["content"]
    assert "code 1" in fixer_prompt[-2]["content"]  # the broken code goes to the fixer


def test_schema_breaking_replies_are_not_evaluated():
    outcome, writer, harness = run([correct(1.2)], writer=FakeWriter(broken={1}))
    first = outcome.attempts[0]
    assert first.result.status == Status.COMPILE_ERROR and "JSON format" in first.feedback
    assert harness.evaluated == ["code 2\n"]  # only the second, valid reply ran
    assert outcome.attempts[1].role == "fixer"


def test_correct_but_slower_gets_optimization_passes_then_stops():
    outcome, writer, _ = run([correct(0.8), correct(0.9), correct(0.95), correct(3.0)])
    assert len(outcome.attempts) == 1 + OPTIMIZE_PASSES
    assert outcome.stop_reason == "correct_not_faster"
    assert outcome.best.speedup == pytest.approx(0.95)


def test_planner_sees_earlier_attempts_and_feedback():
    outcome, writer, _ = run([FAIL, FAIL, correct(1.5)], feedback=lambda r: f"FB {r.status}")
    role, messages = writer.calls[2]
    assert role == "planner"
    contents = [m["content"] for m in messages]
    assert contents.count("FB mismatch") == 2
    assert any('"code 2\\n"' in c for c in contents)


def test_every_attempt_is_reported_as_it_happens():
    seen = []
    run([FAIL, correct(1.1)], on_attempt=seen.append)
    assert [a.n for a in seen] == [1, 2]
    assert seen[0].reply.cost_usd == 0.1 and seen[0].feedback


def test_resuming_continues_from_recorded_attempts():
    first, _, _ = run([FAIL, FAIL], budget=Budget(max_attempts=2, max_usd=3.0))
    resumed, writer, _ = run([correct(1.5)], previous=first.attempts)
    assert [a.n for a in resumed.attempts] == [1, 2, 3]
    assert len(writer.calls) == 1  # only the new attempt called the model
