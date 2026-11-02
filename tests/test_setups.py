"""B16 acceptance: the setup name is stored on every attempt row. Plus which model plays each
role in each setup, and the real writer's parsing of model replies."""

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime

import pytest

from triton_kernel_agent.agent import Budget, run_task
from triton_kernel_agent.config import MODEL_SETUPS, load_config
from triton_kernel_agent.db import ResultsDB, Run
from triton_kernel_agent.harness_result import HarnessResult, Status, Timing
from triton_kernel_agent.llm import CallResult
from triton_kernel_agent.setups import DBRecorder, ModelWriter, attempt_id, models_for
from triton_kernel_agent.tasks import get_task, load_tasks

MODELS = load_config().models
TASK = get_task("L1-19")


def test_models_for_each_setup():
    wonder, spark = "muse-wonder-1.0", "muse-spark-1.3"
    expected = {
        "spark_only": (spark, spark),
        "wonder_only": (wonder, wonder),
        "wonder_spark": (wonder, spark),
    }
    for setup, (planner, fixer) in expected.items():
        roles = models_for(setup, MODELS)
        assert (roles["planner"].name, roles["fixer"].name) == (planner, fixer), setup
        # Efforts follow the role, so an ablation changes only the model.
        assert roles["planner"].reasoning_effort == MODELS.planner.reasoning_effort
        assert roles["fixer"].reasoning_effort == MODELS.fixer.reasoning_effort
    with pytest.raises(ValueError, match="setup"):
        models_for("both", MODELS)


@dataclass
class FakeClient:
    """Stands in for llm.ModelClient: replies compile_error first, then valid JSON."""

    replies: list[str]
    calls: list[dict] = field(default_factory=list)

    def complete(self, model, messages, *, tags=None, **kwargs):
        self.calls.append({"model": model.name, "tags": tags, "kwargs": kwargs})
        text = self.replies[min(len(self.calls), len(self.replies)) - 1]
        return CallResult(text, model.name, model.reasoning_effort, 100, 0, 200, 0.01, 1.0, "r")


def reply(n):
    return json.dumps({"plan_summary": f"plan {n}", "code": f"class ModelNew: pass  # {n}"})


@dataclass
class ScriptedHarness:
    script: list[HarnessResult]
    n: int = 0

    def evaluate(self, task, code):
        self.n += 1
        return self.script[min(self.n, len(self.script)) - 1]


CORRECT = HarnessResult(TASK.id, Status.CORRECT, max_err=0.0,
                        timing=Timing(kernel_ms=1.0, eager_ms=2.0, compile_ms=2.0))  # fmt: skip
COMPILE = HarnessResult(TASK.id, Status.COMPILE_ERROR, error_excerpt="NameError")


@pytest.mark.parametrize("setup", MODEL_SETUPS)
def test_setup_name_is_stored_on_every_attempt_row(setup):
    with ResultsDB() as db:
        db.upsert_tasks(load_tasks(include_canaries=True))
        db.insert_run(Run("run-1", {"setup": setup}, datetime.now(UTC), "abc"))
        client = FakeClient(["not json", reply(2), reply(3)])
        writer = ModelWriter(client, setup, MODELS)
        harness = ScriptedHarness([COMPILE, CORRECT])
        outcome = run_task(TASK, "TASK", writer, harness, Budget(10, 3.0),
                           on_attempt=DBRecorder(db, "run-1", setup))  # fmt: skip

        rows = db.sql("SELECT n, setup, model, code FROM attempts ORDER BY n")
        assert [r[1] for r in rows] == [setup] * len(outcome.attempts) == [setup] * 3
        roles = models_for(setup, MODELS)
        # attempt 1: schema failure (planner); 2: fixer; 3: planner after a compile error
        assert [r[2] for r in rows] == [roles["planner"].name, roles["fixer"].name,
                                        roles["planner"].name]  # fmt: skip
        assert rows[0][3] == ""  # nothing to store for a reply that broke the schema
        assert db.get_eval(attempt_id("run-1", TASK.id, 3)).status == Status.CORRECT
        assert all(c["tags"]["setup"] == setup for c in client.calls)


def test_writer_requests_the_structured_output_schema():
    client = FakeClient([reply(1)])
    response = ModelWriter(client, "wonder_spark", MODELS).write("planner", [], {"task_id": "x"})
    assert response.response.plan_summary == "plan 1"
    assert client.calls[0]["kwargs"]["text"]["format"]["name"] == "kernel_response"
