"""Model setups (B16) and the pieces that connect the agent loop to the real world.

A setup decides which model plays each role (config.toml [models].setup):
- spark_only:   muse-spark-1.3 plans/writes and fixes
- wonder_only:  muse-wonder-1.0 plans/writes and fixes
- wonder_spark: Wonder plans/writes, Spark fixes compile errors cheaply
Efforts come from the role ([models.planner] / [models.fixer]), so an ablation changes only
the model. ModelWriter is the real Writer for agent.run_task; DBRecorder stores every attempt,
with its setup, in the results database (B12).
"""

import hashlib
from dataclasses import replace
from typing import Any

from triton_kernel_agent.agent import AttemptRecord, ModelReply, Role
from triton_kernel_agent.config import MODEL_SETUPS, ModelConfig, ModelsConfig
from triton_kernel_agent.db import Attempt, ResultsDB
from triton_kernel_agent.prompts import RESPONSE_TEXT, ResponseError, parse_response


def models_for(setup: str, models: ModelsConfig) -> dict[Role, ModelConfig]:
    """The model (with its role's effort) for each role under a setup."""
    if setup not in MODEL_SETUPS:
        raise ValueError(f"setup must be one of {MODEL_SETUPS}, got {setup!r}")
    planner, fixer = models.planner, models.fixer
    planner_name = {"spark_only": fixer.name}.get(setup, planner.name)
    fixer_name = {"wonder_only": planner.name}.get(setup, fixer.name)
    return {
        "planner": replace(planner, name=planner_name),
        "fixer": replace(fixer, name=fixer_name),
    }


class ModelWriter:
    """agent.Writer backed by the Meta Model API (llm.ModelClient)."""

    def __init__(self, client: Any, setup: str, models: ModelsConfig):
        self.client, self.setup = client, setup
        self.roles = models_for(setup, models)

    def write(self, role: Role, messages: list[dict[str, str]], tags: dict) -> ModelReply:
        model = self.roles[role]
        call = self.client.complete(
            model, messages, tags={**tags, "setup": self.setup}, text=RESPONSE_TEXT
        )
        try:
            response, error = parse_response(call.text), ""
        except ResponseError as e:
            response, error = None, str(e)
        return ModelReply(
            response=response,
            model=call.model,
            effort=call.reasoning_effort,
            cost_usd=call.cost_usd,
            input_tokens=call.input_tokens,
            cached_input_tokens=call.cached_input_tokens,
            output_tokens=call.output_tokens,
            error=error,
        )


def attempt_id(run_id: str, task_id: str, n: int) -> str:
    return f"{run_id}:{task_id}:{n}"


class DBRecorder:
    """agent.run_task's on_attempt callback: one attempts row and one evals row per attempt,
    written together in one transaction."""

    def __init__(self, db: ResultsDB, run_id: str, setup: str):
        self.db, self.run_id, self.setup = db, run_id, setup

    def __call__(self, record: AttemptRecord) -> None:
        reply = record.reply
        code = reply.response.code if reply.response else ""
        aid = attempt_id(self.run_id, record.result.task_id, record.n)
        self.db.record_attempt(
            Attempt(
                attempt_id=aid,
                run_id=self.run_id,
                task_id=record.result.task_id,
                n=record.n,
                setup=self.setup,
                model=reply.model,
                effort=reply.effort,
                input_tokens=reply.input_tokens,
                cached_input_tokens=reply.cached_input_tokens,
                output_tokens=reply.output_tokens,
                cost_usd=reply.cost_usd,
                code_hash="sha256:" + hashlib.sha256(code.encode()).hexdigest(),
                code=code,
                plan_summary=reply.response.plan_summary if reply.response else "",
                role=record.role,
            ),
            record.result,
        )
