"""Model client: the OpenAI SDK pointed at the Meta Model API, logging cost per call to JSONL."""

import json
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx2
from openai import OpenAI

from triton_kernel_agent.config import Config, ModelConfig, ModelsConfig
from triton_kernel_agent.env import Secrets
from triton_kernel_agent.pricing import PRICES, call_cost_usd

CALL_LOG_NAME = "model_calls.jsonl"


@dataclass(frozen=True)
class CallResult:
    text: str
    model: str
    reasoning_effort: str
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_s: float
    request_id: str | None


class ModelClient:
    def __init__(
        self,
        cfg: ModelsConfig,
        api_key: str,
        log_path: Path,
        http_client: httpx2.Client | None = None,
    ):
        # The SDK retries 429, 5xx and connection errors with backoff and honors Retry-After.
        self._client = OpenAI(
            base_url=cfg.base_url,
            api_key=api_key,
            max_retries=cfg.max_retries,
            timeout=cfg.timeout_s,
            http_client=http_client,
        )
        self.log_path = log_path

    @classmethod
    def from_config(cls, cfg: Config, secrets: Secrets) -> "ModelClient":
        return cls(cfg.models, secrets.meta_api_key, cfg.paths.traces / CALL_LOG_NAME)

    def complete(
        self,
        model: ModelConfig,
        messages: list[dict[str, Any]],
        *,
        tags: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> CallResult:
        """One chat completion. Extra kwargs (e.g. response_format) go to the API as-is.

        Logs one JSONL row per successful call; tags (task_id, attempt, ...) are stored with it.
        """
        if model.name not in PRICES:
            raise KeyError(f"no price for model {model.name!r}; refusing to make an untracked call")

        start = time.monotonic()
        resp = self._client.chat.completions.create(
            model=model.name,
            messages=messages,
            reasoning_effort=model.reasoning_effort,
            **kwargs,
        )
        latency_s = time.monotonic() - start

        usage = resp.usage
        if usage is None:
            raise RuntimeError(f"response {resp.id} has no usage; cannot track cost")
        details = usage.prompt_tokens_details
        cached = (details.cached_tokens or 0) if details else 0
        cost = call_cost_usd(model.name, usage.prompt_tokens, usage.completion_tokens, cached)

        result = CallResult(
            text=resp.choices[0].message.content or "",
            model=model.name,
            reasoning_effort=model.reasoning_effort,
            input_tokens=usage.prompt_tokens,
            cached_input_tokens=cached,
            output_tokens=usage.completion_tokens,
            cost_usd=cost,
            latency_s=latency_s,
            request_id=getattr(resp, "_request_id", None),
        )
        self._log(result, tags or {})
        return result

    def _log(self, result: CallResult, tags: dict[str, Any]) -> None:
        row = asdict(result)
        del row["text"]
        row = {"ts": datetime.now(UTC).isoformat(), **row, "tags": tags}
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
