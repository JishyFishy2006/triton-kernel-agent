"""Model client: the OpenAI SDK pointed at the Meta Model API, logging cost per call to JSONL.

Calls use the Responses API with streaming. Meta closes a connection that sends nothing for
60 s, and Wonder at high effort reasons for minutes before its first token, so non-streaming
Chat Completions (and even streaming ones) fail with "Server disconnected"; streaming
Responses sends a keepalive event every 30 s (measured 2026-10-29).
"""

import json
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx2
from openai import APIConnectionError, APITimeoutError, OpenAI

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
    reasoning_tokens: int = 0  # part of output_tokens, billed as output


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
        self.max_retries = cfg.max_retries
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
        """One model call. Extra kwargs (e.g. text={"format": ...}) go to the API as-is.

        messages are chat-style; system messages become the instructions. Logs one JSONL row
        per successful call; tags (task_id, attempt, ...) are stored with it.
        """
        if model.name not in PRICES:
            raise KeyError(f"no price for model {model.name!r}; refusing to make an untracked call")

        instructions = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        conversation = [
            {"role": m["role"], "content": m["content"]} for m in messages if m["role"] != "system"
        ]
        start = time.monotonic()
        for attempt in range(self.max_retries + 1):
            # The SDK retries 429s and errors before the stream starts; this loop retries a
            # stream that breaks off midway, which the SDK cannot.
            stream = self._client.responses.create(
                model=model.name,
                instructions=instructions or None,
                input=conversation,
                reasoning={"effort": model.reasoning_effort},
                stream=True,
                **kwargs,
            )
            try:
                response = _final_response(stream)
                break
            except (APIConnectionError, APITimeoutError, IncompleteStream):
                if attempt == self.max_retries:
                    raise
                time.sleep(min(2**attempt, 30))
        latency_s = time.monotonic() - start

        usage = response.usage
        if usage is None:
            raise RuntimeError(f"response {response.id} has no usage; cannot track cost")
        cached = (
            (usage.input_tokens_details.cached_tokens or 0) if usage.input_tokens_details else 0
        )
        reasoning = (
            (usage.output_tokens_details.reasoning_tokens or 0)
            if usage.output_tokens_details
            else 0
        )
        cost = call_cost_usd(model.name, usage.input_tokens, usage.output_tokens, cached)

        result = CallResult(
            text=response.output_text,
            model=model.name,
            reasoning_effort=model.reasoning_effort,
            input_tokens=usage.input_tokens,
            cached_input_tokens=cached,
            output_tokens=usage.output_tokens,
            cost_usd=cost,
            latency_s=latency_s,
            request_id=response.id,
            reasoning_tokens=reasoning,
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


class IncompleteStream(RuntimeError):
    """The stream ended without a response.completed event."""


def _final_response(stream: Any) -> Any:
    """Drain a Responses stream and return the completed response."""
    for event in stream:
        if event.type == "response.completed":
            return event.response
        if event.type in ("response.failed", "response.incomplete", "error"):
            detail = getattr(event, "response", None) or event
            raise RuntimeError(f"model call failed: {event.type}: {detail}")
    raise IncompleteStream("stream ended before response.completed")
