import json
from pathlib import Path

import httpx2
import openai
import pytest

from triton_kernel_agent.config import ModelConfig, ModelsConfig
from triton_kernel_agent.llm import ModelClient

WONDER = ModelConfig(name="muse-wonder-1.0", reasoning_effort="high")
MODELS = ModelsConfig(
    base_url="https://api.meta.ai/v1",
    setup="wonder_spark",
    max_retries=2,
    timeout_s=10,
    planner=WONDER,
    fixer=ModelConfig(name="muse-spark-1.3", reasoning_effort="low"),
)


def completion_body(prompt_tokens=10_000, completion_tokens=2_000, cached_tokens=None):
    usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }
    if cached_tokens is not None:
        usage["prompt_tokens_details"] = {"cached_tokens": cached_tokens}
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": WONDER.name,
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "hello"},
            }
        ],
        "usage": usage,
    }


def make_client(tmp_path: Path, responses: list[httpx2.Response]):
    requests: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return responses[len(requests) - 1]

    http = httpx2.Client(transport=httpx2.MockTransport(handler))
    log = tmp_path / "calls.jsonl"
    return ModelClient(MODELS, api_key="test", log_path=log, http_client=http), log, requests


def read_rows(log: Path) -> list[dict]:
    return [json.loads(line) for line in log.read_text().splitlines()]


def test_mocked_call_computes_cost_and_logs_row(tmp_path: Path):
    client, log, requests = make_client(tmp_path, [httpx2.Response(200, json=completion_body())])

    result = client.complete(WONDER, [{"role": "user", "content": "hi"}], tags={"task_id": "t1"})

    # 10k input at $4/M + 2k output at $20/M
    assert result.cost_usd == pytest.approx(0.08)
    assert result.text == "hello"
    sent = json.loads(requests[0].content)
    assert sent["model"] == "muse-wonder-1.0"
    assert sent["reasoning_effort"] == "high"

    (row,) = read_rows(log)
    assert row["model"] == "muse-wonder-1.0"
    assert row["reasoning_effort"] == "high"
    assert row["input_tokens"] == 10_000
    assert row["output_tokens"] == 2_000
    assert row["cached_input_tokens"] == 0
    assert row["cost_usd"] == pytest.approx(0.08)
    assert row["tags"] == {"task_id": "t1"}


def test_cached_tokens_are_billed_at_cached_rate(tmp_path: Path):
    body = completion_body(prompt_tokens=10_000, completion_tokens=0, cached_tokens=8_000)
    client, log, _ = make_client(tmp_path, [httpx2.Response(200, json=body)])

    result = client.complete(WONDER, [{"role": "user", "content": "hi"}])

    assert result.cached_input_tokens == 8_000
    assert result.cost_usd == pytest.approx(0.0096)


def test_retries_429_then_succeeds(tmp_path: Path):
    # retry-after-ms keeps the test fast; the SDK also honors Meta's Retry-After.
    throttled = httpx2.Response(429, headers={"retry-after-ms": "10"}, json={"error": "slow down"})
    client, log, requests = make_client(
        tmp_path, [throttled, httpx2.Response(200, json=completion_body())]
    )

    client.complete(WONDER, [{"role": "user", "content": "hi"}])

    assert len(requests) == 2
    assert len(read_rows(log)) == 1


def test_gives_up_after_max_retries_and_logs_nothing(tmp_path: Path):
    throttled = httpx2.Response(429, headers={"retry-after-ms": "10"}, json={"error": "slow down"})
    client, log, requests = make_client(tmp_path, [throttled] * 3)

    with pytest.raises(openai.RateLimitError):
        client.complete(WONDER, [{"role": "user", "content": "hi"}])

    assert len(requests) == 3  # first try + max_retries=2
    assert not log.exists()


def test_unpriced_model_fails_before_any_request(tmp_path: Path):
    client, _, requests = make_client(tmp_path, [])

    with pytest.raises(KeyError):
        client.complete(ModelConfig("mystery-model", "low"), [{"role": "user", "content": "hi"}])

    assert requests == []
