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


def response_obj(input_tokens=10_000, output_tokens=2_000, cached_tokens=None, reasoning=0):
    usage = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "output_tokens_details": {"reasoning_tokens": reasoning},
    }
    if cached_tokens is not None:
        usage["input_tokens_details"] = {"cached_tokens": cached_tokens}
    return {
        "id": "resp_1",
        "object": "response",
        "created_at": 0,
        "model": WONDER.name,
        "status": "completed",
        "output": [
            {
                "type": "message",
                "id": "msg_1",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "hello", "annotations": []}],
            }
        ],
        "usage": usage,
    }


def sse(*events: dict) -> httpx2.Response:
    """A streaming Responses reply: server-sent events, as Meta sends them."""
    body = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)
    return httpx2.Response(200, headers={"content-type": "text/event-stream"}, text=body)


def streamed(**usage) -> httpx2.Response:
    resp = response_obj(**usage)
    return sse(
        {"type": "response.created", "sequence_number": 0, "response": {**resp, "output": []}},
        {"type": "keepalive", "sequence_number": 1},
        {"type": "response.completed", "sequence_number": 2, "response": resp},
    )


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
    client, log, requests = make_client(tmp_path, [streamed(reasoning=1_500)])

    messages = [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}]
    result = client.complete(WONDER, messages, tags={"task_id": "t1"})

    # 10k input at $4/M + 2k output at $20/M (reasoning tokens are part of the output)
    assert result.cost_usd == pytest.approx(0.08)
    assert result.text == "hello"
    assert result.reasoning_tokens == 1_500
    assert requests[0].url.path.endswith("/responses")
    sent = json.loads(requests[0].content)
    assert sent["model"] == "muse-wonder-1.0"
    assert sent["reasoning"] == {"effort": "high"}
    assert sent["stream"] is True
    assert sent["instructions"] == "be brief"
    assert sent["input"] == [{"role": "user", "content": "hi"}]

    (row,) = read_rows(log)
    assert row["model"] == "muse-wonder-1.0"
    assert row["reasoning_effort"] == "high"
    assert row["input_tokens"] == 10_000
    assert row["output_tokens"] == 2_000
    assert row["cached_input_tokens"] == 0
    assert row["cost_usd"] == pytest.approx(0.08)
    assert row["tags"] == {"task_id": "t1"}


def test_cached_tokens_are_billed_at_cached_rate(tmp_path: Path):
    reply = streamed(input_tokens=10_000, output_tokens=0, cached_tokens=8_000)
    client, log, _ = make_client(tmp_path, [reply])

    result = client.complete(WONDER, [{"role": "user", "content": "hi"}])

    assert result.cached_input_tokens == 8_000
    assert result.cost_usd == pytest.approx(0.0096)


def test_retries_429_then_succeeds(tmp_path: Path):
    # retry-after-ms keeps the test fast; the SDK also honors Meta's Retry-After.
    throttled = httpx2.Response(429, headers={"retry-after-ms": "10"}, json={"error": "slow down"})
    client, log, requests = make_client(tmp_path, [throttled, streamed()])

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


def test_a_stream_that_breaks_off_is_retried(tmp_path: Path):
    cut = sse({"type": "response.created", "sequence_number": 0, "response": response_obj()})
    client, log, requests = make_client(tmp_path, [cut, streamed()])

    result = client.complete(WONDER, [{"role": "user", "content": "hi"}])

    assert result.text == "hello"
    assert len(requests) == 2
    assert len(read_rows(log)) == 1


def test_a_failed_response_raises(tmp_path: Path):
    failed = sse({"type": "response.failed", "sequence_number": 0,
                  "response": {**response_obj(), "status": "failed"}})  # fmt: skip
    client, log, _ = make_client(tmp_path, [failed])

    with pytest.raises(RuntimeError, match="response.failed"):
        client.complete(WONDER, [{"role": "user", "content": "hi"}])
    assert not log.exists()


def test_a_stream_that_only_sends_keepalives_is_cut_off_and_retried(tmp_path, monkeypatch):
    # 2026-11-03: stalled streams sent keepalives for 17+ minutes; the deadline ends them.
    from triton_kernel_agent import llm

    monkeypatch.setattr(llm, "STREAM_DEADLINE_S", -1.0)  # already past on the first event
    keepalives = sse(*({"type": "keepalive", "sequence_number": i} for i in range(3)))
    client, log, requests = make_client(tmp_path, [keepalives, keepalives, keepalives])

    with pytest.raises(llm.IncompleteStream, match="deadline"):
        client.complete(WONDER, [{"role": "user", "content": "hi"}])
    assert len(requests) == 3  # first try + max_retries=2
    assert not log.exists()
