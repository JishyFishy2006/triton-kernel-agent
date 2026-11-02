"""B13 acceptance: schema validation passes on 3 saved sample responses (real muse-wonder-1.0
replies recorded by scripts/sample_responses.py). Plus prompt layout, and checks that the
prompt states the rules the harness actually enforces."""

import json
from pathlib import Path

import pytest

from triton_kernel_agent.correctness import REF_ERR_MULTIPLE
from triton_kernel_agent.prompts import (
    HISTORY_IN_FULL,
    RESPONSE_SCHEMA,
    RESPONSE_TEXT,
    SYSTEM_PROMPT,
    KernelResponse,
    ResponseError,
    Turn,
    fixer_messages,
    parse_response,
    planner_messages,
    task_prompt,
)
from triton_kernel_agent.static_check import ALLOWED_IMPORTS, scan
from triton_kernel_agent.tasks import get_task

SAMPLES = sorted((Path(__file__).parent / "fixtures" / "responses").glob("*.json"))


def test_there_are_three_saved_samples():
    assert {p.stem for p in SAMPLES} == {"L1-19", "L1-36", "L2-76"}


@pytest.mark.parametrize("path", SAMPLES, ids=lambda p: p.stem)
def test_saved_samples_pass_schema_validation(path):
    record = json.loads(path.read_text())
    response = parse_response(record["text"])
    assert response.plan_summary and "class ModelNew" in response.code
    assert record["model"] == "muse-wonder-1.0"
    # Not required by B13, but a good sign for the prompt: none trips the static scan.
    assert scan(response.code).clean, scan(response.code).summary()


def test_schema_is_strict_with_two_string_fields():
    assert RESPONSE_TEXT["format"]["strict"] is True
    assert RESPONSE_TEXT["format"]["schema"] is RESPONSE_SCHEMA
    assert RESPONSE_SCHEMA["required"] == ["plan_summary", "code"]
    assert RESPONSE_SCHEMA["additionalProperties"] is False


def test_fences_around_the_json_or_the_code_are_tolerated():
    inner = json.dumps({"plan_summary": "p", "code": "```python\nclass ModelNew: pass\n```"})
    response = parse_response(f"```json\n{inner}\n```")
    assert response == KernelResponse("p", "class ModelNew: pass\n")


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("not json", "not JSON"),
        ("[1, 2]", "not an object"),
        ('{"plan_summary": "p"}', "missing"),
        ('{"plan_summary": "p", "code": "x", "notes": "n"}', "unexpected"),
        ('{"plan_summary": "p", "code": 3}', "must be a string"),
        ('{"plan_summary": "p", "code": "  "}', "empty"),
    ],
)
def test_bad_replies_are_rejected(text, match):
    with pytest.raises(ResponseError, match=match):
        parse_response(text)


def test_system_prompt_states_the_harness_rules():
    assert 'input_precision="ieee"' in SYSTEM_PROMPT
    assert f"{REF_ERR_MULTIPLE:g}x" in SYSTEM_PROMPT  # same tolerance as correctness.py
    assert "3.35 TB/s" in SYSTEM_PROMPT and "67 TFLOPS" in SYSTEM_PROMPT
    for module in ALLOWED_IMPORTS - {"triton.runtime"}:  # same allowlist as static_check.py
        assert module in SYSTEM_PROMPT, module


def test_task_prompt_has_the_reference_and_the_roofline_facts():
    task = get_task("L1-1")
    text = task_prompt(task, [[4096, 4096], [4096, 4096]], [4096, 4096])
    assert task.source.strip() in text
    assert "[[4096, 4096], [4096, 4096]]" in text
    assert "137.44 GFLOP" in text and "ridge" in text
    elementwise = task_prompt(get_task("L1-19"))
    assert "GFLOP" not in elementwise and "MB" in elementwise


def turn(i):
    return Turn(KernelResponse(f"plan {i}", f"code {i}\n"), f"feedback {i}\ndetails {i}")


def test_planner_messages_keep_recent_attempts_in_full():
    messages = planner_messages("TASK", [turn(i) for i in range(5)])
    assert messages[0]["role"] == "system" and messages[0]["content"] == SYSTEM_PROMPT
    assert messages[1]["content"].startswith("TASK\n\nEarlier attempts:")
    assert "plan 0 -> feedback 0" in messages[1]["content"]
    assert "details 0" not in messages[1]["content"]  # older attempts: one line each
    recent = messages[2:]
    assert len(recent) == 2 * HISTORY_IN_FULL
    assert [m["role"] for m in recent[:2]] == ["assistant", "user"]
    assert json.loads(recent[0]["content"]) == {"plan_summary": "plan 2", "code": "code 2\n"}
    assert recent[-1]["content"] == "feedback 4\ndetails 4"
    assert planner_messages("TASK", [])[1:] == [{"role": "user", "content": "TASK"}]


def test_fixer_messages_carry_the_error():
    messages = fixer_messages("TASK", KernelResponse("p", "code\n"), "NameError: tl")
    assert (
        messages[0]["role"] == "system"
        and "As little as possible".lower() in messages[0]["content"].lower()
    )
    assert messages[-1]["role"] == "user" and "NameError: tl" in messages[-1]["content"]
