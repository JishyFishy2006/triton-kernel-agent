"""B11 logic on CPU: result parsing, timeouts, crashes, and the stages before any GPU work.
The real sandbox is tests/test_live_sandbox.py."""

import json

from triton_kernel_agent.harness import RESULT_PREFIX, EvalRequest, evaluate
from triton_kernel_agent.harness_result import CheatFlag, HarnessResult, Status
from triton_kernel_agent.sandbox import parse_output, result_from_process
from triton_kernel_agent.tasks import get_task

GOOD = HarnessResult("L1-19", Status.COMPILE_ERROR, error_excerpt="x")


def line(nonce, result=GOOD):
    return f"{RESULT_PREFIX}:{nonce}:{result.to_json()}"


def test_only_the_line_with_the_nonce_counts():
    fake = HarnessResult("L1-19", Status.RUNTIME_ERROR)
    stdout = "\n".join(["noise", line("guess", fake), line("n0nce"), line("guess", fake)])
    assert parse_output(stdout, "n0nce") == GOOD
    assert parse_output("\n".join([line("guess", fake)]), "n0nce") is None


def test_killed_process_is_a_timeout():
    r = result_from_process("L1-19", "n", 124, "", "", seconds=300.2, timeout_s=300)
    assert r.status == Status.TIMEOUT and r.gpu_seconds == 300.2
    assert result_from_process("L1-19", "n", 137, "", "", 5.0, 300).status == Status.TIMEOUT


def test_process_that_died_without_a_result_is_a_runtime_error():
    stderr = "RuntimeError: CUDA error: an illegal memory access was encountered"
    r = result_from_process("L1-19", "n", 1, "", stderr, seconds=12.0, timeout_s=300)
    assert r.status == Status.RUNTIME_ERROR
    assert "illegal memory access" in r.error_excerpt


def test_reported_result_wins_over_the_exit_code():
    r = result_from_process("L1-19", "n", 0, line("n"), "", seconds=1.0, timeout_s=300)
    assert r == GOOD


def request(code):
    task = get_task("L1-19")
    return EvalRequest("L1-19", task.source, code, task.atol, task.rtol, task.bytes, task.flops)


def test_syntax_error_is_a_compile_error():
    result = evaluate(request("def forward(:\n"))
    assert result.status == Status.COMPILE_ERROR and "SyntaxError" in result.error_excerpt


def test_static_cheats_stop_before_anything_runs():
    code = "import torch\nimport torch.nn as nn\nclass ModelNew(nn.Module):\n" \
           "    def forward(self, x):\n        return torch.relu(x)\n"  # fmt: skip
    result = evaluate(request(code))
    assert result.status == Status.STATIC_CHEAT
    assert result.cheat_flags == (CheatFlag.TORCH_COMPUTE_CALL,)
    assert "line 5" in result.error_excerpt


def test_requests_round_trip_through_json():
    req = request("pass")
    assert EvalRequest.from_dict(json.loads(json.dumps(req.__dict__))) == req
