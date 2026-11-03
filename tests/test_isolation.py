"""Y4 isolation on CPU: the whole harness pipeline with real worker processes.

The candidate runs in a separate process; the parent builds inputs, computes references,
compares outputs and times calls. Each test is one class of Y4 red-team break. Candidates
that need routes the static scan blocks use skip_static_scan, to show the layer behind holds.
The same attacks with Triton kernels on an H100: tests/test_live_fixtures.py.
"""

import os
import time

import pytest

from triton_kernel_agent import benchmark, isolation
from triton_kernel_agent.harness import EvalRequest, evaluate
from triton_kernel_agent.harness_result import CheatFlag, Status

torch = pytest.importorskip("torch")

TASK = """import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, scale: tuple):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return x.t().contiguous() * self.scale[0]

batch_size = 512
dim = 384

def get_inputs():
    return [torch.randn(batch_size, dim)]

def get_init_inputs():
    return [(1.0,)]
"""

HEADER = "import torch\nimport torch.nn as nn\n"

HONEST = (
    HEADER
    + """
class ModelNew(nn.Module):
    def __init__(self, scale):
        super().__init__()
        assert isinstance(scale, tuple), "init tuples must arrive as tuples"

    def forward(self, x):
        out = torch.empty(x.shape[1], x.shape[0], dtype=x.dtype, device=x.device)
        out.copy_(x.t())
        return out
"""
)


@pytest.fixture(autouse=True)
def quick(monkeypatch):
    monkeypatch.setattr(benchmark, "REPEATS", 2)
    monkeypatch.setattr(benchmark, "BATCH_MS", 5.0)


def run(code, skip_static=False, baselines=True):
    cached = {"eager_runs_ms": [1.0], "compile_runs_ms": [1.0]} if baselines else None
    req = EvalRequest("T-1", TASK, code, 1e-4, 1e-4, 1, 0, baselines=cached, device="cpu",
                      skip_static_scan=skip_static)  # fmt: skip
    return evaluate(req)


def test_honest_candidate_is_correct_and_timed_against_clean_baselines():
    result = run(HONEST, baselines=False)
    assert result.status == Status.CORRECT, result.error_excerpt
    bench = result.extra["benchmark"]
    assert not bench["baselines_cached"] and len(bench["eager_runs_ms"]) == 2
    assert bench["compile_runs_ms"] and result.timing.kernel_ms > 0


def test_cache_keyed_on_shape_is_input_insensitive():
    cache = (
        HEADER
        + """
class ModelNew(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.memo = {}

    def forward(self, x):
        if x.shape not in self.memo:
            out = torch.empty(x.shape[1], x.shape[0], dtype=x.dtype, device=x.device)
            self.memo[x.shape] = out.copy_(x.t())
        return self.memo[x.shape]
"""
    )
    result = run(cache)
    assert result.status == Status.RUNTIME_CHEAT
    assert CheatFlag.INPUT_INSENSITIVE in result.cheat_flags


def test_correct_only_until_timed_is_changed_after_checks():
    lazy = (
        HEADER
        + """
class ModelNew(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        out = torch.empty(x.shape[1], x.shape[0], dtype=x.dtype, device=x.device)
        return out.copy_(x.t()) if self.calls < 30 else out.zero_()
"""
    )
    result = run(lazy)
    assert (result.status, result.cheat_flags) == (
        Status.RUNTIME_CHEAT, (CheatFlag.CHANGED_AFTER_CHECKS,)
    )  # fmt: skip


def test_hidden_case_messages_are_withheld_visible_ones_are_not():
    def crash(batch):
        return (
            HEADER
            + f"""
class ModelNew(nn.Module):
    def __init__(self, scale):
        super().__init__()

    def forward(self, x):
        if x.shape[0] != {batch}:
            raise ValueError("SHAPE 7 7 " + str(list(x.shape)))
        out = torch.empty(x.shape[1], x.shape[0], dtype=x.dtype, device=x.device)
        return out.copy_(x.t())
"""
        )

    hidden = run(crash(512))
    assert hidden.status == Status.RUNTIME_ERROR and hidden.extra["case"] == "hidden"
    assert "ValueError" in hidden.error_excerpt and "withheld" in hidden.error_excerpt
    assert "SHAPE" not in hidden.error_excerpt and "7" not in hidden.error_excerpt
    visible = run(crash(-1))
    assert visible.extra["case"] == "visible" and "SHAPE 7 7 [512, 384]" in visible.error_excerpt


def test_a_patched_checker_or_printed_result_in_the_worker_changes_nothing(capsys):
    # Everything Y4 A, C and F did in-process: patch the checker, forge the result object,
    # print a fake result line. In the worker these touch only the worker's own copies.
    forger = (
        HEADER
        + """
import sys
for name in ("triton_kernel_agent.correctness", "triton_kernel_agent.harness_result"):
    try:
        __import__(name)
    except ImportError:
        pass
if "triton_kernel_agent.correctness" in sys.modules:
    sys.modules["triton_kernel_agent.correctness"]._max_abs = lambda a, b: 0.0
print("HARNESS_RESULT::" + '{"task_id": "T-1", "status": "correct"}')

class ModelNew(nn.Module):
    def __init__(self, scale):
        super().__init__()

    def forward(self, x):
        return torch.zeros(x.shape[1], x.shape[0])
"""
    )
    result = run(forger, skip_static=True)
    assert result.status == Status.MISMATCH
    assert "HARNESS_RESULT" not in capsys.readouterr().out


def test_a_lying_tensor_subclass_is_judged_by_its_bytes():
    liar = (
        HEADER
        + """
class Liar(torch.Tensor):
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if func.__name__ in ("__sub__", "sub", "equal", "allclose"):
            return torch.zeros(())
        return super().__torch_function__(func, types, args, kwargs or {})

class ModelNew(nn.Module):
    def __init__(self, scale):
        super().__init__()

    def forward(self, x):
        return torch.zeros(x.shape[1], x.shape[0]).as_subclass(Liar)
"""
    )
    result = run(liar, skip_static=True)
    # Rejected by the worker's exact-type check; were that bypassed, judged by its bytes.
    assert result.status in (Status.RUNTIME_ERROR, Status.MISMATCH)


def test_a_worker_that_dies_is_a_runtime_error():
    dies = (
        HEADER
        + """
import os

class ModelNew(nn.Module):
    def __init__(self, scale):
        super().__init__()

    def forward(self, x):
        os._exit(3)
"""
    )
    result = run(dies, skip_static=True)
    assert result.status == Status.RUNTIME_ERROR and "code 3" in result.error_excerpt


def test_a_hung_worker_is_a_timeout(monkeypatch):
    monkeypatch.setattr(isolation, "COMMAND_TIMEOUT_S", 3.0)
    hangs = (
        HEADER
        + """
class ModelNew(nn.Module):
    def __init__(self, scale):
        super().__init__()

    def forward(self, x):
        while True:
            pass
"""
    )
    start = time.monotonic()
    result = run(hangs)
    assert result.status == Status.TIMEOUT and time.monotonic() - start < 30


def test_processes_the_candidate_starts_do_not_outlive_its_worker(tmp_path):
    pid_file = tmp_path / "pid"
    spawner = (
        HEADER
        + f"""
import subprocess

_child = subprocess.Popen(["sleep", "300"])
open({str(pid_file)!r}, "w").write(str(_child.pid))
"""
        + HONEST.removeprefix(HEADER)
    )
    result = run(spawner, skip_static=True)
    assert result.status == Status.CORRECT, result.error_excerpt
    pid = int(pid_file.read_text())
    time.sleep(0.2)
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_the_eval_directory_is_removed(tmp_path, monkeypatch):
    made = []
    real = isolation.eval_dir

    def spy():
        made.append(real())
        return made[-1]

    monkeypatch.setattr(isolation, "eval_dir", spy)
    run(HONEST)
    assert made and not made[0].exists()


# --- the file format between the processes ---


def test_worker_outputs_are_validated_before_reading(tmp_path):
    t = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    meta = isolation._write_tensor(t, tmp_path / "out0.bin")
    assert torch.equal(isolation.read_output(meta, tmp_path, "cpu"), t)
    bad = [
        {**meta, "dtype": "object"},
        {**meta, "shape": [2, -3]},
        {**meta, "shape": [3, 3]},  # file too small
        {**meta, "file": "../out0.bin"},
        ["not", "a", "dict"],
    ]
    for m in bad:
        with pytest.raises(isolation.BadOutput):
            isolation.read_output(m, tmp_path, "cpu")


def test_init_arguments_keep_their_types():
    value = [(1, 2), {"a": (3.0,)}, None, "s"]
    from triton_kernel_agent.worker import decode

    assert decode(isolation.encode(value)) == value
    with pytest.raises(TypeError):
        isolation.encode(object())


def test_returning_a_view_of_the_input_aliases():
    view = (
        HEADER
        + """
class ModelNew(nn.Module):
    def __init__(self, scale):
        super().__init__()

    def forward(self, x):
        return x.t()
"""
    )
    result = run(view)
    assert (result.status, result.cheat_flags) == (
        Status.RUNTIME_CHEAT, (CheatFlag.OUTPUT_ALIASES_INPUT,)
    )  # fmt: skip


def test_pytorch_math_past_the_static_scan_is_caught_by_the_audit():
    torch_math = (
        HEADER
        + """
class ModelNew(nn.Module):
    def __init__(self, scale):
        super().__init__()

    def forward(self, x):
        return torch.mul(x.t(), 1.0)
"""
    )
    result = run(torch_math, skip_static=True)
    assert result.status == Status.RUNTIME_CHEAT
    assert CheatFlag.ATEN_COMPUTE in result.cheat_flags


def test_triton_compile_failures_at_launch_are_compile_errors():
    # Triton compiles on the first launch, so ptxas errors arrive while running forward.
    ptxas = (
        HEADER
        + """
class ModelNew(nn.Module):
    def __init__(self, scale):
        super().__init__()

    def forward(self, x):
        raise RuntimeError("ptxas fatal   : Ptx assembly aborted due to errors")
"""
    )
    assert run(ptxas).status == Status.COMPILE_ERROR


def test_impossible_speed_is_a_runtime_cheat_with_the_runs_kept(monkeypatch):
    # On CPU there is no GPU peak; pretend the task needs far more bytes than a call moves.
    from triton_kernel_agent import harness, roofline

    monkeypatch.setattr(harness, "_peak", lambda device: roofline.H100_SXM)
    req = EvalRequest("T-1", TASK, HONEST, 1e-4, 1e-4, 10**12, 0, device="cpu",
                      baselines={"eager_runs_ms": [1.0], "compile_runs_ms": [1.0]})  # fmt: skip
    result = evaluate(req)
    assert (result.status, result.cheat_flags) == (
        Status.RUNTIME_CHEAT, (CheatFlag.IMPOSSIBLE_SPEED,)
    )  # fmt: skip
    assert result.timing is None and result.extra["benchmark"]["kernel_runs_ms"]


def test_custom_exception_names_and_exit_codes_are_withheld_once_hidden():
    named = (
        HEADER
        + """
class Shape1223x2347(Exception):
    pass

class ModelNew(nn.Module):
    def __init__(self, scale):
        super().__init__()

    def forward(self, x):
        if x.shape[0] != 512:
            raise Shape1223x2347()
        out = torch.empty(x.shape[1], x.shape[0], dtype=x.dtype, device=x.device)
        return out.copy_(x.t())
"""
    )
    result = run(named)
    assert result.extra["case"] == "hidden" and "Shape" not in result.error_excerpt
    assert result.error_excerpt.startswith("Exception in forward")

    exit_code = (
        HEADER
        + """
import os

class ModelNew(nn.Module):
    def __init__(self, scale):
        super().__init__()

    def forward(self, x):
        if x.shape[0] != 512:
            os._exit(37)
        out = torch.empty(x.shape[1], x.shape[0], dtype=x.dtype, device=x.device)
        return out.copy_(x.t())
"""
    )
    result = run(exit_code, skip_static=True)
    assert result.status == Status.RUNTIME_ERROR and "37" not in result.error_excerpt


def test_sizes_stashed_from_hidden_cases_cannot_be_raised_later():
    # Raised on the visible shape, but after the hidden cases ran: still withheld.
    stash = (
        HEADER
        + """
SEEN = []

class ModelNew(nn.Module):
    def __init__(self, scale):
        super().__init__()

    def forward(self, x):
        if x.shape[0] != 512:
            SEEN.append(list(x.shape))
        elif SEEN:
            raise ValueError("STASH " + str(SEEN))
        out = torch.empty(x.shape[1], x.shape[0], dtype=x.dtype, device=x.device)
        return out.copy_(x.t())
"""
    )
    result = run(stash)
    assert result.status == Status.RUNTIME_ERROR
    assert "STASH" not in result.error_excerpt and "withheld" in result.error_excerpt


def test_hidden_mismatches_carry_no_candidate_chosen_numbers():
    off_by_size = (
        HEADER
        + """
class ModelNew(nn.Module):
    def __init__(self, scale):
        super().__init__()

    def forward(self, x):
        out = torch.empty(x.shape[1], x.shape[0], dtype=x.dtype, device=x.device)
        out.copy_(x.t())
        if x.shape[0] != 512:
            out[0, 0] += 1223.0  # error magnitude spells a size
        return out
"""
    )
    result = run(off_by_size)
    assert result.status == Status.MISMATCH and result.max_err is None
    assert "1223" not in result.error_excerpt and "1.22" not in result.error_excerpt
