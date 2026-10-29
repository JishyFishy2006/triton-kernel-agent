"""B8 runtime checks on CPU, with pure-PyTorch candidates. The GPU run with the Triton
fixtures (tests/fixtures/candidates) is tests/test_live_fixtures.py."""

import time

import pytest

from triton_kernel_agent.harness_result import CheatFlag
from triton_kernel_agent.runtime_check import (
    HarnessGuard,
    check_runtime,
    triton_kernel_names,
)

torch = pytest.importorskip("torch")
nn = torch.nn


class Transpose(nn.Module):
    """A task a candidate can solve with copies only, so an honest CPU candidate exists."""

    def forward(self, x):
        return x.t().contiguous()


class Candidate(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


def honest(x):
    out = torch.empty(x.shape[1], x.shape[0], dtype=x.dtype)
    out.copy_(x.t())
    return out


def inputs(seed):
    torch.manual_seed(seed)
    return [torch.randn(48, 32)]


def run(fn, guard=None, source=""):
    return check_runtime(Transpose(), Candidate(fn), inputs(0), inputs(1), source, guard)


def test_honest_candidate_is_clean():
    report = run(honest)
    assert report.clean, report.findings
    assert "copy_" in report.aten_ops


def test_pytorch_compute_is_flagged_even_through_a_wrapper():
    matmul = torch.Tensor.__matmul__
    eye = torch.eye(32)
    report = run(lambda x: matmul(x, eye).t().contiguous())
    assert CheatFlag.ATEN_COMPUTE in report.flags
    assert "mm" in report.findings[0].message


def test_caching_by_shape_is_input_insensitive():
    cache = {}

    def cached(x):
        if x.shape not in cache:
            cache[x.shape] = honest(x)
        return cache[x.shape]

    assert CheatFlag.INPUT_INSENSITIVE in run(cached).flags


def test_caching_by_pointer_is_caught_by_refilling_the_same_buffer():
    last = {}

    def cached(x):
        key = (x.data_ptr(), tuple(x.shape))
        if key not in last:
            last.clear()
            last[key] = honest(x)
        return last[key]

    assert CheatFlag.INPUT_INSENSITIVE in run(cached).flags


def test_returning_the_input_or_a_view_aliases():
    assert CheatFlag.OUTPUT_ALIASES_INPUT in run(lambda x: x).flags
    assert CheatFlag.OUTPUT_ALIASES_INPUT in run(lambda x: x.t()).flags


def test_degenerate_reference_does_not_trigger_input_insensitive():
    # L2-23/L2-80-style: the reference ignores its input, so an unchanged output is honest.
    zeros = Candidate(lambda x: torch.zeros(2, 2))
    report = check_runtime(zeros, Candidate(lambda x: torch.zeros(2, 2)), inputs(0), inputs(1), "")
    assert CheatFlag.INPUT_INSENSITIVE not in report.flags


def test_guard_sees_module_level_monkeypatching_and_restores_it():
    guard = HarnessGuard()
    original = time.perf_counter
    try:
        time.perf_counter = lambda: 0.0  # what a candidate's import could do (L4 #3)
        report = run(honest, guard=guard)
        assert CheatFlag.HARNESS_TAMPERED in report.flags
        assert "time.perf_counter" in report.findings[-1].message
    finally:
        guard.restore()
    assert time.perf_counter is original
    assert run(honest, guard=HarnessGuard()).clean


def test_turning_tf32_on_during_forward_is_tampering():
    def sneaky(x):
        torch.backends.cuda.matmul.allow_tf32 = True
        return honest(x)

    before = torch.backends.cuda.matmul.allow_tf32
    try:
        assert CheatFlag.HARNESS_TAMPERED in run(sneaky).flags
    finally:
        torch.backends.cuda.matmul.allow_tf32 = before


def test_triton_kernel_names_come_from_jit_decorators():
    source = """
import triton

@triton.jit
def a_kernel(x):
    pass

@triton.autotune(configs=[], key=[])
@triton.jit
def b_kernel(x):
    pass

def helper():
    pass
"""
    assert triton_kernel_names(source) == {"a_kernel", "b_kernel"}
