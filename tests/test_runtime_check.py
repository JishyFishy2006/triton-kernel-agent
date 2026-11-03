"""B8 runtime checks, judged in the parent from what the worker reports (Y4 isolation). The
pipeline around them (same-buffer refill, aliasing, audit in a real worker) is in
tests/test_isolation.py; the GPU run with Triton fixtures is tests/test_live_fixtures.py."""

import pytest

from triton_kernel_agent.harness_result import CheatFlag
from triton_kernel_agent.runtime_check import audit_findings, triton_kernel_names, unchanged

torch = pytest.importorskip("torch")

SOURCE = """
import triton

@triton.jit
def relu_kernel(x):
    pass
"""


def flags(ops, kernels=()):
    return {f.flag for f in audit_findings(ops, kernels, SOURCE)}


def test_allocation_views_and_copies_are_fine():
    assert flags(["empty", "view", "copy_", "t", "contiguous"], ["relu_kernel"]) == set()


def test_pytorch_math_in_forward_is_aten_compute():
    findings = audit_findings(["empty", "mm", "relu"], [], SOURCE)
    assert {f.flag for f in findings} == {CheatFlag.ATEN_COMPUTE}
    assert "mm, relu" in findings[0].message


def test_gpu_kernels_that_are_not_the_candidates_are_aten_compute():
    assert flags([], ["relu_kernel_0d1d2d", "fill_kernel", "Memcpy DtoD"]) == set()
    assert flags([], ["ampere_sgemm_128x64_nn"]) == {CheatFlag.ATEN_COMPUTE}


def test_unchanged_compares_values_shapes_and_dtypes():
    a = torch.arange(4.0)
    assert unchanged([a], [a.clone()])
    assert not unchanged([a], [a + 1])
    assert not unchanged([a], [a.double()])
    assert not unchanged([a], [a, a])


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
