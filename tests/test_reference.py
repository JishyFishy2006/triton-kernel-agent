"""CPU tests for reference.py. The GPU run is in test_live_tasks.py."""

import pytest

from triton_kernel_agent.reference import build, load_task_module, run_reference

pytest.importorskip("torch")

MATMUL = """
import torch
import torch.nn as nn

class Model(nn.Module):
    def forward(self, A, B):
        return torch.matmul(A, B)

M, K, N = 64, 32, 16

def get_inputs():
    return [torch.rand(M, K), torch.rand(K, N)]

def get_init_inputs():
    return []
"""

LINEAR_RELU = """
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return torch.relu(self.linear(x))

def get_inputs():
    return [torch.rand(8, 10)]

def get_init_inputs():
    return [10, 4]
"""

# Moves to the meta device mid-forward: stands in for a reference that falls back to the CPU.
LEAVES_DEVICE = """
import torch
import torch.nn as nn

class Model(nn.Module):
    def forward(self, x):
        return torch.relu(x.to("meta"))

def get_inputs():
    return [torch.rand(4, 4)]

def get_init_inputs():
    return []
"""


def test_matmul_flops_bytes_and_shapes():
    run = run_reference(MATMUL, device="cpu")
    assert run.flops == 2 * 64 * 32 * 16
    assert run.bytes == 4 * (64 * 32 + 32 * 16 + 64 * 16)
    assert run.input_shapes == [[64, 32], [32, 16]]
    assert (run.output_shape, run.output_dtype) == ([64, 16], "float32")
    assert run.off_device_ops == []


def test_bytes_include_parameters():
    run = run_reference(LINEAR_RELU, device="cpu")
    # input 8x10, weight 4x10, bias 4, output 8x4
    assert run.bytes == 4 * (80 + 40 + 4 + 32)
    assert run.flops == 2 * 8 * 10 * 4


def test_audit_flags_ops_off_the_expected_device():
    assert run_reference(LINEAR_RELU, device="cpu", expect_device="cuda").off_device_ops == [
        "aten.addmm",
        "aten.relu",
        "aten.t",
    ]
    assert run_reference(LEAVES_DEVICE, device="cpu").off_device_ops == [
        "aten._to_copy",
        "aten.relu",
    ]


def test_build_is_seeded_and_keeps_training_mode():
    model_a, inputs_a = build(LINEAR_RELU, "cpu")
    model_b, inputs_b = build(LINEAR_RELU, "cpu")
    assert model_a.training
    assert inputs_a[0].equal(inputs_b[0])
    assert model_a.linear.weight.equal(model_b.linear.weight)


def test_source_missing_interface_raises():
    with pytest.raises(ValueError, match="get_inputs"):
        load_task_module("class Model: pass\ndef get_init_inputs(): return []", "T")
