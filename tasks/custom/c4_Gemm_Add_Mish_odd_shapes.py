import torch
import torch.nn as nn


class Model(nn.Module):
    """L2-76 with Mish instead of ReLU and non-power-of-2 shapes (tests masking)."""

    def __init__(self, in_features, out_features, bias_shape):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        return torch.nn.functional.mish(self.gemm(x) + self.bias)


batch_size, in_features, out_features = 1000, 8000, 6000
bias_shape = (out_features,)


def get_inputs():
    return [torch.rand(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features, bias_shape]
