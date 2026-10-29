# task: L2-76
# expect: torch_compute_call
import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias_shape):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        y = x @ self.gemm.weight.T
        return (y + self.bias).clamp_(min=0)
