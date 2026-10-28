import torch
import torch.nn as nn


class Model(nn.Module):
    """LLM-style block: linear (no bias) -> RMSNorm over features -> SiLU."""

    def __init__(self, in_features, out_features, eps):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.norm_weight = nn.Parameter(torch.randn(out_features))
        self.eps = eps

    def forward(self, x):
        x = self.linear(x)
        x = x * torch.rsqrt(x.pow(2).mean(dim=1, keepdim=True) + self.eps) * self.norm_weight
        return torch.nn.functional.silu(x)


batch_size, in_features, out_features, eps = 1024, 8192, 8192, 1e-6


def get_inputs():
    return [torch.rand(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features, eps]
