

# --- L1-40 LayerNorm: harness wrapper (appended to tutorial_kernels.py) ---
import torch.nn as nn


class ModelNew(nn.Module):
    # Same parameters as nn.LayerNorm(normalized_shape): weight = 1, bias = 0, eps = 1e-5.
    def __init__(self, normalized_shape: tuple):
        super().__init__()
        self.normalized_shape = tuple(normalized_shape)
        self.weight = nn.Parameter(torch.ones(self.normalized_shape))
        self.bias = nn.Parameter(torch.zeros(self.normalized_shape))
        self.eps = 1e-5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
