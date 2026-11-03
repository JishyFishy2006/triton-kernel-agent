

# --- L1-23 Softmax: harness wrapper (appended to tutorial_kernels.py) ---
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return softmax(x, dim=1)
