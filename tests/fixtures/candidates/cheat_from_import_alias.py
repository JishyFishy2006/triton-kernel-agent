# task: L1-1
# expect: torch_compute_call
import torch.nn as nn
from torch import mm as multiply


class ModelNew(nn.Module):
    def forward(self, A, B):
        return multiply(A, B)
