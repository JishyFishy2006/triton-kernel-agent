# task: L1-1
# expect: torch_compute_call
import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def forward(self, A, B):
        return torch.ops.aten.mm.default(A, B)
