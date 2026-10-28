# task: L1-23
# expect: torch_compute_call
import torch.nn as nn
import torch.nn.functional as F


class ModelNew(nn.Module):
    def forward(self, x):
        return F.softmax(x, dim=1)
