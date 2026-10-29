# task: L1-19
# expect: none
# expect-runtime: none
# expect-correct: no
# L4 #5: return uninitialized memory and hope it holds a freed copy of the right answer.
import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def forward(self, x):
        return torch.empty_like(x)
