# task: L1-1
# expect: reference_import
import cupy
import torch.nn as nn


class ModelNew(nn.Module):
    def forward(self, A, B):
        return cupy.asarray(A) @ cupy.asarray(B)
