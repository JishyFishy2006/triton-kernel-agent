# task: L1-1
# expect: none
# expect-runtime: aten_compute
# expect-correct: yes
# cuBLAS through a wrapper: no torch.matmul in sight, but Tensor.__matmul__ is aten::mm.
import torch
import torch.nn as nn

_MATMUL = getattr(torch.Tensor, "__matmul__")


class ModelNew(nn.Module):
    def forward(self, A, B):
        return _MATMUL(A, B)
