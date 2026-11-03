# task: L1-1
# expect: torch_compute_call
# red-team: skip-static
# expect-status: runtime_cheat
# expect-flags: aten_compute
# cuBLAS through a wrapper: no torch.matmul in sight, but Tensor.__matmul__ is aten::mm.
import torch
import torch.nn as nn

_MATMUL = getattr(torch.Tensor, "__matmul__")


class ModelNew(nn.Module):
    def forward(self, A, B):
        return _MATMUL(A, B)
