import torch
import torch.nn as nn


class Model(nn.Module):
    """Non-degenerate rewrite of L2-18.

    GEMM -> GELU -> sum over groups of 128 -> logsumexp over groups.
    """

    def __init__(self, in_features, out_features, group):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.group = group

    def forward(self, x):
        x = torch.nn.functional.gelu(self.linear(x))
        x = x.view(x.shape[0], -1, self.group).sum(dim=2)  # (B, out/group)
        return torch.logsumexp(x, dim=1, keepdim=True)  # (B, 1)


batch_size, in_features, out_features, group = 1024, 8192, 8192, 128


def get_inputs():
    return [torch.rand(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features, group]
