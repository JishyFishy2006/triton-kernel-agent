import torch
import torch.nn as nn


class Model(nn.Module):
    """L2-99 reordered: softmax over the row first, then GELU."""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = torch.softmax(self.linear(x), dim=1)
        return torch.nn.functional.gelu(x)


batch_size, in_features, out_features = 1024, 8192, 8192


def get_inputs():
    return [torch.rand(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features]
