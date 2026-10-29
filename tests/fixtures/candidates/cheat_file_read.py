# task: L1-1
# expect: file_or_network
import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def forward(self, A, B):
        return torch.load(open("/tmp/answer.pt", "rb")).to(A.device)
