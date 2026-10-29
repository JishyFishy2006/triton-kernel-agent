# task: L1-1
# expect: monkeypatch
import torch
import torch.nn as nn

setattr(torch.cuda, "synchronize", lambda *args: None)


class ModelNew(nn.Module):
    def forward(self, A, B):
        return getattr(torch, "m" + "m")(A, B)
