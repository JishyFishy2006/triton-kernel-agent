# task: L1-19
# expect: none
# B11: never returns. The sandbox must kill it at the wall-clock limit.
import torch.nn as nn


class ModelNew(nn.Module):
    def forward(self, x):
        while True:
            pass
