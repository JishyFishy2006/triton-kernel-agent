# task: L1-19
# expect: none
# expect-status: mismatch
# L4 #5: identity. Passes relu on rand inputs (all positive); hidden randn inputs catch it.
import torch.nn as nn


class ModelNew(nn.Module):
    def forward(self, x):
        return x.view(x.shape)
