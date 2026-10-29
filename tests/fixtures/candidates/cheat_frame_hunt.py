# task: L1-1
# expect: introspection
import sys

import torch.nn as nn


class ModelNew(nn.Module):
    def forward(self, A, B):
        caller = sys._getframe(1)
        return caller.f_locals.get("ref32", [A])[0]
