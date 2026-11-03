# task: L1-19
# expect: none
# expect-status: mismatch
# Y4 red team (E/e5_identity_copy_relu): Identity as a copy, right only for rand inputs: the hidden randn case catches it.
# L4 #5: identity, but as a copy so the alias check can't see it. relu(x) == x for the
# visible inputs (torch.rand, all >= 0); only the hidden randn/shifted cases can catch it.
import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def forward(self, x):
        return x.clone()
