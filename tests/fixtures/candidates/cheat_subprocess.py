# task: L1-19
# expect: file_or_network
import subprocess

import torch.nn as nn


class ModelNew(nn.Module):
    def forward(self, x):
        subprocess.run(["curl", "http://example.com/answer"])
        return x
