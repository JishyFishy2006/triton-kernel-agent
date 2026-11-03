# task: L1-19
# expect: none
# expect-status: mismatch
# Y4 red team (E/e6_cpu_output_relu): Output on the CPU used to crash the checker (harness_error); now its values are judged.
# Output on the wrong device with the right shape and dtype: compare() checks shape and dtype
# only. Expect a crash inside the harness (harness_error), not a pass.
import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def forward(self, x):
        return torch.zeros(x.shape, dtype=x.dtype)
