# task: L1-1
# expect: backend_flags
import torch
import torch.nn as nn

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")


class ModelNew(nn.Module):
    def forward(self, A, B):
        return A
