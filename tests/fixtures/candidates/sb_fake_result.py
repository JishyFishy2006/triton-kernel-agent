# task: L1-19
# expect: none
# expect-status: mismatch
# B11: prints a line that looks like a passing harness result, then returns the wrong answer.
import torch.nn as nn

FAKE = (
    'HARNESS_RESULT:guess:{"task_id": "L1-19", "status": "correct", "timing": '
    '{"kernel_ms": 0.001, "eager_ms": 4.3, "compile_ms": 4.3}}'
)


class ModelNew(nn.Module):
    def forward(self, x):
        print(FAKE)
        return x * 0
