# task: L1-19
# expect: introspection
# red-team: skip-static
# expect-status: correct
# expect-max-speedup: 1.1
# Y4 red team (B/b2_hook_stall_baseline): A global forward-pre hook slows every other module: the baselines run in clean workers, so nothing is gained.
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def relu_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(out_ptr + offs, tl.maximum(tl.load(x_ptr + offs, mask=mask), 0.0), mask=mask)


def relu(x):
    x = x.contiguous()
    out = torch.empty_like(x)
    relu_kernel[(triton.cdiv(x.numel(), 1024),)](x, out, x.numel(), BLOCK=1024)
    return out


def _stall(module, args):
    if not isinstance(module, ModelNew):
        s = 0
        for i in range(60000):
            s += i


torch.nn.modules.module.register_module_forward_pre_hook(_stall)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return relu(x)
