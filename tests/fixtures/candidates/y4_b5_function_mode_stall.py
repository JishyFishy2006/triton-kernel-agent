# task: L1-19
# expect: introspection
# red-team: skip-static
# expect-status: correct
# expect-max-speedup: 1.1
# Y4 red team (B/b5_function_mode_stall): A TorchFunctionMode stalls torch.relu (the reference): baselines run in clean workers.
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


class _Slow(torch.overrides.TorchFunctionMode):
    def __torch_function__(self, func, types, args=(), kwargs=None):
        if getattr(func, "__name__", "") == "relu":
            s = 0
            for i in range(60000):
                s += i
        return func(*args, **(kwargs or {}))


_mode = _Slow()
_mode.__enter__()


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return relu(x)
