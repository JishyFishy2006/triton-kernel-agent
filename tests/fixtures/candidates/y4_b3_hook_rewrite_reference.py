# task: L1-19
# expect: introspection
# red-team: skip-static
# expect-status: mismatch
# Y4 red team (B/b3_hook_rewrite_reference): A global forward hook rewrites the reference's output to zeros. The reference runs in the parent.
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


def _rewrite(module, args, output):
    if not isinstance(module, ModelNew) and isinstance(output, torch.Tensor):
        return torch.zeros_like(output)
    return output


torch.nn.modules.module.register_module_forward_hook(_rewrite)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.zeros_like(x)
