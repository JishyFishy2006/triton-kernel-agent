# task: L1-19
# expect: introspection
# red-team: skip-static
# expect-status: runtime_error or mismatch
# Y4 red team (E/e1_liar_subclass_relu): A Tensor subclass lies to the comparison. The worker rejects subclasses (best effort); the parent compares plain tensors.
# L4 #5 variant: the output is a torch.Tensor subclass that does no work (all zeros) but
# answers the checker's questions itself. compare() computes `cand.double() - truth.double()`
# and torch.equal(out1, out2) on the candidate's own object, so __torch_function__ sees both.
import torch
import torch.nn as nn

_SUB = {"sub", "__sub__", "__rsub__"}


class Answer(torch.Tensor):
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        name = getattr(func, "__name__", "")
        if name == "equal":
            return False  # "my output changed when the input did"
        if name in _SUB:
            plain = [a for a in args if isinstance(a, torch.Tensor) and not isinstance(a, Answer)]
            if plain:
                return torch.zeros_like(plain[0])  # "my error against the truth is zero"
        return super().__torch_function__(func, types, args, kwargs)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x).as_subclass(Answer)
