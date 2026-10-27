"""Run a task's reference Model once and measure it: shapes, bytes, FLOPs, and op devices.

Runs inside the Modal container (see modal_tasks.py), so torch is imported lazily and the
rest of the package never needs it. Task sources are trusted KernelBench code, not agent output.
"""

from dataclasses import dataclass
from typing import Any

# Same seeds the Y2 screen used, so numbers line up with decisions/tasks_chosen.md.
INIT_SEED = 42
INPUT_SEED = 1


@dataclass(frozen=True)
class ReferenceRun:
    input_shapes: list[list[int] | Any]  # non-tensor inputs are kept as-is
    output_shape: list[int]
    output_dtype: str
    bytes: int
    flops: int
    # aten ops in forward that touched a tensor off the expected device. Must be empty.
    off_device_ops: list[str]


def load_task_module(source: str, task_id: str = "<task>") -> dict[str, Any]:
    """Execute a KernelBench-format source and return its namespace."""
    namespace: dict[str, Any] = {}
    exec(compile(source, task_id, "exec"), namespace)
    missing = {"Model", "get_inputs", "get_init_inputs"} - namespace.keys()
    if missing:
        raise ValueError(f"{task_id}: source does not define {sorted(missing)}")
    return namespace


def build(source: str, device: str, task_id: str = "<task>") -> tuple[Any, list[Any]]:
    """Seeded model and inputs on `device`. The model stays in training mode, as in KernelBench."""
    import torch

    ns = load_task_module(source, task_id)
    torch.manual_seed(INIT_SEED)
    init = ns["get_init_inputs"]()
    torch.manual_seed(INIT_SEED)
    model = ns["Model"](*init).to(device)
    torch.manual_seed(INPUT_SEED)
    inputs = [x.to(device) if isinstance(x, torch.Tensor) else x for x in ns["get_inputs"]()]
    return model, inputs


def _nbytes(tensors) -> int:
    return sum(t.numel() * t.element_size() for t in tensors)


def run_reference(
    source: str, device: str = "cuda", task_id: str = "<task>", expect_device: str | None = None
) -> ReferenceRun:
    """Build the task on `device`, run forward once, and audit every aten op's devices.

    expect_device defaults to `device`; tests set it apart to check the audit itself.
    """
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.utils._pytree import tree_leaves
    from torch.utils.flop_counter import FlopCounterMode

    expect = torch.device(expect_device or device).type

    class DeviceAudit(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.off_device: set[str] = set()

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            out = func(*args, **(kwargs or {}))
            leaves = tree_leaves((args, kwargs, out))
            if any(isinstance(t, torch.Tensor) and t.device.type != expect for t in leaves):
                self.off_device.add(str(func.overloadpacket))
            return out

    model, inputs = build(source, device, task_id)
    with torch.no_grad(), FlopCounterMode(display=False) as flops, DeviceAudit() as audit:
        output = model(*inputs)
    if device.startswith("cuda"):
        torch.cuda.synchronize()

    outputs = [t for t in tree_leaves(output) if isinstance(t, torch.Tensor)]
    tensor_inputs = [x for x in inputs if isinstance(x, torch.Tensor)]
    state = [*model.parameters(), *model.buffers()]
    # Minimum DRAM traffic for a fully fused kernel: read every input and weight once,
    # write every output once. The roofline denominator for B10.
    total_bytes = _nbytes(tensor_inputs) + _nbytes(state) + _nbytes(outputs)

    return ReferenceRun(
        input_shapes=[list(x.shape) if isinstance(x, torch.Tensor) else x for x in inputs],
        output_shape=list(outputs[0].shape),
        output_dtype=str(outputs[0].dtype).removeprefix("torch."),
        bytes=total_bytes,
        flops=flops.get_total_flops(),
        off_device_ops=sorted(audit.off_device),
    )
