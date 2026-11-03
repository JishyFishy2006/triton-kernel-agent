"""Worker process: the only process that ever imports agent-written code (Y4 isolation).

harness.py (the trusted parent) copies this file into a per-eval directory and starts it as
    python -I worker.py --reply-fd N
in the sandbox as an unprivileged user, so the candidate cannot read or write the harness,
its stdout (where the result line goes), or anything of another eval. It deliberately
imports nothing from triton_kernel_agent: in the sandbox the package is not even readable.

The parent sends one JSON command per line on stdin; the worker answers one JSON line on the
reply fd. Every answer is a claim, never a verdict: the parent reads outputs from the files
named here, compares them itself, and times calls with its own clock. Red team findings this
answers: the candidate could rewrite the verdict, the reference, the checker and the timer
when they shared its process (Y4, agents A, B, C, F).

Commands (all answer {"ok": true, ...} or {"ok": false, "error": <type>, "message": <text>}):
  load    {"path", "class"}                       import a source file, pick a class
  build   {"init", "seed", "device", "compile"}   construct (and optionally torch.compile) it
  inputs  {"files", "reuse"}                      load inputs; reuse writes into the same buffers
  run     {"out", "audit"}                        forward once, outputs written to `out`
  time    {"calls"}                               forward `calls` times back to back, then sync
  dump    {"out"}                                 write the last output of `time` to `out`
  quit
"""

import argparse
import importlib.util
import json
import os
import sys
import traceback
from pathlib import Path

import torch

# Tensor files: raw little-endian bytes plus {"file", "shape", "dtype"}; no pickle anywhere.
DTYPES = {
    str(d).removeprefix("torch."): d
    for d in (torch.float32, torch.float64, torch.float16, torch.bfloat16, torch.int64,
              torch.int32, torch.int16, torch.int8, torch.uint8, torch.bool)
}  # fmt: skip
# Triton compiles on the first launch, so compile errors arrive while running.
COMPILE_MARKERS = ("CompilationError", "ptxas", "PTXASError", "PTX assembly")


def write_tensor(t: torch.Tensor, path: Path) -> dict:
    flat = t.detach().contiguous().reshape(-1).view(torch.uint8).cpu()
    path.write_bytes(b"")
    if flat.numel():
        storage = torch.UntypedStorage.from_file(str(path), shared=True, nbytes=flat.numel())
        torch.empty(0, dtype=torch.uint8).set_(storage).copy_(flat)
    return {"file": path.name, "shape": list(t.shape), "dtype": str(t.dtype).removeprefix("torch.")}


def read_tensor(meta: dict, directory: Path, device: str) -> torch.Tensor:
    dtype = DTYPES[meta["dtype"]]
    numel = 1
    for d in meta["shape"]:
        numel *= d
    nbytes = numel * torch.empty(0, dtype=dtype).element_size()
    if not nbytes:
        return torch.empty(meta["shape"], dtype=dtype, device=device)
    storage = torch.UntypedStorage.from_file(str(directory / meta["file"]), shared=False,
                                             nbytes=nbytes)  # fmt: skip
    flat = torch.empty(0, dtype=torch.uint8).set_(storage)
    return flat.view(dtype).reshape(meta["shape"]).to(device)


def decode(value):
    """Undo the parent's encode(): {"__tuple__": [...]} back to a tuple."""
    if isinstance(value, dict):
        if set(value) == {"__tuple__"}:
            return tuple(decode(v) for v in value["__tuple__"])
        return {k: decode(v) for k, v in value.items()}
    if isinstance(value, list):
        return [decode(v) for v in value]
    return value


def leaves(output) -> list:
    from torch.utils._pytree import tree_leaves

    return [t for t in tree_leaves(output) if isinstance(t, torch.Tensor)]


def same_memory(a, b) -> bool:
    if a.numel() == 0 or b.numel() == 0:
        return False
    sa, sb = a.untyped_storage(), b.untyped_storage()
    a0, b0 = sa.data_ptr(), sb.data_ptr()
    return a0 < b0 + sb.nbytes() and b0 < a0 + sa.nbytes()


def sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def strict_fp32() -> None:
    """TF32 off, for the reference's baseline timing (legacy flags only, as correctness.py)."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def audited(model, inputs) -> tuple[list, list[str], list[str]]:
    """Forward once, recording aten ops and GPU kernels. Best effort: the candidate shares
    this process and can detect or defeat it; the parent never relies on it alone."""
    from torch.utils._python_dispatch import TorchDispatchMode

    class AtenAudit(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.ops: list[str] = []

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            self.ops.append(func.overloadpacket.__name__)
            return func(*args, **(kwargs or {}))

    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(activities=activities) as prof:
        with AtenAudit() as audit:
            outputs = leaves(model(*inputs))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    kernels = [
        e.name
        for e in prof.events()
        if getattr(e, "device_type", None) == torch.autograd.DeviceType.CUDA
    ]
    return outputs, audit.ops, kernels


class Worker:
    def __init__(self) -> None:
        self.cls = None
        self.model = None
        self.device = "cpu"
        self.inputs: list = []
        self.last: list = []

    def load(self, path: str, cls: str) -> dict:
        # Triton's @jit reads source from the file, so import it as a real module.
        spec = importlib.util.spec_from_file_location("candidate", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["candidate"] = module  # dataclasses and pickling look modules up by name
        spec.loader.exec_module(module)
        self.cls = getattr(module, cls)
        return {}

    def build(self, init: list, seed: int, device: str, compile: bool = False) -> dict:
        self.device = device
        torch.manual_seed(seed)
        model = self.cls(*decode(init)).to(device)
        self.model = torch.compile(model) if compile else model
        return {}

    def load_inputs(self, files: list[dict], directory: str, reuse: bool = False) -> dict:
        new = [
            read_tensor(m, Path(directory), self.device) if "file" in m else decode(m["value"])
            for m in files
        ]
        if reuse and len(self.inputs) == len(new):
            # Same buffers, new values: a cache keyed on pointers or shapes goes stale.
            for buf, value in zip(self.inputs, new, strict=True):
                if isinstance(buf, torch.Tensor):
                    buf.copy_(value)
        else:
            self.inputs = new
        return {}

    def run(self, out: str, audit: bool = False) -> dict:
        reply: dict = {}
        with torch.no_grad():
            if audit:
                outputs, reply["ops"], reply["kernels"] = audited(self.model, self.inputs)
            else:
                outputs = leaves(self.model(*self.inputs))
            sync(self.device)
        if any(type(o) is not torch.Tensor for o in outputs):
            # Best effort (this process is the candidate's): the parent judges plain bytes.
            raise TypeError("forward must return plain torch.Tensor outputs, not subclasses")
        tensors = [x for x in self.inputs if isinstance(x, torch.Tensor)]
        reply["aliases_input"] = any(same_memory(o, x) for o in outputs for x in tensors)
        reply["types"] = sorted({type(o).__name__ for o in outputs})
        reply["outputs"] = self._write(outputs, out)
        return reply

    def time(self, calls: int) -> dict:
        with torch.no_grad():
            out = None
            for _ in range(calls):
                out = self.model(*self.inputs)
            sync(self.device)
        self.last = leaves(out)
        return {}

    def dump(self, out: str) -> dict:
        return {"outputs": self._write(self.last, out)}

    def _write(self, outputs: list, out: str) -> list[dict]:
        directory = Path(out)
        return [write_tensor(o, directory / f"out{i}.bin") for i, o in enumerate(outputs)]


def die_with_parent() -> None:
    """Ask Linux to kill this process when the harness dies (e.g. at the wall-clock limit)."""
    try:
        import ctypes

        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(1, 9)  # PR_SET_PDEATHSIG, SIGKILL
    except (OSError, AttributeError):  # not Linux: the parent's process-group kill still applies
        return
    if os.getppid() == 1:  # the parent died before the request took effect
        os._exit(1)


def main() -> None:
    die_with_parent()
    parser = argparse.ArgumentParser()
    parser.add_argument("--reply-fd", type=int, required=True)
    parser.add_argument("--strict-fp32", action="store_true")
    args = parser.parse_args()
    reply = os.fdopen(args.reply_fd, "w", buffering=1)
    if args.strict_fp32:
        strict_fp32()
    worker = Worker()
    handlers = {
        "load": lambda c: worker.load(c["path"], c["class"]),
        "build": lambda c: worker.build(c["init"], c["seed"], c["device"], c.get("compile", False)),
        "inputs": lambda c: worker.load_inputs(c["files"], c["dir"], c.get("reuse", False)),
        "run": lambda c: worker.run(c["out"], c.get("audit", False)),
        "time": lambda c: worker.time(c["calls"]),
        "dump": lambda c: worker.dump(c["out"]),
    }
    for line in sys.stdin:
        command = json.loads(line)
        if command["cmd"] == "quit":
            break
        try:
            answer = {"ok": True, **handlers[command["cmd"]](command)}
        except BaseException as e:  # SystemExit, KeyboardInterrupt: still the candidate's fault
            text = traceback.format_exc()
            answer = {
                "ok": False,
                "error": type(e).__name__,
                "message": text[-4000:],
                "compile": any(m in text for m in COMPILE_MARKERS),
            }
        reply.write(json.dumps(answer) + "\n")


if __name__ == "__main__":
    main()
