"""Validate the correctness checker (B5) on the hidden cases (B6) of every task, on an H100.

For each hidden case, three candidates are checked against the reference:
- copy:    an independent copy of the reference. Must pass (and exercises the fp64 path).
- compile: torch.compile of the reference: legitimate fp32 with different kernels. Must pass
           on the first COMPILE_CASES cases (compiling every case would take too long).
- tf32:    the reference with TF32 turned on inside forward. Should fail on matmul/conv tasks;
           reported, not asserted, since static checks (B7) are the main TF32 defence.

Run with: uv run modal run src/triton_kernel_agent/modal_check.py [--only L1-1,L2-76]
"""

import json
import sys

import modal

from triton_kernel_agent.modal_tasks import GPU, image

COMPILE_CASES = 2

app = modal.App("tka-check")


@app.function(gpu=GPU, image=image, timeout=1800, max_containers=10)
def check_task(task_id: str, source: str, input_domain: str, atol: float, rtol: float) -> dict:
    import copy
    import gc

    import torch

    from triton_kernel_agent.correctness import Case, run_case
    from triton_kernel_agent.hidden_shapes import build_case, generate

    class TF32(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, *args):
            b = torch.backends
            saved = (b.cuda.matmul.allow_tf32, b.cudnn.allow_tf32)
            b.cuda.matmul.allow_tf32 = b.cudnn.allow_tf32 = True
            try:
                return self.model(*args)
            finally:
                b.cuda.matmul.allow_tf32, b.cudnn.allow_tf32 = saved

    def summary(check):
        return {
            "status": check.status,
            "max_err": check.max_err,
            "allowed_err": check.allowed_err,
            "ref_err": check.ref_err,
            "rule": check.rule,
            "message": check.message,
        }

    rows = []
    for i, hidden in enumerate(generate(task_id, source, input_domain=input_domain)):
        ref, _, inputs = build_case(source, hidden, "cuda")
        case = Case(inputs, hidden.shape_class)
        candidates = {"copy": copy.deepcopy(ref), "tf32": TF32(copy.deepcopy(ref))}
        if i < COMPILE_CASES:
            candidates["compile"] = torch.compile(copy.deepcopy(ref))
        row = {"shape_class": hidden.shape_class, "distribution": hidden.distribution}
        for name, cand in candidates.items():
            try:
                row[name] = summary(run_case(ref, cand, case, atol, rtol))
            except Exception as e:  # a harness bug, not a candidate failure: report it
                row[name] = {"status": "harness_error", "message": repr(e)[-500:]}
        rows.append(row)
        del ref, inputs, case, candidates
        gc.collect()
        torch.cuda.empty_cache()
    return {"task_id": task_id, "cases": rows}


@app.local_entrypoint()
def main(only: str = "", out: str = "") -> None:
    from triton_kernel_agent.tasks import load_tasks

    tasks = load_tasks(include_canaries=True)
    if only:
        tasks = [t for t in tasks if t.id in set(only.split(","))]
    args = [(t.id, t.source, t.input_domain, t.atol, t.rtol) for t in tasks]
    results = list(check_task.starmap(args, order_outputs=True))
    if out:
        with open(out, "w") as f:
            json.dump(results, f, indent=1, default=str)
    for r in results:
        statuses = {
            name: "".join(
                "." if c[name]["status"] is None else "x" for c in r["cases"] if name in c
            )
            for name in ("copy", "compile", "tf32")
        }
        print(f"{r['task_id']:>6}  copy {statuses['copy']:8} compile {statuses['compile']:3} "
              f"tf32 {statuses['tf32']}")  # fmt: skip
    print("'.' = passed, 'x' = failed", file=sys.stderr)
