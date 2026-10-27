"""Y2 screen: check candidate KernelBench tasks for degeneracy, TF32 drift and headroom on an H100.

Kept as the record behind decisions/tasks_chosen.md; results.json and replacements.json are its
output from 2026-10-26. It screens tasks outside the vendored set, so it needs a KernelBench clone
at 423217d:
  git clone https://github.com/ScalingIntelligence/KernelBench /tmp/kb
  git -C /tmp/kb checkout 423217d
  uv run modal run scripts/y2_screen/screen.py --kb /tmp/kb --out results.json
Task sources are read locally and sent as strings; they only execute in the container.
"""

import json
import re
import sys
from pathlib import Path

import modal

CUSTOM = Path(__file__).resolve().parents[2] / "tasks" / "custom"

L1 = [1, 3, 7, 8, 9, 12, 14, 19, 23, 25, 29, 35, 36, 40, 42, 47, 51, 54, 56, 88, 89, 95, 97, 98]
L2 = [
    6,
    24,
    27,
    43,
    79,
    1,
    14,
    18,
    23,
    30,
    33,
    39,
    41,
    45,
    51,
    55,
    56,
    64,
    66,
    67,
    73,
    76,
    80,
    83,
    84,
    86,
    88,
    95,
    97,
    99,
]

app = modal.App("tka-y2-screen")
image = modal.Image.debian_slim(python_version="3.11").pip_install("torch", "numpy")


@app.function(gpu="H100!", image=image, timeout=1800, max_containers=10)
def screen(task_id: str, src: str) -> str:
    import gc
    import json
    import time
    import traceback

    import torch
    from triton.testing import do_bench

    r: dict = {"task": task_id, "torch": torch.__version__}

    def flags(matmul: bool, cudnn: bool):
        torch.backends.cuda.matmul.allow_tf32 = matmul
        torch.backends.cudnn.allow_tf32 = cudnn

    def to_cuda(xs):
        return [x.cuda() if isinstance(x, torch.Tensor) else x for x in xs]

    def run(model, xs):
        with torch.no_grad():
            out = model(*xs)
        torch.cuda.synchronize()
        return out

    def err(a, b, chunk=1 << 26):
        # Chunked so 6 GB outputs don't need 13 GB float64 copies.
        a, b = a.reshape(-1), b.reshape(-1)
        max_abs = max_rel = 0.0
        p4 = p2 = True
        for i in range(0, a.numel(), chunk):
            x, y = a[i : i + chunk].double(), b[i : i + chunk].double()
            d = (x - y).abs()
            max_abs = max(max_abs, d.max().item())
            max_rel = max(max_rel, (d / y.abs().clamp_min(1e-6)).max().item())
            p4 = p4 and bool((d <= 1e-4 + 1e-4 * y.abs()).all())
            p2 = p2 and bool((d <= 1e-2 + 1e-2 * y.abs()).all())
        return {"max_abs": max_abs, "max_rel": max_rel, "pass_1e-4": p4, "pass_1e-2": p2}

    try:
        r["default_flags"] = {
            "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        }
        ns: dict = {}
        exec(compile(src, task_id, "exec"), ns)
        Model, get_inputs, get_init_inputs = ns["Model"], ns["get_inputs"], ns["get_init_inputs"]

        torch.manual_seed(42)
        init = get_init_inputs()
        torch.manual_seed(42)
        model = Model(*init).cuda()  # train mode, like KernelBench's eval
        torch.manual_seed(1)
        xs = to_cuda(get_inputs())

        r["inputs"] = [list(x.shape) if isinstance(x, torch.Tensor) else x for x in xs]
        r["input_dtypes"] = [str(x.dtype) for x in xs if isinstance(x, torch.Tensor)]
        r["input_gb"] = (
            sum(x.numel() * x.element_size() for x in xs if isinstance(x, torch.Tensor)) / 1e9
        )
        r["param_m"] = sum(p.numel() for p in model.parameters()) / 1e6
        r["modules"] = sorted({type(m).__name__ for m in model.modules()} - {"Model"})

        # Strict fp32 reference.
        flags(False, False)
        torch.cuda.reset_peak_memory_stats()
        ref = run(model, xs)
        r["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
        r["output"] = list(ref.shape)
        rf = ref.float()
        r["out_stats"] = {
            "mean": rf.mean().item(),
            "std": rf.std().item() if ref.numel() > 1 else 0.0,
            "frac_zero": (ref == 0).sum().item() / ref.numel(),
            "nonfinite": int((~torch.isfinite(ref)).sum().item()),
        }
        del rf

        # Same inputs twice: anything random inside (dropout)?
        again = run(model, xs)
        r["nondeterministic_max_abs"] = err(again, ref)["max_abs"]
        del again

        # Fresh inputs from the same distribution: does the output depend on them?
        torch.manual_seed(2)
        xs_b = to_cuda(get_inputs())
        out_b = run(model, xs_b)
        r["input_sensitive"] = not torch.allclose(out_b, ref, atol=1e-6, rtol=1e-6)
        del xs_b, out_b

        # Identity under the default rand inputs, and under randn.
        float_ins = [x for x in xs if isinstance(x, torch.Tensor) and x.is_floating_point()]
        x0 = float_ins[0] if float_ins else None
        r["identity_under_rand"] = bool(
            x0 is not None and x0.shape == ref.shape and torch.equal(ref, x0)
        )
        torch.manual_seed(3)
        xs_n = [
            torch.randn_like(x) if isinstance(x, torch.Tensor) and x.is_floating_point() else x
            for x in xs
        ]
        out_n = run(model, xs_n)
        r["identity_under_randn"] = bool(
            x0 is not None and out_n.shape == xs_n[0].shape and torch.equal(out_n, xs_n[0])
        )
        r["randn_nonfinite"] = (
            int((~torch.isfinite(out_n)).sum().item()) if out_n.is_floating_point() else 0
        )
        del xs_n, out_n

        # Train vs eval mode (BatchNorm, Dropout).
        if any(
            m in r["modules"]
            for m in ("BatchNorm1d", "BatchNorm2d", "BatchNorm3d", "Dropout", "Dropout2d")
        ):
            model.eval()
            out_e = run(model, xs)
            model.train()
            r["train_vs_eval_max_abs"] = err(out_e, ref)["max_abs"]
            del out_e

        # TF32 drift against strict fp32; and KernelBench's default flags (cudnn TF32 on).
        if ref.is_floating_point():
            flags(True, True)
            out_t = run(model, xs)
            r["tf32_vs_fp32"] = err(out_t, ref)
            del out_t
            flags(False, True)
            out_k = run(model, xs)
            r["kb_default_vs_fp32"] = err(out_k, ref)
            del out_k

        # Timing (ms, median, L2 flushed by do_bench) under KernelBench default flags.
        flags(False, True)
        r["eager_ms"] = do_bench(lambda: run(model, xs), return_mode="median")
        flags(True, True)
        r["eager_tf32_ms"] = do_bench(lambda: run(model, xs), return_mode="median")
        flags(False, True)

        t0 = time.time()
        compiled = torch.compile(model)
        run(compiled, xs)
        r["compile_s"] = time.time() - t0
        r["compile_ms"] = do_bench(lambda: run(compiled, xs), return_mode="median")

        # Algebraic shortcuts claimed in the notes: does the cheap form reproduce the output?
        with torch.no_grad():
            x = xs[0]
            short = None
            if task_id == "L2-14":
                short = (x @ model.weight.sum(0)).unsqueeze(1) / 2 * model.scaling_factor
            elif task_id == "L2-18":
                lin = model.linear
                short = (x @ lin.weight.sum(0) + lin.bias.sum()).unsqueeze(1)
            elif task_id == "L2-51":
                g = model.gemm
                m = (
                    x @ g.weight.mean(0)
                    + (g.bias.mean() if g.bias is not None else 0)
                    - model.subtract.mean()
                )
                short = torch.nn.functional.gelu(m.unsqueeze(1)) + x
            if short is not None:
                r["shortcut_vs_fp32"] = err(short, ref)
    except Exception:
        r["error"] = traceback.format_exc()[-1500:]
    finally:
        gc.collect()
        torch.cuda.empty_cache()
    return json.dumps(r, default=str)


def collect(kb: Path) -> list[tuple[str, str]]:
    tasks = []
    for level, nums in (("1", L1), ("2", L2)):
        folder = kb / "KernelBench" / f"level{level}"
        files = {int(re.match(r"(\d+)_", p.name).group(1)): p for p in folder.glob("*.py")}
        tasks += [(f"L{level}-{n}", files[n].read_text()) for n in nums]
    for p in sorted(CUSTOM.glob("c*.py")):
        tasks.append((f"C-{p.name.split('_')[0][1:]}", p.read_text()))
    return tasks


@app.local_entrypoint()
def main(kb: str, out: str = "results.json", only: str = ""):
    tasks = collect(Path(kb))
    if only:
        keep = set(only.split(","))
        tasks = [t for t in tasks if t[0] in keep]
    print(f"screening {len(tasks)} tasks", file=sys.stderr)
    results = list(screen.starmap(tasks, order_outputs=True, return_exceptions=True))
    results = [json.loads(r) if isinstance(r, str) else {"error": repr(r)} for r in results]
    Path(out).write_text(json.dumps(results, indent=1, default=str))
    for r in results:
        print(r.get("task"), "ERROR" if "error" in r else f"eager {r.get('eager_ms', 0):.2f} ms")
