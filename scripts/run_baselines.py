"""Y3: run the human baseline kernels through the same harness as the agent (B11 sandbox).

Each candidate is baselines/tutorial_kernels.py with a task's ModelNew wrapper appended,
so the harness sees one source file, like an agent attempt. Results go to
data/baselines.json for the report's comparison row.

Run with: uv run python scripts/run_baselines.py
"""

import json
from pathlib import Path

from triton_kernel_agent.sandbox import SandboxRunner
from triton_kernel_agent.tasks import get_task

ROOT = Path(__file__).resolve().parents[1]
BASELINES = ROOT / "baselines"
OUT = ROOT / "data" / "baselines.json"
LABEL = "Triton tutorial kernels (adapted)"
WRAPPERS = {"L1-23": "l1_23.py", "L1-40": "l1_40.py"}


def candidate_source(wrapper: str) -> str:
    return (BASELINES / "tutorial_kernels.py").read_text() + (BASELINES / wrapper).read_text()


def main() -> None:
    results = {}
    with SandboxRunner.from_config() as sandbox:
        for task_id, wrapper in WRAPPERS.items():
            result = sandbox.evaluate(get_task(task_id), candidate_source(wrapper))
            results[task_id] = json.loads(result.to_json())
            t = result.timing
            speed = f"{t.speedup_eager:.2f}x eager, {t.speedup_compile:.2f}x compile" if t else ""
            print(f"{task_id}: {result.status} {speed} {result.error_excerpt or ''}".rstrip())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"label": LABEL, "results": results}, indent=2) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
