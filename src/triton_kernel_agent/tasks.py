"""Load the Y2 task list: task sources from tasks/ plus their per-task metadata.

tasks/manifest.toml records which tasks are in (see decisions/tasks_chosen.md). Bytes and FLOPs
come from tasks/metadata.json, measured on an H100 by modal_tasks.py. Pure Python, no torch.
"""

import json
import re
import tomllib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from triton_kernel_agent.config import REPO_ROOT

TASKS_DIR = REPO_ROOT / "tasks"
MANIFEST_PATH = TASKS_DIR / "manifest.toml"
METADATA_PATH = TASKS_DIR / "metadata.json"

SPLITS = ("dev", "held_out")
ROLES = ("scored", "canary")
INPUT_DOMAINS = ("any", "fixed")
# (atol, rtol) per dtype, from the tech spec. B5 decides how these combine with the
# fp64-relative rule accepted in Y2; here they are just the per-task numbers.
TOLERANCES = {
    "float32": (1e-4, 1e-4),
    "float16": (1e-2, 1e-2),
    "bfloat16": (1e-2, 1e-2),
}

_ID_RE = re.compile(r"^(?:L([12])-(\d+)|C-(\d+))$")


@dataclass(frozen=True)
class Task:
    id: str
    level: int  # own tasks (C-n) are fused chains, so they count as Level 2
    name: str  # file name without the number prefix, e.g. "Gemm_Add_ReLU"
    group: str
    split: str | None  # None for canaries
    role: str
    in_fast_p: bool
    path: Path
    dtype: str
    atol: float
    rtol: float
    # Per forward call, measured on GPU. None until tasks/metadata.json has the task.
    bytes: int | None = None  # inputs + parameters + buffers + outputs, each touched once
    flops: int | None = None  # matmul/conv/attention FLOPs (FlopCounterMode); 0 if elementwise
    # "fixed": hidden cases keep get_inputs()'s distribution (L1-98 needs probabilities).
    input_domain: str = "any"
    note: str = ""

    @property
    def source(self) -> str:
        return self.path.read_text()


def _task_file(task_id: str, tasks_dir: Path) -> tuple[int, Path]:
    match = _ID_RE.match(task_id)
    if not match:
        raise ValueError(f"bad task id {task_id!r}: expected L1-<n>, L2-<n> or C-<n>")
    level, number, own = match.groups()
    if own:
        level, pattern, folder = "2", f"c{own}_*.py", tasks_dir / "custom"
    else:
        pattern, folder = f"{number}_*.py", tasks_dir / "kernelbench" / f"level{level}"
    files = sorted(folder.glob(pattern))
    if len(files) != 1:
        raise ValueError(f"{task_id}: expected one file matching {folder / pattern}, got {files}")
    return int(level), files[0]


def _task(raw: dict[str, Any], defaults: dict[str, Any], tasks_dir: Path) -> Task:
    task_id = raw["id"]
    level, path = _task_file(task_id, tasks_dir)

    role = raw.get("role", "scored")
    if role not in ROLES:
        raise ValueError(f"{task_id}: role must be one of {ROLES}, got {role!r}")
    split = raw.get("split")
    if role == "canary" and split is not None:
        raise ValueError(f"{task_id}: canaries are unscored and take no split")
    if role == "scored" and split not in SPLITS:
        raise ValueError(f"{task_id}: split must be one of {SPLITS}, got {split!r}")

    dtype = raw.get("dtype", defaults["dtype"])
    if dtype not in TOLERANCES:
        raise ValueError(f"{task_id}: dtype must be one of {tuple(TOLERANCES)}, got {dtype!r}")
    atol, rtol = TOLERANCES[dtype]

    input_domain = raw.get("input_domain", "any")
    if input_domain not in INPUT_DOMAINS:
        raise ValueError(f"{task_id}: input_domain must be one of {INPUT_DOMAINS}")

    return Task(
        id=task_id,
        level=level,
        name=path.stem.split("_", 1)[1],
        group=raw["group"],
        split=split,
        role=role,
        in_fast_p=raw.get("in_fast_p", role == "scored"),
        path=path,
        dtype=dtype,
        atol=atol,
        rtol=rtol,
        input_domain=input_domain,
        note=raw.get("note", ""),
    )


def load_tasks(
    split: str | None = None,
    include_canaries: bool = False,
    manifest_path: Path = MANIFEST_PATH,
    metadata_path: Path = METADATA_PATH,
) -> list[Task]:
    """Tasks in manifest order. split=None means both splits."""
    if split is not None and split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")

    manifest_path = Path(manifest_path)
    with manifest_path.open("rb") as f:
        manifest = tomllib.load(f)
    tasks = [_task(raw, manifest["defaults"], manifest_path.parent) for raw in manifest["task"]]

    ids = [t.id for t in tasks]
    if dupes := sorted({i for i in ids if ids.count(i) > 1}):
        raise ValueError(f"duplicate task ids in {manifest_path}: {dupes}")

    measured = {}
    if Path(metadata_path).exists():
        measured = json.loads(Path(metadata_path).read_text())["tasks"]
    tasks = [
        replace(t, bytes=m["bytes"], flops=m["flops"]) if (m := measured.get(t.id)) else t
        for t in tasks
    ]

    return [
        t
        for t in tasks
        if (include_canaries or t.role == "scored") and (split is None or t.split == split)
    ]


def get_task(task_id: str, **kwargs: Any) -> Task:
    for task in load_tasks(include_canaries=True, **kwargs):
        if task.id == task_id:
            return task
    raise KeyError(task_id)
