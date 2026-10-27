import ast
import json

import pytest

from triton_kernel_agent.tasks import TASKS_DIR, get_task, load_tasks

# Decided in Y2 (decisions/tasks_chosen.md).
DEV = {
    "L1-1", "L1-9", "L1-19", "L1-25", "L1-36", "L1-42", "L1-47",
    "L2-1", "L2-30", "L2-56", "L2-76", "L2-86",
}  # fmt: skip


def test_loads_the_decided_43_tasks_and_split():
    tasks = load_tasks()
    assert len(tasks) == 43
    assert {t.id for t in tasks if t.split == "dev"} == DEV
    assert len(load_tasks(split="held_out")) == 31
    assert {t.id for t in load_tasks(split="dev")} == DEV


def test_l2_27_replaces_l2_23_and_canaries_are_unscored():
    scored = {t.id for t in load_tasks()}
    assert "L2-27" in scored
    assert "L2-23" not in scored

    canaries = [t for t in load_tasks(include_canaries=True) if t.role == "canary"]
    assert {t.id for t in canaries} == {"L2-23", "L2-80"}
    assert all(t.split is None and not t.in_fast_p for t in canaries)


def test_only_l1_12_is_left_out_of_fast_p():
    assert [t.id for t in load_tasks() if not t.in_fast_p] == ["L1-12"]


def test_every_source_has_the_kernelbench_interface():
    for task in load_tasks(include_canaries=True):
        names = {
            node.name
            for node in ast.parse(task.source).body
            if isinstance(node, ast.FunctionDef | ast.ClassDef)
        }
        assert {"Model", "get_inputs", "get_init_inputs"} <= names, task.id


def test_levels_names_and_fp32_tolerances():
    task = get_task("L2-76")
    assert (task.level, task.name, task.group) == (2, "Gemm_Add_ReLU", "L2 GEMM chain")
    assert get_task("C-4").level == 2
    assert get_task("L1-19").name == "ReLU"
    assert all((t.dtype, t.atol, t.rtol) == ("float32", 1e-4, 1e-4) for t in load_tasks())


def test_unknown_task_raises():
    with pytest.raises(KeyError):
        get_task("L1-2")


def test_bad_split_argument_raises():
    with pytest.raises(ValueError, match="split"):
        load_tasks(split="train")


def _manifest(tmp_path, body: str):
    (tmp_path / "kernelbench").symlink_to(TASKS_DIR / "kernelbench")
    (tmp_path / "custom").symlink_to(TASKS_DIR / "custom")
    path = tmp_path / "manifest.toml"
    path.write_text('[defaults]\ndtype = "float32"\n' + body)
    return path


@pytest.mark.parametrize(
    ("body", "match"),
    [
        ('[[task]]\nid = "L1-1"\ngroup = "g"\nsplit = "dev"\n' * 2, "duplicate"),
        ('[[task]]\nid = "L1-1"\ngroup = "g"\nsplit = "train"\n', "split"),
        ('[[task]]\nid = "L1-1"\ngroup = "g"\n', "split"),
        ('[[task]]\nid = "L2-23"\ngroup = "g"\nrole = "canary"\nsplit = "dev"\n', "canar"),
        ('[[task]]\nid = "L1-2"\ngroup = "g"\nsplit = "dev"\n', "expected one file"),
        ('[[task]]\nid = "L3-1"\ngroup = "g"\nsplit = "dev"\n', "bad task id"),
        ('[[task]]\nid = "L1-1"\ngroup = "g"\nsplit = "dev"\ndtype = "int8"\n', "dtype"),
    ],
)
def test_bad_manifest_raises(tmp_path, body, match):
    with pytest.raises(ValueError, match=match):
        load_tasks(manifest_path=_manifest(tmp_path, body), metadata_path=tmp_path / "none.json")


def test_metadata_is_merged_by_id(tmp_path):
    manifest = _manifest(tmp_path, '[[task]]\nid = "L1-1"\ngroup = "g"\nsplit = "dev"\n')
    metadata = tmp_path / "metadata.json"
    metadata.write_text(json.dumps({"tasks": {"L1-1": {"bytes": 12, "flops": 34}}}))
    (task,) = load_tasks(manifest_path=manifest, metadata_path=metadata)
    assert (task.bytes, task.flops) == (12, 34)

    (task,) = load_tasks(manifest_path=manifest, metadata_path=tmp_path / "missing.json")
    assert (task.bytes, task.flops) == (None, None)
