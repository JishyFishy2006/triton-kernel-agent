"""B7 acceptance: fixtures that call torch.matmul, F.softmax, or import the reference are
flagged; a real Triton kernel passes. One fixture per item of the L4 cheat list too."""

from pathlib import Path

import pytest

from triton_kernel_agent.harness_result import CheatFlag
from triton_kernel_agent.static_check import scan

FIXTURES = Path(__file__).parent / "fixtures" / "candidates"


def header(path: Path, key: str) -> str:
    for line in path.read_text().splitlines():
        if line.startswith(f"# {key}:"):
            return line.split(":", 1)[1].strip()
    raise ValueError(f"{path.name} has no '# {key}:' line")


def expected(path: Path) -> set[CheatFlag]:
    value = header(path, "expect")
    return set() if value == "none" else {CheatFlag(v.strip()) for v in value.split(",")}


ALL = sorted(FIXTURES.glob("*.py"))
OK = [p for p in ALL if p.name.startswith("ok_")]
CHEATS = [p for p in ALL if expected(p)]
# Real kernels, plus runtime cheats (rt_*) that are written to get past the static scan.
STATICALLY_CLEAN = [p for p in ALL if not expected(p)]


@pytest.mark.parametrize("path", STATICALLY_CLEAN, ids=lambda p: p.stem)
def test_real_triton_kernels_pass(path):
    report = scan(path.read_text())
    assert report.clean, report.summary()


@pytest.mark.parametrize("path", CHEATS, ids=lambda p: p.stem)
def test_cheats_are_flagged(path):
    report = scan(path.read_text())
    assert expected(path) <= set(report.flags), report.summary()
    assert all(f.flag.is_static for f in report.findings)


def test_roadmap_acceptance_fixtures_exist():
    names = {p.stem for p in CHEATS}
    assert {"cheat_torch_matmul", "cheat_f_softmax", "cheat_reference_import"} <= names
    assert len(OK) >= 3


def test_l4_cheat_list_is_covered():
    flags = set().union(*(expected(p) for p in CHEATS))
    l4 = {
        CheatFlag.INTROSPECTION,  # 1: find the precomputed reference tensor
        CheatFlag.SIDE_STREAM,  # 2: unsynchronized side stream
        CheatFlag.MONKEYPATCH,  # 3: patch timers and checks
        CheatFlag.HARDCODED_DATA,  # 4: hardcoded answers
    }
    assert l4 <= flags  # 5 (identity / uninitialized output) is caught at runtime (B5, B8)


def flags_of(source: str) -> set[CheatFlag]:
    return set(scan(source).flags)


@pytest.mark.parametrize(
    "source",
    [
        "import torch\ny = torch.empty_like(x).view(-1).contiguous()",
        "import torch\nout = torch.empty((x.shape[0], 8), device=x.device, dtype=x.dtype)",
        "import triton\ngrid = (triton.cdiv(n, 1024), triton.next_power_of_2(m))",
        "import math\nscale = 1 / math.sqrt(d)",
        "block = max(16, min(n, 128))",  # builtins, not torch
        "from triton.runtime import driver",
        "import torch.nn as nn\nself.gemm = nn.Linear(4, 8)\nw = self.gemm.weight.t()",
        "configs = [(64, 64, 32, 4), (128, 64, 32, 8), (128, 128, 64, 8)]",  # small table
        "import torch\nx = torch.cat([a, b], dim=1).reshape(2, -1)",
    ],
)
def test_ordinary_wrapper_code_is_clean(source):
    assert flags_of(source) == set(), scan(source).summary()


@pytest.mark.parametrize(
    ("source", "flag"),
    [
        ("y = x.sum(dim=1)", CheatFlag.TORCH_COMPUTE_CALL),
        ("y = x.exp_()", CheatFlag.TORCH_COMPUTE_CALL),
        ("import torch\nf = torch.compile(model)", CheatFlag.TORCH_COMPUTE_CALL),
        ("import torch\ny = torch._C._nn.linear(x, w)", CheatFlag.TORCH_COMPUTE_CALL),
        ("import torch\nf = getattr(torch, 'matmul')", CheatFlag.TORCH_COMPUTE_CALL),
        (
            "import torch.nn.functional\ny = torch.nn.functional.gelu(x)",
            CheatFlag.TORCH_COMPUTE_CALL,
        ),
        ("import torch\nt = (torch.matmul(a, b),)", CheatFlag.TORCH_COMPUTE_CALL),
        ("import torch\ntorch.backends.cudnn.allow_tf32 = True", CheatFlag.BACKEND_FLAGS),
        (
            "import torch\ntorch.backends.cuda.matmul.fp32_precision = 'tf32'",
            CheatFlag.BACKEND_FLAGS,
        ),
        ("import torch\ng = torch.cuda.CUDAGraph()", CheatFlag.SIDE_STREAM),
        ("k[grid](x, stream=s)", CheatFlag.SIDE_STREAM),
        ("import triton.testing", CheatFlag.MONKEYPATCH),
        ("import torch\ndel torch.cuda.synchronize", CheatFlag.MONKEYPATCH),
        ("__builtins__['print'] = None", CheatFlag.MONKEYPATCH),
        ("v = vars(obj)", CheatFlag.INTROSPECTION),
        ("from . import harness", CheatFlag.REFERENCE_IMPORT),
        ("import numpy as np\na = np.load('x.npy')", CheatFlag.FILE_OR_NETWORK),
        ("blob = '" + "A" * 5000 + "'", CheatFlag.HARDCODED_DATA),
        ("t = [[0.5] * 1] * 1 + [" + ", ".join(["1.0"] * 80) + "]", CheatFlag.HARDCODED_DATA),
    ],
)
def test_single_rules(source, flag):
    assert flag in flags_of(source), scan(source).summary()


def test_tl_dot_inside_kernels_needs_ieee_but_not_outside():
    kernel = """
import triton
import triton.language as tl

@triton.jit
def k(a, b):
    good = tl.dot(a, b, input_precision="ieee")
    bad = tl.dot(a, b, input_precision="tf32")
    worse = tl.dot(a, b)
"""
    findings = [f for f in scan(kernel).findings if f.flag == CheatFlag.TF32_DOT]
    assert [f.line for f in findings] == [8, 9]


def test_kernel_code_may_use_tl_math():
    kernel = """
import triton
import triton.language as tl

@triton.autotune(configs=[], key=["n"])
@triton.jit
def k(x_ptr, n, BLOCK: tl.constexpr):
    x = tl.load(x_ptr + tl.arange(0, BLOCK))
    y = tl.exp(x - tl.max(x, axis=0))
    tl.store(x_ptr, tl.sum(y, axis=0) / tl.sqrt(y))
"""
    assert scan(kernel).clean, scan(kernel).summary()


def test_findings_carry_lines_and_syntax_errors_raise():
    report = scan("import torch\n\n\ny = torch.matmul(a, b)\n")
    assert [(f.flag, f.line) for f in report.findings] == [(CheatFlag.TORCH_COMPUTE_CALL, 4)]
    assert "line 4" in report.summary()
    with pytest.raises(SyntaxError):
        scan("def broken(:\n")


def test_autotune_config_lists_are_not_hardcoded_data():
    # Regression: a real muse-wonder L2-76 reply had 8 triton.Config entries (72 numbers).
    configs = ",\n".join(
        f"    triton.Config({{'BLOCK_M': {m}, 'BLOCK_N': {n}, 'BLOCK_K': 32, 'GROUP_M': 8}}, "
        f"num_warps={w}, num_stages={st})"
        for m, n, w, st in [(128, 256, 8, 3), (256, 128, 8, 3), (128, 128, 4, 4), (64, 256, 4, 4),
                            (128, 64, 4, 4), (64, 128, 4, 4), (64, 64, 4, 5), (32, 64, 2, 5)]
    )  # fmt: skip
    source = f"import triton\nCONFIGS = [\n{configs}\n]\n"
    assert scan(source).clean, scan(source).summary()
    # The same numbers as a bare table are still data.
    table = "T = [" + ", ".join(f"({i}, {i + 1}, -{i + 2})" for i in range(30)) + "]"
    assert CheatFlag.HARDCODED_DATA in flags_of(table)


@pytest.mark.parametrize(
    "code, flag",
    [
        ("f = exec\nf('x = 1')\n", CheatFlag.MONKEYPATCH),
        ("g = getattr\nm = g(torch, 'mat' + 'mul')\n", CheatFlag.MONKEYPATCH),
        ("calls = [vars]\n", CheatFlag.MONKEYPATCH),
        ("import triton.language as tl\nd = tl.core.dot\n", CheatFlag.INTROSPECTION),
        ("import triton.language as tl\nd = tl.core.dot\n", CheatFlag.TF32_DOT),
    ],
)
def test_review_gaps_are_flagged(code, flag):
    # Review of the Y4 scan: banned builtins used as bare names, and tl.core.
    assert flag in scan("import torch\n" + code).flags


def test_triton_device_properties_are_fine_torch_utils_is_not():
    tutorial = (
        "import triton\nfrom triton.runtime import driver\n"
        "props = driver.active.utils.get_device_properties(0)\n"
    )
    assert scan(tutorial).clean, scan(tutorial).summary()
    assert CheatFlag.INTROSPECTION in scan("import torch\nm = torch.utils.data\n").flags
