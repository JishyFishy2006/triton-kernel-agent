"""B5 acceptance tests on CPU: a known-correct kernel passes; one perturbed by 1e-2 fails;
NaN output fails; wrong shape fails. Plus the Y2 rules (fp64-relative tolerance, strict fp32).
Candidates here are our own stand-ins, run in-process with check_trusted(); agent code runs in
a worker process (tests/test_isolation.py)."""

import pytest

from triton_kernel_agent.correctness import (
    FLOOR_ULPS,
    REF_ERR_MULTIPLE,
    check_outputs,
    check_trusted,
    compare,
    strict_fp32,
    tf32_enabled,
)
from triton_kernel_agent.harness_result import Status

torch = pytest.importorskip("torch")
nn = torch.nn

ATOL = RTOL = 1e-4


class GemmGelu(nn.Module):
    """Reference: Linear then GELU, like the Level 2 GEMM chains."""

    def __init__(self, in_features=512, out_features=96):
        super().__init__()
        torch.manual_seed(0)
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return nn.functional.gelu(self.linear(x))


class Candidate(nn.Module):
    """Same weights as the reference; `fn` decides what forward returns."""

    def __init__(self, ref, fn):
        super().__init__()
        self.ref, self.fn = ref, fn

    def forward(self, x):
        return self.fn(self.ref, x)


def chunked_gemm(ref, x, chunk=64):
    # Known-correct alternative: same math in fp32, different summation order.
    w, b = ref.linear.weight, ref.linear.bias
    acc = torch.zeros(x.shape[0], w.shape[0], dtype=x.dtype)
    for k in range(0, x.shape[1], chunk):
        acc += x[:, k : k + chunk] @ w[:, k : k + chunk].T
    return nn.functional.gelu(acc + b)


def round_to_tf32(t):
    # TF32 keeps 10 mantissa bits: zero the low 13 of fp32's 23.
    return (t.view(torch.int32) & ~0x1FFF).view(torch.float32)


def tf32_gemm(ref, x):
    w, b = ref.linear.weight, ref.linear.bias
    return nn.functional.gelu(round_to_tf32(x) @ round_to_tf32(w).T + b)


def run(fn, x=None, ref=None, hidden=True):
    ref = ref or GemmGelu()
    torch.manual_seed(1)
    x = torch.randn(64, 512) if x is None else x
    return check_trusted(ref, Candidate(ref, fn), [x], ATOL, RTOL, hidden=hidden)


# --- the four roadmap acceptance checks ---


def test_known_correct_kernel_passes():
    result = run(chunked_gemm)
    assert result.passed, result.message
    assert result.rule == "fp64_relative"
    assert result.max_err <= result.allowed_err


def test_perturbed_by_1e_2_fails():
    result = run(lambda ref, x: ref(x) + 1e-2)
    assert result.status == Status.MISMATCH
    assert result.max_err == pytest.approx(1e-2, rel=1e-3)


def test_nan_output_fails():
    def one_nan(ref, x):
        out = ref(x)
        out[3, 7] = float("nan")
        return out

    assert run(one_nan).status == Status.NAN_INF
    assert run(lambda ref, x: ref(x) * float("inf")).status == Status.NAN_INF


def test_wrong_shape_or_dtype_fails():
    assert run(lambda ref, x: ref(x)[:, :-1]).status == Status.WRONG_SHAPE
    assert run(lambda ref, x: ref(x).unsqueeze(0)).status == Status.WRONG_SHAPE
    assert run(lambda ref, x: ref(x).double()).status == Status.WRONG_SHAPE


# --- Y2 decisions ---


def test_tf32_rounding_fails_the_fp64_relative_rule():
    result = run(tf32_gemm)
    assert result.status == Status.MISMATCH
    assert result.max_err > 10 * REF_ERR_MULTIPLE * result.ref_err


class GemmDivideSumScale(nn.Module):
    """L2-14 in miniature: (x @ W.T) / 2, summed over features, scaled."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.weight = nn.Parameter(torch.randn(1024, 2048))

    def forward(self, x):
        return torch.sum(x @ self.weight.T / 2, dim=1, keepdim=True) * 1.5


def test_exact_algebraic_shortcut_passes():
    # Collapsing the GEMM to a matrix-vector product is exact algebra: it must not be
    # rejected for summing in a different order (Y2 decision 3).
    def shortcut(ref, x):
        return (x @ ref.weight.sum(0)).unsqueeze(1) / 2 * 1.5

    torch.manual_seed(1)
    result = run(shortcut, x=torch.rand(256, 2048), ref=GemmDivideSumScale())
    assert result.passed, result.message


def test_exact_ops_are_held_to_a_floor_of_a_few_ulps():
    relu = nn.ReLU()
    result = run(lambda ref, x: torch.clamp(x, min=0), ref=relu)
    assert result.passed and result.ref_err == 0.0
    torch.manual_seed(1)
    biggest = torch.randn(64, 512).clamp(min=0).max().item()  # same input as run()
    eps = torch.finfo(torch.float32).eps
    assert result.allowed_err == pytest.approx(FLOOR_ULPS * eps * biggest)
    assert run(lambda ref, x: torch.clamp(x, min=0) + 1e-5, ref=relu).status == Status.MISMATCH


def test_floor_scales_with_the_output():
    # The same small absolute error is fine on huge outputs, wrong on tiny ones.
    torch.manual_seed(3)
    ref = torch.randn(1000, dtype=torch.float64)
    for scale, ok in ((1e4, True), (1.0, False)):
        truth = ref * scale
        check = compare((truth + 5e-5).float(), truth.float(), truth, ATOL, RTOL)
        assert check.passed is ok, scale


def test_tiny_outputs_are_not_hidden_by_an_absolute_atol():
    # L2-99-like: softmax outputs around 1e-4. A 1e-4 atol would accept anything; TF32 must fail.
    torch.manual_seed(4)
    logits = torch.randn(32, 4096, dtype=torch.float64)
    truth = torch.softmax(logits, dim=1)
    ref32 = torch.softmax(logits.float(), dim=1)
    tf32ish = torch.softmax(round_to_tf32(logits.float()), dim=1)
    assert compare(ref32.clone(), ref32, truth, ATOL, RTOL).passed
    assert compare(tf32ish, ref32, truth, ATOL, RTOL).status == Status.MISMATCH


def test_nonfinite_where_the_reference_is_nonfinite_is_fine():
    log = Candidate(None, lambda ref, x: torch.log(x))  # log of negatives is NaN
    assert run(lambda ref, x: torch.log(x), ref=log).passed
    assert run(lambda ref, x: torch.log(x.abs()), ref=log).status == Status.MISMATCH


def test_integer_outputs_must_match_exactly():
    argmax = Candidate(None, lambda ref, x: torch.argmax(x, dim=1))
    assert run(lambda ref, x: torch.argmax(x, dim=1), ref=argmax).passed
    off_by_one = run(lambda ref, x: torch.argmax(x, dim=1) + 1, ref=argmax)
    assert off_by_one.status == Status.MISMATCH and off_by_one.rule == "exact"


def test_without_fp64_the_fixed_tolerance_applies():
    torch.manual_seed(2)
    ref32 = torch.randn(100, 100)
    assert compare(ref32 + 5e-5, ref32, None, ATOL, RTOL).passed
    failed = compare(ref32 + 1e-3, ref32, None, ATOL, RTOL)
    assert (failed.status, failed.rule) == (Status.MISMATCH, "fixed")


def test_hidden_failure_messages_do_not_leak_shapes():
    message = run(lambda ref, x: ref(x)[:, :-1]).message
    assert "96" not in message and "95" not in message
    assert "[64, 95]" in run(lambda ref, x: ref(x)[:, :-1], hidden=False).message


def test_candidate_cannot_corrupt_the_reference_inputs():
    def clobber(ref, x):
        out = ref(x)
        x.zero_()
        return out

    assert run(clobber).passed


# --- check_outputs(): every output tensor, worst error or first failure ---


def test_every_output_tensor_is_checked():
    torch.manual_seed(1)
    a, b = torch.randn(8, 8), torch.randn(4)
    assert check_outputs([a, b], [a, b], [a.double(), b.double()], ATOL, RTOL, True).passed
    bad = check_outputs([a, b + 1], [a, b], [a.double(), b.double()], ATOL, RTOL, True)
    assert bad.status == Status.MISMATCH
    count = check_outputs([a], [a, b], None, ATOL, RTOL, True)
    assert count.status == Status.WRONG_SHAPE and "returned 1 tensors" in count.message


# --- strict fp32 ---


def test_strict_fp32_turns_tf32_off_and_restores_it():
    before = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    torch.backends.cudnn.allow_tf32 = True
    try:
        with strict_fp32():
            assert not tf32_enabled()
            assert torch.backends.cuda.matmul.fp32_precision == "ieee"
        assert torch.backends.cudnn.allow_tf32
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = before


def test_reference_runs_in_strict_fp32_even_if_the_candidate_turns_tf32_on():
    seen = []

    class Spy(nn.Module):
        def forward(self, x):
            seen.append(torch.backends.cuda.matmul.allow_tf32)
            return x * 2

    def sneaky(ref, x):
        torch.backends.cuda.matmul.allow_tf32 = True
        return x * 2

    before = torch.backends.cuda.matmul.allow_tf32
    try:
        check_trusted(Spy(), Candidate(None, sneaky), [torch.randn(4, 4)], ATOL, RTOL)
        assert seen and not any(seen)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = before
