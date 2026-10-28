"""B6 acceptance: the same seed gives the same shapes; every task gets at least 5 cases."""

import pytest

from triton_kernel_agent.harness_result import ShapeClass
from triton_kernel_agent.hidden_shapes import (
    LARGE_BYTES,
    SMALL_BYTES,
    _bytes_on_meta,
    _is_pow2,
    build_case,
    generate,
    size_constants,
    with_sizes,
)
from triton_kernel_agent.reference import load_task_module
from triton_kernel_agent.tasks import get_task, load_tasks

torch = pytest.importorskip("torch")

ALL = load_tasks(include_canaries=True)


@pytest.fixture(scope="module")
def cases():
    return {t.id: generate(t.id, t.source, input_domain=t.input_domain) for t in ALL}


def test_same_seed_gives_the_same_cases(cases):
    for task in ALL[:10]:
        assert generate(task.id, task.source, input_domain=task.input_domain) == cases[task.id]
    task = get_task("L2-76")
    assert generate(task.id, task.source, seed=1) != generate(task.id, task.source, seed=2)


def test_every_task_gets_at_least_5_cases(cases):
    for task in ALL:
        classes = [c.shape_class for c in cases[task.id]]
        assert len(classes) >= 5, task.id
        assert classes.count(ShapeClass.NON_POW2) >= 2, task.id
        assert ShapeClass.SIZE_ONE in classes and ShapeClass.LARGE in classes, task.id


def test_every_case_is_valid_and_within_its_byte_budget(cases):
    for task in ALL:
        for case in cases[task.id]:
            used = _bytes_on_meta(task.source, case.sizes)
            assert used is not None, (task.id, case)
            budget = LARGE_BYTES if case.shape_class == ShapeClass.LARGE else SMALL_BYTES
            assert used <= budget, (task.id, case.shape_class, used)


def test_non_pow2_cases_change_every_size_to_a_non_power_of_2(cases):
    for task in ALL:
        visible = size_constants(task.source)
        for case in cases[task.id]:
            if case.shape_class == ShapeClass.NON_POW2:
                assert case.sizes.keys() == visible.keys(), task.id
                assert not any(_is_pow2(v) for v in case.sizes.values()), (task.id, case)


def test_size_one_and_batch_cases(cases):
    for task in ALL:
        by_class = {}
        for case in cases[task.id]:
            by_class.setdefault(case.shape_class, case)
        size_one = by_class[ShapeClass.SIZE_ONE].sizes
        assert 1 in size_one.values(), task.id
        if batch := by_class.get(ShapeClass.BATCH):
            # Both start from the same shrunk sizes; only the batch differs.
            (name,) = [k for k in batch.sizes if "batch" in k]
            assert size_one[name] == 1 and batch.sizes[name] > 1, task.id
            assert batch.sizes | {name: 1} == size_one, task.id


def test_input_distribution_cases_respect_fixed_domains(cases):
    assert get_task("L1-98").input_domain == "fixed"
    assert not any(c.distribution != "default" for c in cases["L1-98"])
    for task_id in ("L1-19", "L2-76", "L1-95"):
        assert {c.distribution for c in cases[task_id]} == {"default", "randn", "shifted"}


def test_shape_cases_do_not_reuse_the_visible_shapes(cases):
    # Input-distribution cases may keep the visible shape when it is small (L2-23).
    for task in ALL:
        visible = size_constants(task.source)
        shape_cases = [c for c in cases[task.id] if c.distribution == "default"]
        assert all(c.sizes != visible for c in shape_cases), task.id


def test_with_sizes_updates_derived_and_tuple_constants():
    ns = load_task_module(with_sizes(get_task("L2-76").source, {"out_features": 1000}))
    assert (ns["out_features"], ns["bias_shape"]) == (1000, (1000,))

    scan = get_task("L1-89").source
    assert size_constants(scan) == {"batch_size": 32768, "input_shape[0]": 32768}
    ns = load_task_module(with_sizes(scan, {"batch_size": 3, "input_shape[0]": 7}))
    assert [list(x.shape) for x in ns["get_inputs"]()] == [[3, 7]]


def test_structural_and_small_constants_are_not_sizes():
    sizes = size_constants(get_task("L1-42").source)  # max pool: kernel 4, stride 1, ...
    assert sizes.keys() == {"batch_size", "channels", "height", "width"}
    assert "reduce_dim" not in size_constants(get_task("L1-47").source)
    assert "group" not in size_constants(get_task("C-1").source)


def test_build_case_is_seeded_and_redraws_float_inputs(cases):
    task = get_task("L1-19")
    by_dist = {c.distribution: c for c in cases["L1-19"]}

    _, _, a = build_case(task.source, by_dist["default"], "cpu")
    _, _, b = build_case(task.source, by_dist["default"], "cpu")
    assert a[0].equal(b[0]) and a[0].min() >= 0  # rand: [0, 1)

    _, _, randn = build_case(task.source, by_dist["randn"], "cpu")
    assert randn[0].min() < 0  # relu is no longer the identity

    _, _, shifted = build_case(task.source, by_dist["shifted"], "cpu")
    assert 1.5 < shifted[0].mean() < 2.5 and shifted[0].std() > 3


def test_integer_inputs_keep_their_domain(cases):
    task = get_task("L1-95")  # cross entropy: float logits, integer class labels
    case = next(c for c in cases["L1-95"] if c.distribution == "shifted")
    _, init, inputs = build_case(task.source, case, "cpu")
    logits, labels = inputs
    assert logits.min() < 0
    assert labels.dtype == torch.int64 and labels.min() >= 0 and labels.max() < logits.shape[1]


def test_build_case_matches_the_case_sizes(cases):
    task = get_task("L2-76")
    case = next(c for c in cases["L2-76"] if c.shape_class == ShapeClass.NON_POW2)
    model, init, inputs = build_case(task.source, case, "cpu")
    assert list(inputs[0].shape) == [case.sizes["batch_size"], case.sizes["in_features"]]
    assert model.gemm.weight.shape == (case.sizes["out_features"], case.sizes["in_features"])
    assert model.training


def test_unknown_input_domain_raises():
    task = get_task("L1-1")
    with pytest.raises(ValueError, match="input_domain"):
        generate(task.id, task.source, input_domain="positive")
