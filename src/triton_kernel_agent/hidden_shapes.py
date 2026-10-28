"""Hidden-shape generator (B6): deterministic test cases the agent never sees.

KernelBench tasks size themselves with module-level integer constants (batch_size, in_features,
dim1, ...) that get_inputs() and get_init_inputs() read. A hidden case overrides some of them,
re-runs the module so derived constants (bias_shape = (out_features,)) follow, and optionally
redraws the float inputs from another distribution. Every case is checked by running the
reference on PyTorch's meta device (shapes only, no memory), so it is always a valid input.

Per task (Y2 policies): two non-power-of-2 cases, a size-1 case, a varied-batch case, one large
case of at most LARGE_BYTES, and two input-distribution cases (randn, shifted) unless the task's
inputs have a fixed domain (L1-98 needs probabilities). Same seed, same cases.
"""

import ast
import random
import zlib
from dataclasses import dataclass, field
from typing import Any

from triton_kernel_agent.harness_result import ShapeClass
from triton_kernel_agent.reference import INIT_SEED, load_task_module
from triton_kernel_agent.tasks import INPUT_DOMAINS

# Fixed seed for the hidden cases of a harness version. Never shown to the agent.
HIDDEN_SEED = 20261104
# Inputs + parameters + outputs per case. Small cases keep the check fast; the large one stays under
# ~1 GB (Y2: default sizes reach 8.6 GB of inputs, which is only worth paying for timing).
SMALL_BYTES = 64 * 2**20
LARGE_BYTES = 2**30
# Integer constants named like these are structure, not size: kernel 3 stays kernel 3.
STRUCTURAL = frozenset({"kernel_size", "stride", "padding", "dilation", "num_groups", "group"})
# Smaller integers are left alone too: axes (dim = 1), channels of 3, and so on.
MIN_SIZE = 16
# Non-power-of-2 sizes are odd multiples of one of these; later ones satisfy divisibility
# (GroupNorm groups, group sums) when the earlier ones make the reference fail.
MULTIPLES = (1, 8, 32, 128, 256)
DISTRIBUTIONS = ("default", "randn", "shifted")


@dataclass(frozen=True)
class HiddenCase:
    shape_class: ShapeClass
    sizes: dict[str, int] = field(default_factory=dict)  # "name" or "name[i]" -> new value
    distribution: str = "default"  # how float inputs are redrawn (DISTRIBUTIONS)
    seed: int = 0  # input seed; weights always use INIT_SEED


# --- reading and rewriting the size constants ---


def _targets(node: ast.Assign) -> list[str]:
    names = []
    for target in node.targets:
        elts = target.elts if isinstance(target, ast.Tuple) else [target]
        names += [e.id for e in elts if isinstance(e, ast.Name)]
    return names


def _is_literal(node: ast.AST) -> bool:
    return not any(isinstance(n, ast.Name) for n in ast.walk(node))


def size_constants(source: str) -> dict[str, int]:
    """Module-level sizes a hidden case may change, with their visible values.

    Integers >= MIN_SIZE (not STRUCTURAL) are sizes. Literal tuples of integers, such as
    input_shape = (32768,), contribute their elements as "input_shape[0]"; tuples built from
    other names (bias_shape = (out_features,)) follow those names instead.
    """
    namespace = load_task_module(source)
    sizes: dict[str, int] = {}
    for node in ast.parse(source).body:
        if not isinstance(node, ast.Assign):
            continue
        tuple_value = isinstance(node.value, ast.Tuple) and len(node.targets) == 1
        for name in _targets(node):
            value = namespace.get(name)
            if name in STRUCTURAL or isinstance(value, bool):
                continue
            if isinstance(value, int) and value >= MIN_SIZE:
                sizes[name] = value
            elif (
                isinstance(value, tuple)
                and tuple_value
                and isinstance(node.targets[0], ast.Name)
                and _is_literal(node.value)
            ):
                for i, v in enumerate(value):
                    if isinstance(v, int) and v >= MIN_SIZE:
                        sizes[f"{name}[{i}]"] = v
    return sizes


def with_sizes(source: str, sizes: dict[str, int]) -> str:
    """The source with each size constant reassigned right after its original assignment."""
    if not sizes:
        return source
    namespace = load_task_module(source)
    tuples: dict[str, dict[int, int]] = {}
    ints: dict[str, int] = {}
    for key, value in sizes.items():
        if key.endswith("]"):
            name, index = key[:-1].split("[")
            tuples.setdefault(name, {})[int(index)] = value
        else:
            ints[key] = value

    tree = ast.parse(source)
    body: list[ast.stmt] = []
    for node in tree.body:
        body.append(node)
        if not isinstance(node, ast.Assign):
            continue
        for name in _targets(node):
            if name in ints:
                body += ast.parse(f"{name} = {ints[name]!r}").body
            elif name in tuples:
                value = list(namespace[name])
                for i, v in tuples[name].items():
                    value[i] = v
                body += ast.parse(f"{name} = {tuple(value)!r}").body
    tree.body = body
    return ast.unparse(tree)


# --- checking a candidate case on the meta device ---


def _bytes_on_meta(source: str, sizes: dict[str, int]) -> int | None:
    """Inputs + parameters + outputs for these sizes, or None if the reference rejects them."""
    import torch
    from torch.utils._pytree import tree_leaves

    try:
        ns = load_task_module(with_sizes(source, sizes))
        with torch.device("meta"), torch.no_grad():
            model = ns["Model"](*ns["get_init_inputs"]())
            inputs = ns["get_inputs"]()
            outputs = tree_leaves(model(*inputs))
    except Exception:
        return None
    tensors = [
        t
        for t in [*inputs, *model.parameters(), *model.buffers(), *outputs]
        if isinstance(t, torch.Tensor)
    ]
    return sum(t.numel() * t.element_size() for t in tensors)


def _fit(source: str, sizes: dict[str, int], budget: int) -> dict[str, int]:
    """Halve the largest size the reference still accepts halved, until the case fits."""
    sizes = dict(sizes)
    used = _bytes_on_meta(source, sizes)
    while used is not None and used > budget:
        for name in sorted(sizes, key=sizes.get, reverse=True):
            if sizes[name] < 2 * MIN_SIZE:
                continue
            halved = sizes | {name: sizes[name] // 2}
            if (halved_used := _bytes_on_meta(source, halved)) is not None:
                sizes, used = halved, halved_used
                break
        else:
            break  # nothing left to shrink; the case stays over budget
    return sizes


def _is_pow2(n: int) -> bool:
    return n > 0 and n & (n - 1) == 0


def _odd_multiple(target: float, multiple: int) -> int:
    k = max(1, round(target / multiple))
    if k % 2 == 0:
        k += 1
    n = k * multiple
    return n if not _is_pow2(n) else n + 2 * multiple


def _non_pow2(source, fitted, budget, rng, names=None) -> dict[str, int] | None:
    """Every size in `names` (default: all) moved to a nearby non-power-of-2 value."""
    names = list(fitted) if names is None else names
    shrink = {name: rng.uniform(0.55, 0.95) for name in names}
    for multiple in MULTIPLES:
        sizes = dict(fitted)
        for name in names:
            sizes[name] = _odd_multiple(fitted[name] * shrink[name], multiple)
        used = _bytes_on_meta(source, sizes)
        if used is not None and used <= budget:
            return sizes
    return None


def _batch_name(sizes: dict[str, int]) -> str | None:
    return next((name for name in sizes if "batch" in name), None)


# --- the generator ---


def generate(
    task_id: str, source: str, seed: int = HIDDEN_SEED, input_domain: str = "any"
) -> list[HiddenCase]:
    """The task's hidden cases, in a fixed order. Same (task, source, seed) -> same cases."""
    if input_domain not in INPUT_DOMAINS:
        raise ValueError(f"input_domain must be one of {INPUT_DOMAINS}, got {input_domain!r}")
    rng = random.Random(zlib.crc32(f"{seed}:{task_id}".encode()))
    visible = size_constants(source)
    small = _fit(source, visible, SMALL_BYTES)
    large = _fit(source, visible, LARGE_BYTES)
    cases: list[HiddenCase] = []

    def add(shape_class, sizes, distribution="default"):
        if sizes is not None:
            cases.append(HiddenCase(shape_class, sizes, distribution, rng.randrange(2**31)))

    add(ShapeClass.NON_POW2, _non_pow2(source, small, SMALL_BYTES, rng))
    add(ShapeClass.NON_POW2, _non_pow2(source, small, SMALL_BYTES, rng))

    # Size 1: the batch if there is one, else the first size the reference accepts at 1.
    batch = _batch_name(small)
    for name in [batch] if batch else sorted(small, key=small.get):
        if _bytes_on_meta(source, small | {name: 1}) is not None:
            add(ShapeClass.SIZE_ONE, small | {name: 1})
            break

    if batch:
        add(ShapeClass.BATCH, _non_pow2(source, small, SMALL_BYTES, rng, names=[batch]))
    else:
        add(ShapeClass.NON_POW2, _non_pow2(source, small, SMALL_BYTES, rng))

    add(ShapeClass.LARGE, _non_pow2(source, large, LARGE_BYTES, rng) or large)

    if input_domain == "any":
        add(ShapeClass.INPUT_DISTRIBUTION, small, "randn")
        add(ShapeClass.INPUT_DISTRIBUTION, small, "shifted")
    return cases


def build_case(source: str, case: HiddenCase, device: str) -> tuple[Any, list[Any], list[Any]]:
    """(reference model, init inputs, inputs) for a case, seeded like reference.build().

    The harness builds the candidate from the same init inputs after torch.manual_seed(INIT_SEED).
    """
    import torch

    ns = load_task_module(with_sizes(source, case.sizes))
    torch.manual_seed(INIT_SEED)
    init = ns["get_init_inputs"]()
    torch.manual_seed(INIT_SEED)
    model = ns["Model"](*init).to(device)
    torch.manual_seed(case.seed)
    inputs = [x.to(device) if isinstance(x, torch.Tensor) else x for x in ns["get_inputs"]()]
    if case.distribution != "default":
        inputs = [_redraw(x, case.distribution) for x in inputs]
    return model, init, inputs


def _redraw(x: Any, distribution: str) -> Any:
    import torch

    if not (isinstance(x, torch.Tensor) and x.is_floating_point()):
        return x  # integer inputs (class labels) keep their domain
    if distribution == "randn":
        return torch.randn_like(x)  # negatives: relu(rand) is the identity, relu(randn) isn't
    if distribution == "shifted":
        return 4 * torch.randn_like(x) + 2  # wider and off-centre: tests stability (softmax)
    raise ValueError(f"unknown distribution {distribution!r}")
