"""Static anti-cheat (B7): scan a candidate's source before it runs. Pure Python, no torch.

Each rule maps to a CheatFlag, and most come from the L4 cheat list:
- torch compute outside @triton.jit kernels (torch.matmul, F.softmax, x @ w, x.sum(), calling
  an nn.Linear/Conv/... submodule, torch.ops / torch._C, torch.compile);
- imports off the allowlist (the reference, cupy, ...);
- file, process or network access;
- TF32: torch.backends / matmul precision, and tl.dot without input_precision="ieee";
- introspection that could find the reference's tensors (L4 #1: gc, inspect, frames, __globals__);
- monkeypatching timers or checks (L4 #3: module attribute assignment, setattr, exec/eval);
- side CUDA streams and CUDA graphs (L4 #2: work a timer might not wait for);
- large literal data (L4 #4: precomputed answers).
Static checks are a first line: B8's runtime checks catch what slips past (e.g. getattr tricks).
"""

import ast
from dataclasses import dataclass

from triton_kernel_agent.harness_result import CheatFlag

# Modules a candidate may import. Everything else is flagged (and categorised below).
ALLOWED_IMPORTS = frozenset(
    {
        "torch",
        "torch.nn",
        "torch.nn.functional",
        "triton",
        "triton.language",
        "triton.runtime",
        "math",
        "typing",
        "functools",
        "dataclasses",
        "itertools",
        "collections",
        "numbers",
        "enum",
    }
)
FILE_OR_NETWORK_MODULES = frozenset(
    {
        "os", "subprocess", "socket", "shutil", "pathlib", "io", "urllib", "http", "requests",
        "pickle", "tempfile", "glob", "ftplib", "smtplib", "shelve", "sqlite3", "torch.hub",
        "torch.utils.cpp_extension",
    }
)  # fmt: skip
INTROSPECTION_MODULES = frozenset(
    {"gc", "inspect", "sys", "ctypes", "traceback", "weakref", "types", "dis", "code", "ast"}
)
MONKEYPATCH_MODULES = frozenset(
    {
        "importlib", "builtins", "time", "timeit", "unittest", "threading", "multiprocessing",
        "signal", "atexit", "triton.testing", "torch.testing",
    }
)  # fmt: skip

# torch functions, F.* functions and Tensor methods that compute. Shape and memory helpers
# (empty, view, contiguous, stride, cat, ...) are fine: they do not do the task's math.
COMPUTE = frozenset(
    {
        # linear algebra
        "matmul", "mm", "bmm", "addmm", "addmv", "mv", "baddbmm", "addbmm", "einsum",
        "tensordot", "dot", "vdot", "inner", "outer", "linear", "bilinear", "chain_matmul",
        # convolution and pooling
        "conv1d", "conv2d", "conv3d", "conv_transpose1d", "conv_transpose2d",
        "conv_transpose3d", "convolution", "max_pool1d", "max_pool2d", "max_pool3d",
        "avg_pool1d", "avg_pool2d", "avg_pool3d", "adaptive_avg_pool1d", "adaptive_avg_pool2d",
        "adaptive_avg_pool3d", "adaptive_max_pool1d", "adaptive_max_pool2d",
        "adaptive_max_pool3d", "lp_pool1d", "lp_pool2d", "interpolate", "unfold", "fold",
        # activations and elementwise math
        "softmax", "log_softmax", "softmin", "relu", "relu6", "gelu", "silu", "mish", "sigmoid",
        "tanh", "hardtanh", "hardswish", "hardsigmoid", "leaky_relu", "elu", "selu", "celu",
        "softplus", "softsign", "logsigmoid", "tanhshrink", "threshold", "glu", "exp", "exp2",
        "expm1", "log", "log2", "log10", "log1p", "sqrt", "rsqrt", "pow", "square", "abs",
        "sin", "cos", "erf", "erfc", "erfinv", "clamp", "clamp_min", "clamp_max", "clip",
        "where", "add", "sub", "subtract", "mul", "multiply", "div", "divide", "true_divide",
        "addcmul", "addcdiv", "maximum", "minimum", "lerp", "neg", "reciprocal", "sign", "dropout",
        # reductions, norms, losses, attention
        "sum", "mean", "prod", "cumsum", "cumprod", "max", "min", "amax", "amin", "argmax",
        "argmin", "logsumexp", "logcumsumexp", "norm", "var", "std", "var_mean", "std_mean",
        "layer_norm", "group_norm", "batch_norm", "instance_norm", "rms_norm",
        "local_response_norm", "normalize", "scaled_dot_product_attention",
        "multi_head_attention_forward", "cross_entropy", "nll_loss", "kl_div", "mse_loss",
        "l1_loss", "smooth_l1_loss", "huber_loss", "binary_cross_entropy",
        "binary_cross_entropy_with_logits", "sort", "topk", "kthvalue", "median", "mode",
        "cdist", "triu", "tril",
    }
)  # fmt: skip
# nn modules whose call does the task's math. Holding one for its weights is fine.
COMPUTE_MODULES = frozenset(
    {
        "Linear", "Bilinear", "Conv1d", "Conv2d", "Conv3d", "ConvTranspose1d",
        "ConvTranspose2d", "ConvTranspose3d", "BatchNorm1d", "BatchNorm2d", "BatchNorm3d",
        "GroupNorm", "LayerNorm", "InstanceNorm1d", "InstanceNorm2d", "InstanceNorm3d",
        "RMSNorm", "Softmax", "LogSoftmax", "ReLU", "GELU", "SiLU", "Mish", "Sigmoid", "Tanh",
        "Hardtanh", "Hardswish", "LeakyReLU", "Softplus", "MaxPool1d", "MaxPool2d", "MaxPool3d",
        "AvgPool1d", "AvgPool2d", "AvgPool3d", "AdaptiveAvgPool2d", "Dropout",
        "MultiheadAttention", "CrossEntropyLoss", "KLDivLoss", "Sequential",
    }
)  # fmt: skip
# Raw op namespaces and compilers: calling these is calling PyTorch's kernels.
RAW_TORCH = ("torch.ops", "torch._C", "torch._", "torch.compile", "torch.jit", "torch._dynamo")
FILE_CALLS = frozenset({"open", "input", "torch.load", "torch.save", "torch.from_file"})
INTROSPECTION_CALLS = frozenset({"globals", "locals", "vars"})
INTROSPECTION_ATTRS = frozenset(
    {
        "__globals__", "__closure__", "__code__", "__subclasses__", "__bases__", "__mro__",
        "__self__", "f_back", "f_locals", "f_globals", "f_code", "gi_frame", "tb_frame",
        "_getframe", "get_objects", "get_referrers", "get_referents",
    }
)  # fmt: skip
MONKEYPATCH_CALLS = frozenset({"setattr", "delattr", "exec", "eval", "compile", "__import__"})
# Builtins that must not even be named outside a direct call: `f = exec; f(src)` or
# `g = getattr` would get past the checks on calls (review of the Y4 scan).
BARE_BUILTINS = MONKEYPATCH_CALLS | INTROSPECTION_CALLS | FILE_CALLS | {"getattr", "hasattr"}
SIDE_STREAM_ATTRS = frozenset(
    {
        "Stream", "stream", "set_stream", "ExternalStream", "CUDAGraph", "graph", "graphs",
        "make_graphed_callables", "wait_stream", "StreamContext",
    }
)  # fmt: skip
BACKEND_ATTRS = frozenset({"backends", "set_float32_matmul_precision", "allow_tf32"})
# Y4 red team: routes around the import allowlist and the checks. Modules reachable as
# attributes of torch (torch.sys is the real sys), and parts of torch that hook its dispatch.
MODULE_ATTRS = frozenset(
    {
        "sys", "os", "modules", "builtins", "importlib", "subprocess", "gc", "inspect",
        "ctypes", "threading", "multiprocessing", "signal", "overrides", "library", "hub",
        "testing", "fx", "export", "distributed", "package", "core", "semantic",
    }
)  # fmt: skip
# Dunder attributes a kernel wrapper needs; any other reaches into objects (x.__dict__,
# type.__setattr__, Tensor.__torch_function__, an explicit __enter__()).
DUNDERS_ALLOWED = frozenset({"__init__", "__name__", "__class__", "__doc__", "__qualname__"})
# What caching and "behave only when checked" need: buffer identity, version counters, memory
# state, hooks and dispatch modes (Y4 A5, A6, B1, B2, D1, E3).
DETECTION_ATTRS = frozenset(
    {
        "data_ptr", "_version", "untyped_storage", "storage", "storage_offset", "_cdata",
        "memory_reserved", "memory_allocated", "max_memory_allocated", "memory_stats",
        "mem_get_info", "as_subclass", "_make_subclass", "fn", "update_wrapper",
    }
)  # fmt: skip
# Strict fp32 (Y2 decision 1): no lower-precision values, even split into several (Y4 D8).
LOW_PRECISION_ATTRS = frozenset(
    {
        "float16", "half", "bfloat16", "float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz",
        "float8_e5m2fnuz", "float8e4nv", "float8e5", "float8e4b15", "float8e4b8", "float8e5b16",
        "fp8e4nv", "fp8e5",
    }
)  # fmt: skip
JIT_DECORATORS = frozenset({"triton.jit", "triton.autotune", "triton.heuristics"})
# Literal data limits: a block size table is fine, a precomputed answer is not.
MAX_LITERAL_NUMBERS = 64
MAX_STRING_CHARS = 4096


@dataclass(frozen=True)
class Finding:
    flag: CheatFlag
    line: int
    message: str


@dataclass(frozen=True)
class StaticReport:
    findings: tuple[Finding, ...]

    @property
    def flags(self) -> tuple[CheatFlag, ...]:
        return tuple(sorted({f.flag for f in self.findings}))

    @property
    def clean(self) -> bool:
        return not self.findings

    def summary(self) -> str:
        return "\n".join(f"line {f.line}: [{f.flag}] {f.message}" for f in self.findings)


def _module_flag(module: str) -> CheatFlag | None:
    """The flag for importing `module`, or None if it is allowed."""

    def matches(names: frozenset[str]) -> bool:
        return any(module == n or module.startswith(n + ".") for n in names)

    if module in ALLOWED_IMPORTS:
        return None
    if matches(FILE_OR_NETWORK_MODULES):
        return CheatFlag.FILE_OR_NETWORK
    if matches(INTROSPECTION_MODULES):
        return CheatFlag.INTROSPECTION
    if matches(MONKEYPATCH_MODULES):
        return CheatFlag.MONKEYPATCH
    return CheatFlag.REFERENCE_IMPORT


class _Scanner(ast.NodeVisitor):
    def __init__(self) -> None:
        self.findings: list[Finding] = []
        self.aliases: dict[str, str] = {}  # local name -> dotted module or object path
        self.compute_attrs: set[str] = set()  # self.<attr> holding a compute nn.Module
        self.in_kernel = False
        self.in_literal = False
        self.call_funcs: set[int] = set()  # id() of nodes in call position

    def flag(self, flag: CheatFlag, node: ast.AST, message: str) -> None:
        self.findings.append(Finding(flag, getattr(node, "lineno", 0), message))

    # --- names ---

    def dotted(self, node: ast.AST) -> str | None:
        """'torch.nn.functional.softmax' for F.softmax, resolving import aliases."""
        parts = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if not isinstance(node, ast.Name):
            return None
        root = self.aliases.get(node.id, node.id)
        path = ".".join([root, *reversed(parts)])
        # tl.core.dot is tl.dot: judge it the same way (it is also flagged as internal).
        return path.replace("triton.language.core.", "triton.language.")

    def is_module_path(self, path: str | None) -> bool:
        if not path:
            return False
        root = path.split(".")[0]
        return root in {"torch", "triton", "math"} or any(
            a.split(".")[0] == root for a in self.aliases.values()
        )

    # --- imports ---

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.aliases[alias.asname or alias.name.split(".")[0]] = (
                alias.name if alias.asname else alias.name.split(".")[0]
            )
            self.check_import(alias.name, node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        if node.level:  # relative import: only the harness has neighbours to import
            self.flag(CheatFlag.REFERENCE_IMPORT, node, "relative import")
            return
        for alias in node.names:
            full = f"{module}.{alias.name}"
            self.aliases[alias.asname or alias.name] = full
            # `from torch import nn` imports a module; `from torch import mm` an object.
            self.check_import(full if full in ALLOWED_IMPORTS else module, node)
            if self.compute_path(full):
                self.flag(CheatFlag.TORCH_COMPUTE_CALL, node, f"imports {full}")

    def check_import(self, module: str, node: ast.AST) -> None:
        if flag := _module_flag(module):
            self.flag(flag, node, f"import of {module} is not allowed")

    # --- functions: track whether we are inside a Triton kernel ---

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        for dec in node.decorator_list:
            self.visit(dec)
        decorators = {
            self.dotted(d.func if isinstance(d, ast.Call) else d) for d in node.decorator_list
        }
        if decorators & JIT_DECORATORS and not decorators <= JIT_DECORATORS:
            # @wrap @triton.jit can hand back the plain Python function (Y4 A s1).
            self.flag(CheatFlag.MONKEYPATCH, node, "other decorators around @triton.jit")
        was = self.in_kernel
        self.in_kernel = was or bool(decorators & JIT_DECORATORS)
        for stmt in node.body:
            self.visit(stmt)
        self.in_kernel = was

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        for base in node.bases:
            path = self.dotted(base) or ""
            if path.startswith("torch.") and path.split(".")[-1] in ("Tensor", "TensorBase"):
                # A subclass can answer the checker's own arithmetic (Y4 C01, E1).
                self.flag(CheatFlag.INTROSPECTION, node, f"subclasses {path}")
        self.generic_visit(node)

    # --- calls ---

    def compute_path(self, path: str | None) -> bool:
        if not path or not path.startswith("torch"):
            return False
        if any(path == r or path.startswith(r if r.endswith("_") else r + ".") for r in RAW_TORCH):
            return True
        return path.split(".")[-1].strip("_") in COMPUTE

    def visit_Call(self, node: ast.Call) -> None:
        self.call_funcs.add(id(node.func))
        path = self.dotted(node.func)
        name = path.split(".")[-1] if path else None

        if self.in_kernel:
            if path in ("triton.language.dot", "triton.language.dot_scaled"):
                self.check_dot(node, path)
            elif path and path.split(".")[0] == "torch":
                self.flag(CheatFlag.TORCH_COMPUTE_CALL, node, f"calls {path} inside a kernel")
        else:
            self.check_compute_call(node, path)
        if isinstance(node.func, ast.Call):
            ctor = (self.dotted(node.func.func) or "").split(".")[-1]
            if ctor in COMPUTE_MODULES:
                self.flag(CheatFlag.TORCH_COMPUTE_CALL, node, f"calls a new nn.{ctor} module")

        if name in MONKEYPATCH_CALLS and path == name:
            self.flag(CheatFlag.MONKEYPATCH, node, f"{name}() can rewrite code at runtime")
        if path in ("getattr", "hasattr") and node.args:
            target = node.args[1] if len(node.args) > 1 else None
            if isinstance(target, ast.Constant) and isinstance(target.value, str):
                attr_path = f"{self.dotted(node.args[0])}.{target.value}"
                if self.compute_path(attr_path):
                    self.flag(CheatFlag.TORCH_COMPUTE_CALL, node, f"getattr reaches {attr_path}")
                # getattr(x, "name") is x.name: same attribute rules.
                fake = ast.Attribute(value=node.args[0], attr=target.value, ctx=ast.Load())
                ast.copy_location(fake, node)
                self.check_attribute(fake, self.dotted(fake))
            else:  # a computed name hides what is reached ("mat" + "mul", Y4 A s3)
                self.flag(CheatFlag.MONKEYPATCH, node, f"{path} with a computed attribute name")
        if path in FILE_CALLS or (path or "").startswith(("numpy.load", "numpy.fromfile")):
            self.flag(CheatFlag.FILE_OR_NETWORK, node, f"{path}() reads or writes files")
        if path in INTROSPECTION_CALLS:
            self.flag(CheatFlag.INTROSPECTION, node, f"{path}() exposes other code's variables")
        if path and path.startswith("torch") and "set_float32_matmul_precision" in path:
            self.flag(CheatFlag.BACKEND_FLAGS, node, "changes the fp32 matmul precision")
        if any(k.arg == "stream" for k in node.keywords):
            self.flag(CheatFlag.SIDE_STREAM, node, "launches on an explicit stream")
        self.generic_visit(node)

    def check_compute_call(self, node: ast.Call, path: str | None) -> None:
        func = node.func
        if self.compute_path(path):
            self.flag(CheatFlag.TORCH_COMPUTE_CALL, node, f"calls {path} outside a Triton kernel")
            return
        if isinstance(func, ast.Attribute):
            receiver = self.dotted(func.value)
            root = (receiver or "").split(".")[0]
            # self.gemm(x) where self.gemm = nn.Linear(...): cuBLAS does the work.
            if receiver == "self" and func.attr in self.compute_attrs:
                self.flag(CheatFlag.TORCH_COMPUTE_CALL, node, f"calls the self.{func.attr} module")
            # x.sum(), y.softmax(...): Tensor methods that compute. Skip modules and self.
            elif (
                root not in {"self", "triton", "math"}
                and not self.is_module_path(receiver)
                and func.attr.strip("_") in COMPUTE
            ):
                self.flag(CheatFlag.TORCH_COMPUTE_CALL, node, f".{func.attr}() computes in PyTorch")

    def check_dot(self, node: ast.Call, path: str) -> None:
        if path.endswith("dot_scaled"):
            self.flag(CheatFlag.TF32_DOT, node, "tl.dot_scaled is low precision")
            return
        precision = next((k.value for k in node.keywords if k.arg == "input_precision"), None)
        if not (isinstance(precision, ast.Constant) and precision.value == "ieee"):
            self.flag(CheatFlag.TF32_DOT, node, 'tl.dot needs input_precision="ieee" (strict fp32)')

    # --- attributes, assignments and literals ---

    def visit_Attribute(self, node: ast.Attribute) -> None:
        self.check_attribute(node, self.dotted(node))
        self.generic_visit(node)

    def check_attribute(self, node: ast.Attribute, path: str | None) -> None:
        attr = node.attr
        module = self.is_module_path(path)
        # torch.utils holds the dispatch and pytree internals; triton's driver.active.utils
        # (device properties, used by the official tutorials) is fine.
        torch_utils = attr == "utils" and (path or "").startswith("torch.")
        if module and (attr in MODULE_ATTRS or torch_utils):
            self.flag(CheatFlag.INTROSPECTION, node, f"{path} reaches past the import allowlist")
        elif module and attr.startswith("_") and not attr.startswith("__"):
            self.flag(CheatFlag.INTROSPECTION, node, f"{path} is a private internal")
        dunder = attr.startswith("__") and attr.endswith("__")
        # x.__matmul__(w) is reported as compute; other dunders reach into objects.
        if dunder and attr not in DUNDERS_ALLOWED and attr.strip("_") not in COMPUTE:
            self.flag(CheatFlag.INTROSPECTION, node, f".{attr} reaches into objects")
        if attr in DETECTION_ATTRS or "hook" in attr.lower():
            self.flag(CheatFlag.INTROSPECTION, node, f".{attr} can tell checked calls apart")
        if attr in LOW_PRECISION_ATTRS:
            self.flag(CheatFlag.LOW_PRECISION, node, f"{attr} is below strict fp32")
        if path in ("triton.language.dot", "triton.language.dot_scaled") and (
            id(node) not in self.call_funcs
        ):
            self.flag(CheatFlag.TF32_DOT, node, "tl.dot used as a value hides its precision")
        if attr in INTROSPECTION_ATTRS:
            self.flag(CheatFlag.INTROSPECTION, node, f".{node.attr} reaches into other code")
        if node.attr == "__builtins__":
            self.flag(CheatFlag.MONKEYPATCH, node, "touches __builtins__")
        if (
            path
            and path.startswith("torch")
            and (set(path.split(".")) & BACKEND_ATTRS or "fp32_precision" in path)
        ):
            self.flag(CheatFlag.BACKEND_FLAGS, node, f"touches {path}")
        if (
            path
            and path.startswith(("torch.cuda.", "torch.Stream"))
            and node.attr in SIDE_STREAM_ATTRS
        ):
            self.flag(CheatFlag.SIDE_STREAM, node, f"uses {path}")

    def visit_Name(self, node: ast.Name) -> None:
        if node.id == "__builtins__":
            self.flag(CheatFlag.MONKEYPATCH, node, "touches __builtins__")
        if node.id in BARE_BUILTINS and id(node) not in self.call_funcs:
            self.flag(CheatFlag.MONKEYPATCH, node, f"{node.id} used as a value, not called")

    def check_targets(self, targets: list[ast.expr], node: ast.AST) -> None:
        for target in targets:
            for t in target.elts if isinstance(target, ast.Tuple) else [target]:
                if not isinstance(t, ast.Attribute):
                    continue
                path = self.dotted(t)
                if path and path.startswith("torch") and "backends" in path.split("."):
                    continue  # reported as backend_flags by visit_Attribute
                if self.is_module_path(path):
                    self.flag(CheatFlag.MONKEYPATCH, node, f"assigns {path}")

    def visit_Assign(self, node: ast.Assign) -> None:
        self.check_targets(node.targets, node)
        # cu = torch.cuda; cu.Stream(): follow simple aliases of module paths (Y4 B4, D4).
        value = (
            self.dotted(node.value) if isinstance(node.value, ast.Attribute | ast.Name) else None
        )
        if value and value.split(".")[0] in {"torch", "triton"}:
            for t in node.targets:
                if isinstance(t, ast.Name):
                    self.aliases[t.id] = value
        # Remember self.x = nn.Linear(...) so a later self.x(...) call is caught.
        if isinstance(node.value, ast.Call):
            ctor = (self.dotted(node.value.func) or "").split(".")[-1]
            for t in node.targets:
                if (
                    ctor in COMPUTE_MODULES
                    and isinstance(t, ast.Attribute)
                    and self.dotted(t.value) == "self"
                ):
                    self.compute_attrs.add(t.attr)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self.check_targets([node.target], node)
        self.generic_visit(node)

    def visit_Delete(self, node: ast.Delete) -> None:
        self.check_targets(node.targets, node)
        self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if isinstance(node.op, ast.MatMult) and not self.in_kernel:
            self.flag(CheatFlag.TORCH_COMPUTE_CALL, node, "the @ operator runs a PyTorch matmul")
        self.generic_visit(node)

    def check_literal(self, node: ast.AST) -> None:
        numbers = sum(
            1
            for n in ast.walk(node)
            if isinstance(n, ast.Constant) and isinstance(n.value, int | float | complex)
        )
        if numbers > MAX_LITERAL_NUMBERS:
            self.flag(CheatFlag.HARDCODED_DATA, node, f"literal with {numbers} numbers")

    def visit_List(self, node: ast.List) -> None:
        # Data is a literal made only of numbers (nested lists/tuples of them): an answer table.
        # A list of triton.Config(...) calls has as many numbers but is configuration.
        # Count once, at the outermost data literal; always scan children for calls.
        data = _is_numeric_data(node)
        if data and not self.in_literal:
            self.check_literal(node)
        was = self.in_literal
        self.in_literal = was or data
        self.generic_visit(node)
        self.in_literal = was

    visit_Tuple = visit_Set = visit_Dict = visit_List

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, str | bytes) and len(node.value) > MAX_STRING_CHARS:
            self.flag(CheatFlag.HARDCODED_DATA, node, f"literal of {len(node.value)} characters")


def _is_numeric_data(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant):
        return isinstance(node.value, int | float | complex) and not isinstance(node.value, bool)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub | ast.UAdd):
        return _is_numeric_data(node.operand)
    if isinstance(node, ast.List | ast.Tuple | ast.Set):
        return bool(node.elts) and all(_is_numeric_data(e) for e in node.elts)
    return False


def scan(source: str) -> StaticReport:
    """Scan candidate source. Raises SyntaxError if it does not parse (a compile error)."""
    scanner = _Scanner()
    scanner.visit(ast.parse(source))
    # One finding per flag and line: torch.backends.cuda.matmul.allow_tf32 is one problem.
    unique: dict[tuple[CheatFlag, int], Finding] = {}
    for finding in scanner.findings:
        unique.setdefault((finding.flag, finding.line), finding)
    return StaticReport(tuple(sorted(unique.values(), key=lambda f: (f.line, f.flag))))
