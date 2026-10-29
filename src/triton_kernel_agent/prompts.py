"""Prompt templates and the structured-output schema (B13).

The model answers every request with JSON {plan_summary, code}. The system prompt states the
harness's actual rules (B5-B11), so a candidate is never rejected for something it wasn't told.
Feedback text after an eval comes from B15 (feedback.py); this module only lays out messages.
"""

import json
import re
from dataclasses import dataclass
from typing import Any

from triton_kernel_agent.roofline import H100_SXM
from triton_kernel_agent.tasks import Task

RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "plan_summary": {
            "type": "string",
            "description": "2-5 sentences: kernel design (fusion, tiling, block sizes) and why.",
        },
        "code": {
            "type": "string",
            "description": "One complete Python module defining ModelNew. No markdown fences.",
        },
    },
    "required": ["plan_summary", "code"],
    "additionalProperties": False,
}
# The Responses API's structured-output setting: pass as complete(..., text=RESPONSE_TEXT).
RESPONSE_TEXT: dict[str, Any] = {
    "format": {
        "type": "json_schema",
        "name": "kernel_response",
        "strict": True,
        "schema": RESPONSE_SCHEMA,
    }
}
# How many earlier attempts go back to the model in full; older ones are one line each.
HISTORY_IN_FULL = 3

SYSTEM_PROMPT = f"""\
You write Triton GPU kernels that replace PyTorch reference code on an NVIDIA H100 SXM \
(peak {H100_SXM.bandwidth_bytes_per_s / 1e12:.2f} TB/s HBM3, \
{H100_SXM.fp32_flops_per_s / 1e12:.0f} TFLOPS fp32 on CUDA cores).

You get a reference module `Model` with `forward`, plus `get_inputs()` and `get_init_inputs()`.
Return a Python module that defines `class ModelNew(nn.Module)`:
- `ModelNew.__init__` takes the same arguments as `Model.__init__` and creates parameters in the
  same order with the same shapes, so seeded initialisation gives identical weights. Keeping the
  reference's nn.Linear / nn.Conv modules as weight holders is fine; calling them is not.
- `ModelNew.forward` takes the same inputs and returns the same output (shape and dtype). The
  real work happens in your own `@triton.jit` kernels.

How your code is checked (a failure at any step ends the attempt):
1. Static scan. Outside `@triton.jit` functions you may allocate (`torch.empty`, ...), reshape,
   view, make contiguous and launch kernels. You may NOT compute with PyTorch: no torch.* or F.*
   math, no tensor methods like `.sum()` or `.exp()`, no `@`, no calling nn modules, no
   torch.ops / torch._C / torch.compile. Allowed imports: torch, torch.nn,
   torch.nn.functional (imports only), triton, triton.language, math, typing, functools,
   dataclasses, itertools, collections, numbers, enum. No files, network, subprocesses,
   introspection (gc, inspect, sys, frames), monkeypatching (setattr, exec, assigning module
   attributes), CUDA streams or graphs, or large literal data.
2. Precision is strict IEEE fp32. Every `tl.dot` must pass `input_precision="ieee"`. Never touch
   torch.backends or TF32 settings.
3. Correctness against an fp64 run of the reference on the visible inputs and on hidden inputs:
   non-power-of-2 sizes, size 1, a different batch size, one large case, and inputs drawn from
   other distributions (negative, wider, shifted). Use masks; never assume shapes, strides,
   sizes divisible by block sizes, or positive inputs. Your worst error may be at most 16x the
   fp32 reference's own error. Models run in training mode (BatchNorm uses batch statistics).
4. Runtime checks: only your Triton kernels may compute (PyTorch ops and cuBLAS are detected);
   outputs must change when input values change (no caching), must not alias inputs, and must
   be written by your kernels (unwritten memory is filled with NaN).
5. Timing: median of many runs with L2 flushed, against PyTorch eager and torch.compile, then
   the output is checked again.

Respond with JSON only: {{"plan_summary": "...", "code": "..."}}. `code` is the complete module
as plain text, without markdown fences.
"""

FIXER_SYSTEM_PROMPT = """\
You fix Python/Triton code that failed to compile or import. Change as little as possible:
keep the kernel design and fix only the reported error. The same rules apply as before
(Triton kernels only, tl.dot with input_precision="ieee", no PyTorch compute outside kernels).
Respond with JSON only: {"plan_summary": "...", "code": "..."} where code is the full module.
"""


@dataclass(frozen=True)
class KernelResponse:
    plan_summary: str
    code: str


class ResponseError(ValueError):
    """The model's reply does not match RESPONSE_SCHEMA."""


_FENCE = re.compile(r"^\s*```[a-zA-Z]*\s*\n(.*?)\n?\s*```\s*$", re.DOTALL)


def _unfence(text: str) -> str:
    match = _FENCE.match(text)
    return match.group(1) if match else text


def parse_response(text: str) -> KernelResponse:
    """Validate a reply against RESPONSE_SCHEMA. Tolerates a markdown fence around the JSON
    or around the code, which models add despite instructions."""
    try:
        data = json.loads(_unfence(text.strip()))
    except json.JSONDecodeError as e:
        raise ResponseError(f"reply is not JSON: {e}") from e
    if not isinstance(data, dict):
        raise ResponseError(f"reply is a JSON {type(data).__name__}, not an object")
    missing = [k for k in RESPONSE_SCHEMA["required"] if k not in data]
    extra = [k for k in data if k not in RESPONSE_SCHEMA["properties"]]
    if missing or extra:
        raise ResponseError(f"reply keys: missing {missing}, unexpected {extra}")
    for key in RESPONSE_SCHEMA["required"]:
        if not isinstance(data[key], str):
            raise ResponseError(f"{key} must be a string, got {type(data[key]).__name__}")
    code = _unfence(data["code"]).strip()
    if not code:
        raise ResponseError("code is empty")
    return KernelResponse(plan_summary=data["plan_summary"].strip(), code=code + "\n")


def task_prompt(
    task: Task, input_shapes: list | None = None, output_shape: list | None = None
) -> str:
    """The first user message for a task: the reference and what the roofline says about it."""
    lines = [f"Task {task.id}: {task.name.replace('_', ' ')}.", ""]
    if input_shapes is not None:
        lines.append(f"Visible input shapes: {input_shapes}; output shape: {output_shape}.")
    if task.bytes:
        lines.append(f"Minimum memory traffic per call: {task.bytes / 1e6:,.1f} MB.")
        if task.flops:
            intensity = task.flops / task.bytes
            lines.append(
                f"Matmul/conv FLOPs per call: {task.flops / 1e9:,.2f} GFLOP "
                f"({intensity:.1f} FLOP/byte; the H100 fp32 ridge point is ~20)."
            )
        floor_ms = task.bytes / H100_SXM.bandwidth_bytes_per_s * 1e3
        lines.append(f"At peak bandwidth the data alone takes {floor_ms:.3f} ms.")
    lines += ["", "Reference:", "```python", task.source.rstrip(), "```", ""]
    lines.append("Write ModelNew with Triton kernels that are correct and faster than this.")
    return "\n".join(lines)


@dataclass(frozen=True)
class Turn:
    """One earlier attempt: what the model sent and the feedback it got (from B15)."""

    response: KernelResponse
    feedback: str


def _assistant(response: KernelResponse) -> dict[str, str]:
    content = json.dumps({"plan_summary": response.plan_summary, "code": response.code})
    return {"role": "assistant", "content": content}


def planner_messages(task_text: str, history: list[Turn]) -> list[dict[str, str]]:
    """Messages for the planner/writer: system, task, then earlier attempts and feedback."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task_text},
    ]
    older, recent = history[:-HISTORY_IN_FULL], history[-HISTORY_IN_FULL:]
    if older:
        summary = "\n".join(
            f"- Attempt {i + 1}: {t.response.plan_summary} -> {t.feedback.splitlines()[0]}"
            for i, t in enumerate(older)
        )
        messages[-1] = {"role": "user", "content": f"{task_text}\n\nEarlier attempts:\n{summary}"}
    for turn in recent:
        messages += [_assistant(turn.response), {"role": "user", "content": turn.feedback}]
    return messages


def fixer_messages(task_text: str, broken: KernelResponse, error: str) -> list[dict[str, str]]:
    """Messages for the cheap fixer (compile and import errors only)."""
    return [
        {"role": "system", "content": FIXER_SYSTEM_PROMPT},
        {"role": "user", "content": task_text},
        _assistant(broken),
        {"role": "user", "content": f"This failed to compile or import:\n\n{error}\n\nFix it."},
    ]
