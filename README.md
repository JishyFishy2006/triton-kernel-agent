# Triton Kernel Agent

An agent where Muse Wonder writes Triton kernels for PyTorch ops, and a hardened
harness on Modal H100s checks correctness, blocks cheating, and benchmarks speed.

Roadmap and tech spec: https://claude.ai/artifact/7JoEkyFCszJkHTs5SF1Tdr

## Problem

LLMs can write GPU kernels, but the numbers they report are easy to fake. Published kernel
agents have been caught returning cached outputs, calling back into PyTorch, dropping to
lower precision, or timing only part of the work. A speedup only means something if the
harness that measured it could not be fooled by the code it was measuring.

This project asks two questions on 43 KernelBench-style tasks (Level 1 single ops, Level 2
fused chains, and 4 tasks of our own) on an H100 in strict fp32:

1. How often can an agent write a Triton kernel that is correct and faster than PyTorch?
2. Which model setup is worth paying for: the large model (Muse Wonder), the small one
   (Muse Spark), or Wonder writing with Spark fixing compile errors?

## Method

**Agent loop.** For each task the planner model gets the PyTorch source and writes a Triton
`ModelNew`. The harness result goes back as feedback (compile error, which shape failed and
by how much, the speedup, the % of peak bandwidth), and the agent tries again, up to 10
attempts. Compile errors go to the fixer model with a short repair prompt. The best correct
attempt counts.

**Harness** (tag `harness-v1`, details in `decisions/harness_isolation.md`):

- *Isolation.* Each eval runs in a fresh Modal Sandbox with no network and no secrets. Inside
  it, a trusted parent process never imports the candidate. The candidate runs in a worker as
  the unprivileged `nobody` user, which gets inputs and returns outputs as raw tensor files.
- *Correctness.* The parent computes the reference in fp64 and fp32. The candidate's error may
  be at most 16× the fp32 reference's own error, with a 4-ulp floor. It runs on the visible
  shape, two input sets, and at least 5 hidden shapes (non-power-of-2, size 1, large, shifted
  inputs). After the hidden stage, error messages are withheld.
- *Timing.* Measured on the parent's clock, in batches that alternate between two checked
  input sets written into the same buffers, with a random number of calls per batch. The last
  output of each batch is checked again. PyTorch eager and `torch.compile` baselines are timed
  in clean workers with no candidate process alive.
- *Cheat checks.* A static scan flags PyTorch compute calls, TF32 and fp16/bf16/fp8,
  introspection, reference imports and hardcoded data. Runtime checks flag outputs that ignore
  the inputs, results that change after the checks, and any time faster than the bytes moved
  allow at the H100's peak bandwidth (`impossible_speed`).
- *Red team* (Y4). 37 attack kernels are kept as fixtures. With the static scan turned off, the
  runtime checks alone stop 32 of them. The 5 that only the scan stops are listed under
  Failure modes.

**Setups.**

| Setup | Writes kernels | Fixes compile errors |
|---|---|---|
| `wonder_only` | Muse Wonder 1.0 | Muse Wonder 1.0 |
| `spark_only` | Muse Spark 1.3 | Muse Spark 1.3 |
| `wonder_spark` | Muse Wonder 1.0 | Muse Spark 1.3 |

Tasks are split 12 dev / 31 held-out (`decisions/tasks_chosen.md`). Prompts and the harness
were tuned on dev only. The headline numbers are held-out.

## Results

One batch per setup on 2026-11-03 (`batch-<setup>-20261103`), 8 tasks in parallel, harness v1.
`uv run tka report` builds the full page (`site/index.html`) from `data/results.duckdb`.

**Held-out, 31 tasks** (from `tka report`):

| Setup | Solved | fast₁ (correct and > 1× eager) | fast₂ | > 1× `torch.compile` | Correct on attempt 1 | Median speedup | Geomean speedup | API $ / correct task |
|---|---|---|---|---|---|---|---|---|
| `wonder_only` | **31/31** | **47%** | **20%** | 23% | 65% | 0.95× | **1.28×** | $0.85 |
| `wonder_spark` | **31/31** | **47%** | 17% | **33%** | 61% | **0.97×** | 1.14× | $0.91 |
| `spark_only` | 30/31 | 40% | 17% | 20% | **87%** | 0.82× | 1.01× | **$0.37** |

All 43 tasks: `wonder_only` and `wonder_spark` solved 43/43, `spark_only` 42/43 (it never
solved L1-8, a 3D tensor-matrix product: 8 mismatches in 10 attempts).

**By level** (held-out geomean vs eager):

| Setup | Level 1 | Level 2 + custom | All held-out without L2-14 |
|---|---|---|---|
| `wonder_only` | 1.76× | 0.89× | 1.16× |
| `wonder_spark` | 1.64× | 0.76× | 1.03× |
| `spark_only` | 1.49× | 0.66× | 0.91× |

L2-14 is an algebraic shortcut (see below). The Y2 rules count it as correct but say to report
it separately, and the report page does not do that yet, so the last column leaves it out by
hand.

**Against a human baseline.** On the two tasks where we have hand-tuned Triton (the official
Triton tutorial softmax and LayerNorm, adapted to these shapes; `baselines/`), run through
the same harness:

| Task | Triton tutorial (adapted) | `wonder_only` | `wonder_spark` | `spark_only` |
|---|---|---|---|---|
| L1-23 softmax | 1.25× | 1.26× | 1.29× | 1.23× |
| L1-40 LayerNorm | 27.99× | 26.99× | 16.63× | 16.62× |

**What it means.**

- On single ops (Level 1) the agent is clearly useful: a 1.5–1.8× geomean over eager, and
  near the human kernels on softmax and LayerNorm.
- On fused GEMM chains (Level 2) in strict fp32 it mostly loses to cuBLAS. That matches what
  we expected in Y2: `torch.compile` only gains a median 1.01× on those tasks.
- Wonder writes faster kernels. Spark gets almost the same pass rate, more often on the first
  try, at 36% of the cost.
- Adding Spark as Wonder's fixer saved nothing: the fixer was called twice in 120 attempts,
  because Wonder rarely fails to compile.

`docs/interesting_kernels.md` walks through the three most interesting kernels (Y7):

- **L1-40 LayerNorm** spreads 16 long rows over all 132 SMs (27× eager, 1.5× `torch.compile`).
- **L2-14** replaces a GEMM with a column sum and dot product (23×).
- **L1-88 GELU** fuses seven ops into one pass at 88% of peak bandwidth (9×).

## Failure modes

**What the attempts got wrong** (347 attempts across the three batches):

| Result | Attempts | Notes |
|---|---|---|
| correct | 234 | |
| static_cheat | 47 | 40 `low_precision` (below; one also flagged introspection). The other 7: introspection 4, PyTorch compute 1, hardcoded data 1, reference import 1. |
| compile_error | 31 | 17 from `spark_only`, 10 of those because Spark's reply did not match the required JSON format. Wonder never broke the format. |
| mismatch | 24 | 19 on hidden cases: size-1 dimensions 11, non-power-of-2 4, shifted inputs 3, large 1. 5 on the visible shape. |
| runtime_error | 8 | 3 illegal memory accesses, 2 Triton compiler crashes at launch, 3 `TypeError`s in `forward`. |
| timeout | 2 | |
| wrong_shape | 1 | |

**Wonder reaches for lower precision.** 37 of the 40 `low_precision` flags came from Wonder.
It splits fp32 GEMMs into bf16 or fp16 pieces to use tensor cores, which is fast but not strict
fp32. The scan blocks it and the next attempt usually goes back to fp32. Spark almost never
tries it (3 flags).

**No runtime cheat fired in the batch.** Every attempt that passed the static scan and the
correctness checks also passed the input-sensitivity, after-checks and `impossible_speed`
checks. The red team shows those checks catch real attacks, so this suggests the models were
not trying to game the timer, not that the checks are idle. Two known limits:

- **5 red-team attacks are stopped only by the static scan.** Three call PyTorch or cuBLAS in
  ways the runtime audit cannot see (fixtures a5, c04, e4). Two emulate fp32 GEMMs with split
  bf16/fp16 (d4, d8).
- **Value-keyed caching** that stays under the bandwidth floor is not detected at runtime.

**Hand audit.** The Y6 audit of 10 passing kernels has not been done yet. Until it is, a
passing kernel means "passed the harness", not "checked by a person".

## Cost

| | `wonder_spark` | `spark_only` | `wonder_only` | Total |
|---|---|---|---|---|
| Model calls | 121 | 113 | 116 | 352 |
| Meta API | $31.28 | $11.11 | $30.53 | $73.30 |

- **Meta API total:** $73.30, from `traces/model_calls.jsonl`. It includes two untagged calls,
  which is why it is a little more than the three columns.
- **Model calls:** each setup's count includes calls whose replies failed the format check.
- **GPU:** the evals used 4.1 H100-hours, about $16 at $3.95/h. The Modal bill also covers
  sandbox start-up and idle time; the run's guard budgeted $35.20 for it.
- **Wall clock:** about 3.5 hours for all three setups, run one after another. That includes
  a 25-minute stall in the first run, when a model stream only sent keepalives. A stream
  deadline (PR #27) now cuts such streams off.
- **Cost per correct task:** about $0.37 for Spark and $0.85–0.91 for Wonder. The extra cost
  of Wonder buys speed, not pass rate.

## Running the agent

```sh
uv run tka demo L1-19                 # one task end to end, with a live view of each attempt
uv run tka run --split dev --workers 4   # batch run (resume with --run-id <id>)
uv run tka report                     # results DB -> site/index.html
```

`--setup spark_only|wonder_only|wonder_spark` picks the model setup (default from
`config.toml`). Every attempt, its code and its harness result go to `data/results.duckdb`.

## Development

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11.

```sh
uv sync                     # create .venv and install deps
cp .env.example .env        # then fill in META_API_KEY
uv run pytest               # tests (offline; no API spend)
uv run pytest -m live       # real Meta API call + Modal H100 run (spends a few cents)
uv run ruff check .         # lint
uv run ruff format .        # format
uv run modal setup          # one-time Modal login
uv run modal run src/triton_kernel_agent/modal_hello.py   # which GPU do we get?
uv run modal run src/triton_kernel_agent/modal_tasks.py   # re-measure tasks/metadata.json
uv run modal run src/triton_kernel_agent/modal_check.py   # correctness checker on hidden cases
uv run python -m triton_kernel_agent.report               # results DB -> site/index.html
```

Settings live in two places:

- `config.toml`: non-secret settings (models, budget, Modal GPU, paths). Committed.
- `.env`: secrets (`META_API_KEY`). Git-ignored. Variables already set in your shell win.

Every model call is appended to `traces/model_calls.jsonl` with its token
counts and dollar cost. Prices live in `src/triton_kernel_agent/pricing.py`.

## Tasks

The task list is decided in `decisions/tasks_chosen.md` (Y2) and recorded in
`tasks/manifest.toml`: 43 scored tasks (12 dev, 31 held-out) plus two unscored canaries.

- `tasks/kernelbench/`: the chosen KernelBench sources, vendored unchanged from commit `423217d`.
- `tasks/custom/`: our own tasks C-1 to C-4, in KernelBench format.
- `tasks/metadata.json`: per-task shapes, bytes and FLOPs per forward call, measured on an H100.

```python
from triton_kernel_agent.tasks import load_tasks

dev = load_tasks(split="dev")  # 12 tasks; split="held_out" gives 31, None gives all 43
task = dev[0]
task.id, task.source, task.atol, task.bytes, task.flops
```

`scripts/y2_screen/` holds the Y2 screening script and its raw results.

## Harness

- `harness_result.py`: the one result every eval returns (`docs/harness_result.md`).
- `correctness.py` (B5): strict fp32; the candidate's error vs an fp64 reference may be at most
  16x the fp32 reference's own error, with a floor of 4 ulps of the output's largest value.
- `db.py` (B12): DuckDB file (`data/results.duckdb`) with tasks, runs, attempts and evals;
  one eval row per HarnessResult, speedups as columns.
- `roofline.py` (B10): achieved bandwidth and fp32 FLOP rate as a percentage of H100 peak, from
  the task's bytes and FLOPs; `modal_ncu.py` adds opt-in Nsight Compute profiles
  (`decisions/ncu_on_modal.md`).
- `hidden_shapes.py` (B6): at least 5 seeded hidden cases per task (non-power-of-2, size 1,
  varied batch, one large case under 1 GiB, randn and shifted inputs).
