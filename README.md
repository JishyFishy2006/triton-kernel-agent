# Triton Kernel Agent

An agent where Muse Wonder writes Triton kernels for PyTorch ops, and a hardened
harness on Modal H100s checks correctness, blocks cheating, and benchmarks speed.

Roadmap and tech spec: https://claude.ai/artifact/7JoEkyFCszJkHTs5SF1Tdr

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
