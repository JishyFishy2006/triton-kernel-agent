# Triton Kernel Agent

An agent where Muse Wonder writes Triton kernels for PyTorch ops, and a hardened
harness on Modal H100s checks correctness, blocks cheating, and benchmarks speed.

Roadmap and tech spec: https://claude.ai/artifact/7JoEkyFCszJkHTs5SF1Tdr

## Development

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11.

```sh
uv sync                     # create .venv and install deps
cp .env.example .env        # then fill in META_API_KEY
uv run pytest               # tests
uv run ruff check .         # lint
uv run ruff format .        # format
```

Settings live in two places:

- `config.toml`: non-secret settings (models, budget, Modal GPU, paths). Committed.
- `.env`: secrets (`META_API_KEY`). Git-ignored. Variables already set in your shell win.
