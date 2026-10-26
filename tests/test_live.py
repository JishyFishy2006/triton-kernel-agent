"""Live call to the Meta Model API. Costs a fraction of a cent.

Run with: uv run pytest -m live
"""

import json
from pathlib import Path

import pytest

from triton_kernel_agent.config import load_config
from triton_kernel_agent.env import load_secrets
from triton_kernel_agent.llm import ModelClient

pytestmark = pytest.mark.live


def test_one_live_call_logs_a_row(tmp_path: Path):
    try:
        secrets = load_secrets()
    except RuntimeError as e:
        pytest.skip(str(e))
    cfg = load_config()
    log = tmp_path / "calls.jsonl"
    client = ModelClient(cfg.models, secrets.meta_api_key, log)

    result = client.complete(
        cfg.models.fixer,
        [{"role": "user", "content": "Reply with the single word OK."}],
        tags={"test": "live"},
    )

    assert result.text.strip()
    (row,) = [json.loads(line) for line in log.read_text().splitlines()]
    assert row["model"] == cfg.models.fixer.name
    assert row["input_tokens"] > 0
    assert row["output_tokens"] > 0
    assert row["cost_usd"] > 0
    print(f"\nlive call: {row}")
