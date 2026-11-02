"""Record real model replies for B13's schema test (tests/fixtures/responses/). Spends credit.

Run with: uv run python scripts/sample_responses.py L1-19 L1-36 L2-76
"""

import json
import sys
from pathlib import Path

from triton_kernel_agent.config import load_config
from triton_kernel_agent.env import load_secrets
from triton_kernel_agent.llm import ModelClient
from triton_kernel_agent.prompts import RESPONSE_TEXT, planner_messages, task_prompt
from triton_kernel_agent.tasks import METADATA_PATH, get_task

OUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "responses"


def main(task_ids: list[str]) -> None:
    cfg = load_config()
    client = ModelClient.from_config(cfg, load_secrets())
    metadata = json.loads(METADATA_PATH.read_text())["tasks"]
    OUT.mkdir(parents=True, exist_ok=True)
    for task_id in task_ids:
        task, meta = get_task(task_id), metadata[task_id]
        text = task_prompt(task, meta["input_shapes"], meta["output_shape"])
        result = client.complete(
            cfg.models.planner,
            planner_messages(text, []),
            tags={"purpose": "b13_sample", "task_id": task_id},
            text=RESPONSE_TEXT,
        )
        record = {
            "task_id": task_id,
            "model": result.model,
            "reasoning_effort": result.reasoning_effort,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "reasoning_tokens": result.reasoning_tokens,
            "cost_usd": result.cost_usd,
            "text": result.text,
        }
        (OUT / f"{task_id}.json").write_text(json.dumps(record, indent=1) + "\n")
        print(f"{task_id}: {result.output_tokens} output tokens, ${result.cost_usd:.4f}")


if __name__ == "__main__":
    main(sys.argv[1:])
