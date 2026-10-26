"""Load non-secret settings from config.toml into typed, frozen dataclasses."""

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.toml"

# "none" is also an API value, but Muse Spark rejects it with HTTP 400.
REASONING_EFFORTS = ("minimal", "low", "medium", "high", "xhigh")
MODEL_SETUPS = ("spark_only", "wonder_only", "wonder_spark")


@dataclass(frozen=True)
class ModelConfig:
    name: str
    reasoning_effort: str


@dataclass(frozen=True)
class ModelsConfig:
    base_url: str
    setup: str
    max_retries: int
    timeout_s: float
    planner: ModelConfig
    fixer: ModelConfig


@dataclass(frozen=True)
class BudgetConfig:
    max_attempts: int
    max_usd_per_task: float


@dataclass(frozen=True)
class ModalConfig:
    gpu: str
    eval_timeout_s: int


@dataclass(frozen=True)
class PathsConfig:
    db: Path
    traces: Path


@dataclass(frozen=True)
class Config:
    models: ModelsConfig
    budget: BudgetConfig
    modal: ModalConfig
    paths: PathsConfig


def _model(raw: dict[str, Any]) -> ModelConfig:
    effort = raw["reasoning_effort"]
    if effort not in REASONING_EFFORTS:
        raise ValueError(f"reasoning_effort must be one of {REASONING_EFFORTS}, got {effort!r}")
    return ModelConfig(name=raw["name"], reasoning_effort=effort)


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> Config:
    """Parse config.toml. Missing keys raise KeyError; bad values raise ValueError."""
    path = Path(path).resolve()
    with path.open("rb") as f:
        raw = tomllib.load(f)

    models = raw["models"]
    if models["setup"] not in MODEL_SETUPS:
        raise ValueError(f"models.setup must be one of {MODEL_SETUPS}, got {models['setup']!r}")

    base = path.parent
    return Config(
        models=ModelsConfig(
            base_url=models["base_url"],
            setup=models["setup"],
            max_retries=int(models["max_retries"]),
            timeout_s=float(models["timeout_s"]),
            planner=_model(models["planner"]),
            fixer=_model(models["fixer"]),
        ),
        budget=BudgetConfig(
            max_attempts=int(raw["budget"]["max_attempts"]),
            max_usd_per_task=float(raw["budget"]["max_usd_per_task"]),
        ),
        modal=ModalConfig(
            gpu=raw["modal"]["gpu"],
            eval_timeout_s=int(raw["modal"]["eval_timeout_s"]),
        ),
        paths=PathsConfig(
            db=base / raw["paths"]["db"],
            traces=base / raw["paths"]["traces"],
        ),
    )
