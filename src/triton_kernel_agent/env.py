"""Load secrets from .env. They live only in the local orchestrator, never in config.toml."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from triton_kernel_agent.config import REPO_ROOT

DEFAULT_ENV_PATH = REPO_ROOT / ".env"


@dataclass(frozen=True)
class Secrets:
    meta_api_key: str = field(repr=False)


def load_secrets(env_path: Path = DEFAULT_ENV_PATH) -> Secrets:
    """Read secrets from env_path into the process env; variables already set take priority."""
    load_dotenv(env_path, override=False)
    key = os.environ.get("META_API_KEY")
    if not key:
        raise RuntimeError("META_API_KEY is not set. Copy .env.example to .env and fill it in.")
    return Secrets(meta_api_key=key)
