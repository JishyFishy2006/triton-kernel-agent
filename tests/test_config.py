from pathlib import Path

import pytest

from triton_kernel_agent.config import REPO_ROOT, Config, load_config


def test_repo_config_loads():
    cfg = load_config()
    assert isinstance(cfg, Config)
    assert cfg.models.base_url == "https://api.meta.ai/v1"
    assert cfg.models.max_retries == 5
    assert cfg.models.planner.name == "muse-wonder-1.0"
    assert cfg.models.planner.reasoning_effort == "high"
    assert cfg.models.fixer.name == "muse-spark-1.3"
    assert cfg.models.fixer.reasoning_effort == "low"
    assert cfg.budget.max_attempts == 10
    assert cfg.modal.gpu == "H100!"


def test_paths_resolve_relative_to_config_file(tmp_path: Path):
    src = (REPO_ROOT / "config.toml").read_text()
    path = tmp_path / "config.toml"
    path.write_text(src)
    cfg = load_config(path)
    assert cfg.paths.db.is_absolute()
    assert cfg.paths.db.parent == tmp_path / "data"


def test_missing_key_raises(tmp_path: Path):
    path = tmp_path / "config.toml"
    path.write_text('[models]\nbase_url = "x"\n')
    with pytest.raises(KeyError):
        load_config(path)


def test_bad_reasoning_effort_raises(tmp_path: Path):
    src = (REPO_ROOT / "config.toml").read_text().replace('"high"', '"extreme"')
    path = tmp_path / "config.toml"
    path.write_text(src)
    with pytest.raises(ValueError, match="reasoning_effort"):
        load_config(path)
