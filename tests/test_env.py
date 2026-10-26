from pathlib import Path

import pytest

from triton_kernel_agent.env import Secrets, load_secrets


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: pytest.MonkeyPatch):
    # setenv first so monkeypatch records the original state and undoes what
    # load_dotenv writes into os.environ during the test.
    monkeypatch.setenv("META_API_KEY", "")
    monkeypatch.delenv("META_API_KEY")


def test_loads_key_from_env_file(tmp_path: Path):
    env_file = tmp_path / ".env"
    env_file.write_text("META_API_KEY=from-file\n")
    assert load_secrets(env_file) == Secrets(meta_api_key="from-file")


def test_real_env_wins_over_env_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    env_file = tmp_path / ".env"
    env_file.write_text("META_API_KEY=from-file\n")
    monkeypatch.setenv("META_API_KEY", "from-shell")
    assert load_secrets(env_file).meta_api_key == "from-shell"


def test_missing_key_raises_with_hint(tmp_path: Path):
    with pytest.raises(RuntimeError, match=r"META_API_KEY.*\.env\.example"):
        load_secrets(tmp_path / "does-not-exist.env")


def test_repr_hides_secret(tmp_path: Path):
    env_file = tmp_path / ".env"
    env_file.write_text("META_API_KEY=sk-very-secret\n")
    assert "sk-very-secret" not in repr(load_secrets(env_file))
