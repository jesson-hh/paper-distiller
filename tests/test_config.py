from pathlib import Path

import pytest

from paper_distiller.config import Config, load_config


def test_config_required_fields(monkeypatch, tmp_path: Path):
    vault = tmp_path / "v"; vault.mkdir()
    monkeypatch.setenv("PD_API_KEY", "sk-test")
    monkeypatch.setenv("PD_BASE_URL", "https://x/v1")
    monkeypatch.setenv("PD_MODEL", "qwen-plus")

    cfg = load_config(vault_path=vault, topic="diffusion", n=3)
    assert cfg.vault_path == vault
    assert cfg.topic == "diffusion"
    assert cfg.top_n == 3
    assert cfg.api_key == "sk-test"
    assert cfg.model == "qwen-plus"
    assert cfg.pool == 30  # default
    assert cfg.force is False


def test_config_missing_api_key_raises(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("PD_API_KEY", raising=False)
    with pytest.raises(ValueError, match="PD_API_KEY"):
        load_config(vault_path=tmp_path, topic="x")


def test_config_either_topic_or_author(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("PD_API_KEY", "sk-test")
    monkeypatch.setenv("PD_BASE_URL", "https://x/v1")
    monkeypatch.setenv("PD_MODEL", "m")

    with pytest.raises(ValueError, match="topic or author"):
        load_config(vault_path=tmp_path, topic=None, author=None)


def test_config_dry_run_flag(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("PD_API_KEY", "sk-test")
    monkeypatch.setenv("PD_BASE_URL", "https://x/v1")
    monkeypatch.setenv("PD_MODEL", "m")
    cfg = load_config(vault_path=tmp_path, topic="x", dry_run=True)
    assert cfg.dry_run is True


def test_config_source_field(monkeypatch, tmp_path: Path):
    """Config exposes a `source` field defaulting to 'both', accepts arxiv/ss/both, rejects others."""
    monkeypatch.setenv("PD_API_KEY", "sk-test")
    monkeypatch.setenv("PD_BASE_URL", "https://x/v1")
    monkeypatch.setenv("PD_MODEL", "m")

    # Default
    cfg = load_config(vault_path=tmp_path, topic="x")
    assert cfg.source == "both"

    # Explicit valid values
    for val in ("arxiv", "ss", "both"):
        cfg = load_config(vault_path=tmp_path, topic="x", source=val)
        assert cfg.source == val

    # Invalid value raises
    with pytest.raises(ValueError, match="source"):
        load_config(vault_path=tmp_path, topic="x", source="nonsense")

    # ss_api_key reads from env
    monkeypatch.setenv("PD_SS_API_KEY", "ss-key-123")
    cfg = load_config(vault_path=tmp_path, topic="x")
    assert cfg.ss_api_key == "ss-key-123"
