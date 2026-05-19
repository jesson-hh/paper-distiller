"""Tests for paper-distiller-chat (no subcommand → REPL)."""
import pytest


def test_chat_no_subcommand_launches_repl(mocker, tmp_path, monkeypatch):
    monkeypatch.setenv("PD_API_KEY", "sk-test")
    monkeypatch.setenv("PD_BASE_URL", "https://x/v1")
    monkeypatch.setenv("PD_MODEL", "qwen-plus")
    fake_repl = mocker.patch("paper_distiller.chat.cli.REPL")
    fake_repl.return_value.run.return_value = 0
    from paper_distiller.chat.cli import main
    rc = main(["--vault", str(tmp_path)])
    assert rc == 0
    fake_repl.assert_called_once()


def test_chat_no_subcommand_no_vault_returns_error():
    from paper_distiller.chat.cli import main
    rc = main([])  # no --vault, no subcommand
    assert rc == 2  # error code
