"""Tests for chat.slash_commands — registry + 7 handlers."""

from __future__ import annotations

from unittest.mock import MagicMock


def _fake_loop(vault_path, llm=None):
    """Build a stand-in AgentLoop for slash handlers — only the attrs they read."""
    loop = MagicMock()
    loop.vault_path = str(vault_path)
    loop.llm = llm or MagicMock(
        model="qwen-plus", total_tokens_in=0, total_tokens_out=0,
        estimated_cost_cny=0.0,
    )
    loop.messages = [{"role": "system", "content": "sys"}]
    loop.auto_mode = False
    loop.console = MagicMock()
    return loop


def test_parse_command_recognizes_known():
    from paper_distiller.chat.slash_commands import parse_slash

    assert parse_slash("/help") == ("help", [])
    assert parse_slash("/show foo articles") == ("show", ["foo", "articles"])
    assert parse_slash("/cost   ") == ("cost", [])


def test_parse_command_returns_none_for_non_slash():
    from paper_distiller.chat.slash_commands import parse_slash

    assert parse_slash("hello") is None
    assert parse_slash("") is None
    assert parse_slash("  /help") is None


def test_dispatch_unknown_command_returns_error(tmp_path):
    from paper_distiller.chat.slash_commands import dispatch_slash

    loop = _fake_loop(tmp_path)
    result = dispatch_slash("nope", [], loop)
    assert result.startswith("[unknown command")


def test_clear_resets_history(tmp_path):
    from paper_distiller.chat.slash_commands import dispatch_slash

    loop = _fake_loop(tmp_path)
    loop.messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
    ]
    result = dispatch_slash("clear", [], loop)
    assert loop.messages == [{"role": "system", "content": "sys"}]
    assert "cleared" in result.lower()


def test_cost_shows_session_totals(tmp_path):
    from paper_distiller.chat.slash_commands import dispatch_slash

    llm = MagicMock(
        model="qwen-plus",
        total_tokens_in=12345,
        total_tokens_out=678,
        estimated_cost_cny=0.0123,
    )
    loop = _fake_loop(tmp_path, llm=llm)
    result = dispatch_slash("cost", [], loop)
    assert "12,345" in result and "678" in result
    assert "0.0123" in result


def test_help_lists_all_commands(tmp_path):
    from paper_distiller.chat.slash_commands import dispatch_slash

    loop = _fake_loop(tmp_path)
    result = dispatch_slash("help", [], loop)
    for name in ("clear", "cost", "help", "show", "history", "exit", "auto"):
        assert f"/{name}" in result


def test_show_missing_slug_returns_usage(tmp_path):
    from paper_distiller.chat.slash_commands import dispatch_slash

    loop = _fake_loop(tmp_path)
    result = dispatch_slash("show", [], loop)
    assert "usage" in result.lower()


def test_show_reads_existing_entry(tmp_path):
    from paper_distiller.chat.slash_commands import dispatch_slash
    from paper_distiller.vault.store import VaultStore

    VaultStore(tmp_path).save_entry(
        title="T", category="articles", body="hello body",
        slug="t",
    )
    loop = _fake_loop(tmp_path)
    result = dispatch_slash("show", ["t"], loop)
    assert "hello body" in result


def test_history_lists_recent_entries(tmp_path):
    from paper_distiller.chat.slash_commands import dispatch_slash
    from paper_distiller.vault.store import VaultStore

    vault = VaultStore(tmp_path)
    for i in range(3):
        vault.save_entry(
            title=f"P{i}", category="articles", body="x", slug=f"p{i}",
        )
    loop = _fake_loop(tmp_path)
    result = dispatch_slash("history", [], loop)
    for i in range(3):
        assert f"p{i}" in result


def test_exit_signals_quit(tmp_path):
    from paper_distiller.chat.slash_commands import dispatch_slash, EXIT_SIGNAL

    loop = _fake_loop(tmp_path)
    result = dispatch_slash("exit", [], loop)
    assert result == EXIT_SIGNAL


def test_auto_toggles_loop_flag(tmp_path):
    from paper_distiller.chat.slash_commands import dispatch_slash

    loop = _fake_loop(tmp_path)
    loop.auto_mode = False
    dispatch_slash("auto", [], loop)
    assert loop.auto_mode is True
    dispatch_slash("auto", [], loop)
    assert loop.auto_mode is False
