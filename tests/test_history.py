"""Tests for chat.history — persistent input history."""

from __future__ import annotations

import json


def test_history_append_basic(tmp_path):
    from paper_distiller.chat.history import InputHistory
    h = InputHistory(tmp_path / "hist.jsonl")
    h.append("find some diffusion papers")
    recent = h.recent()
    assert len(recent) == 1
    assert recent[0]["display"] == "find some diffusion papers"


def test_history_skips_blank(tmp_path):
    from paper_distiller.chat.history import InputHistory
    h = InputHistory(tmp_path / "hist.jsonl")
    h.append("")
    h.append("   ")
    h.append("\n\n")
    assert h.recent() == []


def test_history_skips_slash_commands(tmp_path):
    """Slash commands should not pollute history — they're not the kind of
    thing you want ↑ to recall."""
    from paper_distiller.chat.history import InputHistory
    h = InputHistory(tmp_path / "hist.jsonl")
    h.append("/cost")
    h.append("/help")
    h.append("real query")
    recent = h.recent()
    assert len(recent) == 1
    assert recent[0]["display"] == "real query"


def test_history_recent_newest_first(tmp_path):
    from paper_distiller.chat.history import InputHistory
    h = InputHistory(tmp_path / "hist.jsonl")
    h.append("first")
    h.append("second")
    h.append("third")
    recent = h.recent()
    assert [r["display"] for r in recent] == ["third", "second", "first"]


def test_history_recent_respects_limit(tmp_path):
    from paper_distiller.chat.history import InputHistory
    h = InputHistory(tmp_path / "hist.jsonl")
    for i in range(10):
        h.append(f"q{i}")
    recent = h.recent(limit=3)
    assert len(recent) == 3
    assert recent[0]["display"] == "q9"


def test_history_handles_missing_file(tmp_path):
    from paper_distiller.chat.history import InputHistory
    h = InputHistory(tmp_path / "nonexistent.jsonl")
    assert h.recent() == []


def test_history_all_displays_newest_first(tmp_path):
    from paper_distiller.chat.history import InputHistory
    h = InputHistory(tmp_path / "hist.jsonl")
    h.append("a")
    h.append("b")
    h.append("c")
    displays = list(h.all_displays_newest_first())
    assert displays == ["c", "b", "a"]


def test_history_handles_corrupt_lines(tmp_path):
    """Malformed JSON lines should be skipped silently."""
    from paper_distiller.chat.history import InputHistory
    path = tmp_path / "hist.jsonl"
    path.write_text(
        '{"display": "good", "ts": "2026-05-21T00:00:00"}\n'
        'not valid json\n'
        '{"display": "also good", "ts": "2026-05-21T00:01:00"}\n',
        encoding="utf-8",
    )
    h = InputHistory(path)
    recent = h.recent()
    assert len(recent) == 2
    assert {r["display"] for r in recent} == {"good", "also good"}


def test_history_clear_removes_file(tmp_path):
    from paper_distiller.chat.history import InputHistory
    h = InputHistory(tmp_path / "hist.jsonl")
    h.append("x")
    h.clear()
    assert h.recent() == []
    # Clearing a non-existent file shouldn't raise
    h.clear()


def test_history_env_override(monkeypatch, tmp_path):
    """PD_HISTORY_FILE env should override default path."""
    monkeypatch.setenv("PD_HISTORY_FILE", str(tmp_path / "custom.jsonl"))
    from paper_distiller.chat.history import InputHistory
    h = InputHistory()
    assert h.path == tmp_path / "custom.jsonl"
