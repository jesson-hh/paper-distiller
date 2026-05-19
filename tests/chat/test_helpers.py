"""Tests for read-only REPL helper handlers."""
import json
from pathlib import Path

import pytest

from paper_distiller.chat.repl.helpers import (
    handle_vault, handle_sessions, handle_provider, handle_agents,
    handle_show, handle_help,
)


def test_handle_vault_shows_counts(tmp_path):
    vault = tmp_path
    (vault / "articles").mkdir()
    (vault / "surveys").mkdir()
    (vault / "articles" / "a.md").write_text("---\ntitle: A\n---\n", encoding="utf-8")
    (vault / "articles" / "b.md").write_text("---\ntitle: B\n---\n", encoding="utf-8")
    (vault / "surveys" / "s.md").write_text("---\ntitle: S\n---\n", encoding="utf-8")
    out = handle_vault(vault)
    assert "articles: 2" in out
    assert "surveys: 1" in out


def test_handle_vault_empty(tmp_path):
    out = handle_vault(tmp_path)
    assert "articles: 0" in out


def test_handle_sessions_lists_state_json(tmp_path):
    vault = tmp_path
    sessions = vault / ".paper_distiller" / "qa-sessions"
    sessions.mkdir(parents=True)
    sid_dir = sessions / "20260519-1234-abc"
    sid_dir.mkdir()
    (sid_dir / "state.json").write_text(json.dumps({
        "session_id": "20260519-1234-abc", "question": "why?",
        "config_snapshot": {}, "started_at": "2026-05-19T12:34:00",
        "rounds_completed": 2, "articles_distilled": [], "articles_seen_ids": [],
        "history": [], "last_reflection": None, "cost_cny": 0.5,
        "tokens_in_total": 0, "tokens_out_total": 0,
        "is_done": True, "stop_reason": "llm_done",
    }), encoding="utf-8")
    out = handle_sessions(vault)
    assert "20260519-1234-abc" in out
    assert "llm_done" in out


def test_handle_sessions_no_sessions(tmp_path):
    out = handle_sessions(tmp_path)
    assert "no sessions" in out.lower()


def test_handle_provider_shows_config(monkeypatch):
    monkeypatch.setenv("PD_API_KEY", "sk-test-abc")
    monkeypatch.setenv("PD_BASE_URL", "https://x/v1")
    monkeypatch.setenv("PD_MODEL", "qwen-plus")
    out = handle_provider()
    assert "qwen-plus" in out
    assert "https://x/v1" in out
    # API key MUST be masked — not the raw value
    assert "sk-test-abc" not in out


def test_handle_agents_lists_registered():
    out = handle_agents()
    assert "arxiv-searcher" in out
    assert "paper-processor" in out
    assert "answer-synthesizer" in out


def test_handle_show_displays_article(tmp_path):
    vault = tmp_path
    (vault / "articles").mkdir()
    (vault / "articles" / "myslug.md").write_text(
        "---\ntitle: My Article\n---\n\n# My Article\n\nBody.",
        encoding="utf-8",
    )
    out = handle_show(vault, "myslug")
    assert "My Article" in out
    assert "Body." in out


def test_handle_show_not_found(tmp_path):
    out = handle_show(tmp_path, "no-such-slug")
    assert "not found" in out.lower()


def test_handle_help_lists_commands():
    out = handle_help()
    assert "/distill" in out
    assert "/ask" in out
    assert "/vault" in out
    assert "/quit" in out
