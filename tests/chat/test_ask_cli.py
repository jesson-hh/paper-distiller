"""Tests for paper-distiller-chat 'ask' subcommand."""
from unittest.mock import MagicMock

import pytest


def test_ask_cli_parses_args(monkeypatch):
    from paper_distiller.chat.cli import build_parser
    p = build_parser()
    args = p.parse_args([
        "ask", "--vault", "/tmp/v", "--question", "why?",
        "--max-rounds", "3", "--per-round", "2",
    ])
    assert args.subcommand == "ask"
    assert args.vault == "/tmp/v"
    assert args.question == "why?"
    assert args.max_rounds == 3
    assert args.per_round == 2


def test_ask_cli_dispatches_to_run_qa_loop(mocker, tmp_path, monkeypatch):
    monkeypatch.setenv("PD_API_KEY", "sk-test")
    monkeypatch.setenv("PD_BASE_URL", "https://x/v1")
    monkeypatch.setenv("PD_MODEL", "qwen-plus")
    fake_run = mocker.patch("paper_distiller.chat.cli.run_qa_loop")
    fake_run.return_value = {
        "session_id": "sid-1", "stop_reason": "llm_done",
        "rounds_completed": 2, "articles_distilled_count": 4,
        "cost_cny": 0.5, "tokens_in_total": 1000, "tokens_out_total": 500,
    }
    from paper_distiller.chat.cli import main
    rc = main([
        "ask", "--vault", str(tmp_path), "--question", "why?",
        "--max-rounds", "3",
    ])
    assert rc == 0
    fake_run.assert_called_once()
    cfg = fake_run.call_args[0][0]
    assert cfg.qa_question == "why?"
    assert cfg.qa_max_rounds == 3
