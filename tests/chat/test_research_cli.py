"""Tests for paper-distiller-chat 'research' subcommand."""
import pytest


def test_research_cli_parses_args():
    from paper_distiller.chat.cli import build_parser
    p = build_parser()
    args = p.parse_args([
        "research", "--vault", "/tmp/v", "--question", "why?",
        "--max-papers", "20", "--duration", "2h",
    ])
    assert args.subcommand == "research"
    assert args.question == "why?"
    assert args.max_papers == 20
    assert args.duration == "2h"


def test_research_cli_dispatches_to_run_research_loop(mocker, tmp_path, monkeypatch):
    monkeypatch.setenv("PD_API_KEY", "sk-test")
    monkeypatch.setenv("PD_BASE_URL", "https://x/v1")
    monkeypatch.setenv("PD_MODEL", "qwen-plus")
    fake_run = mocker.patch("paper_distiller.chat.cli.run_research_loop")
    fake_run.return_value = {
        "session_id": "rs-1", "stop_reason": "all_themes_synthesized",
        "papers_distilled_count": 12, "themes_count": 3, "synthesis_count": 3,
        "final_report_slug": "research-x-20260519", "total_cost_cny": 15.5,
        "total_tokens_in": 30000, "total_tokens_out": 12000,
        "iterations_completed": 2,
    }
    from paper_distiller.chat.cli import main
    rc = main([
        "research", "--vault", str(tmp_path), "--question", "why?",
        "--max-papers", "10", "--duration", "1h",
    ])
    assert rc == 0
    fake_run.assert_called_once()
