"""Tests for slash-command parsing."""
import pytest

from paper_distiller.chat.repl.commands import parse_slash, SlashError, KNOWN_COMMANDS


def test_parse_simple_command_no_args():
    parsed = parse_slash("/vault")
    assert parsed.name == "vault"
    assert parsed.args == []


def test_parse_command_with_args():
    parsed = parse_slash("/distill diffusion models --n 3")
    assert parsed.name == "distill"
    assert parsed.args == ["diffusion", "models", "--n", "3"]


def test_parse_command_with_quoted_arg():
    parsed = parse_slash('/ask "why diffusion models?"')
    assert parsed.name == "ask"
    assert parsed.args == ["why diffusion models?"]


def test_parse_unknown_command_raises():
    with pytest.raises(SlashError, match="unknown"):
        parse_slash("/nosuchcommand")


def test_parse_non_slash_input_raises():
    with pytest.raises(SlashError, match="not a slash command"):
        parse_slash("hello world")


def test_known_commands_includes_core():
    assert "distill" in KNOWN_COMMANDS
    assert "ask" in KNOWN_COMMANDS
    assert "resume" in KNOWN_COMMANDS
    assert "vault" in KNOWN_COMMANDS
    assert "sessions" in KNOWN_COMMANDS
    assert "provider" in KNOWN_COMMANDS
    assert "agents" in KNOWN_COMMANDS
    assert "show" in KNOWN_COMMANDS
    assert "help" in KNOWN_COMMANDS
    assert "quit" in KNOWN_COMMANDS
