"""Slash-command parsing + known-command registry."""

from __future__ import annotations

import shlex
from dataclasses import dataclass


class SlashError(ValueError):
    pass


KNOWN_COMMANDS = {
    "distill", "ask", "resume",       # action commands (delegate to one-shot handlers)
    "vault", "sessions", "provider",  # read-only helpers
    "agents", "show", "help", "quit",
}


@dataclass
class ParsedSlash:
    name: str
    args: list[str]


def parse_slash(line: str) -> ParsedSlash:
    """Parse '/cmd arg1 "arg 2" --flag x' → ParsedSlash(name='cmd', args=['arg1','arg 2','--flag','x'])"""
    s = line.strip()
    if not s.startswith("/"):
        raise SlashError(f"not a slash command: {line!r}")
    body = s[1:].strip()
    if not body:
        raise SlashError("empty slash command")
    try:
        tokens = shlex.split(body)
    except ValueError as e:
        raise SlashError(f"could not parse slash command: {e}")
    if not tokens:
        raise SlashError("empty slash command after parse")
    name, *args = tokens
    if name not in KNOWN_COMMANDS:
        raise SlashError(f"unknown slash command: /{name}")
    return ParsedSlash(name=name, args=args)
