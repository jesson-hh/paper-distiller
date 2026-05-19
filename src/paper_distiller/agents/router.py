"""IntentRouter — single JSON-out LLM call mapping natural language to a slash command."""

from __future__ import annotations

import json
from pathlib import Path


class RoutingError(RuntimeError):
    pass


_PROMPT_FILE = Path(__file__).parent / "prompts" / "route.md"
_VALID_COMMANDS = {"distill", "ask", "resume", "show"}
_REQUIRED_KEYS = {"command", "params", "missing_params", "confidence"}


class IntentRouter:
    name = "intent-router"
    deps: list[str] = []

    def __init__(self, llm):
        self.llm = llm

    def classify(self, user_input: str) -> dict:
        prompt = _PROMPT_FILE.read_text(encoding="utf-8").format(user_input=user_input)
        messages = [{"role": "user", "content": prompt}]
        for attempt in (1, 2):
            raw = self.llm.complete(messages, temperature=0.2, response_format="json")
            try:
                parsed = json.loads(raw)
                missing = _REQUIRED_KEYS - set(parsed.keys())
                if missing:
                    raise ValueError(f"missing keys: {missing}")
                if parsed["command"] not in _VALID_COMMANDS:
                    raise ValueError(f"unknown command: {parsed['command']!r}")
                return parsed
            except (json.JSONDecodeError, ValueError) as e:
                if attempt == 2:
                    if "unknown command" in str(e):
                        raise RoutingError(f"unknown command in router output: {raw[:200]}")
                    raise RoutingError(f"intent router returned malformed JSON: {raw[:200]}")
                continue
        raise RoutingError("unreachable")
