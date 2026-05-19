"""Read-only REPL helper handlers.

These do NOT call the LLM. They inspect vault / env / agent registry and return
strings that the REPL prints. Action commands (/distill, /ask, /resume) live in
loop.py because they need access to the argparse handlers.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


def handle_vault(vault_path: Path) -> str:
    cats = ["articles", "surveys", "techniques", "directions", "open-problems", "authors"]
    lines = [f"Vault: {vault_path}"]
    for cat in cats:
        folder = vault_path / cat
        count = len(list(folder.glob("*.md"))) if folder.exists() else 0
        lines.append(f"  {cat}: {count}")
    return "\n".join(lines)


def handle_sessions(vault_path: Path) -> str:
    sessions_dir = vault_path / ".paper_distiller" / "qa-sessions"
    if not sessions_dir.exists():
        return "no sessions found."
    entries = sorted(sessions_dir.iterdir(), reverse=True)  # newest first
    if not entries:
        return "no sessions found."
    lines = ["QA sessions (newest first):"]
    found_any = False
    for entry in entries:
        state_path = entry / "state.json"
        if not state_path.exists():
            continue
        try:
            data = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        found_any = True
        sid = data.get("session_id", entry.name)
        stop = data.get("stop_reason", "?")
        rounds = data.get("rounds_completed", 0)
        cost = data.get("cost_cny", 0.0)
        question = (data.get("question", "") or "")[:60]
        is_done = "done" if data.get("is_done") else "open"
        lines.append(f"  {sid}  {is_done}  {stop}  ({rounds} rounds, CNY {cost:.2f})")
        if question:
            lines.append(f"    Q: {question}")
    if not found_any:
        return "no sessions found."
    return "\n".join(lines)


def handle_provider() -> str:
    base_url = os.getenv("PD_BASE_URL", "(not set)")
    model = os.getenv("PD_MODEL", "(not set)")
    provider = os.getenv("PD_PROVIDER_NAME", "unspecified")
    key = os.getenv("PD_API_KEY", "")
    key_status = "(set)" if key else "(not set)"
    return (
        f"Provider: {provider}\n"
        f"Base URL: {base_url}\n"
        f"Model:    {model}\n"
        f"API key:  {key_status}"
    )


def handle_agents() -> str:
    return (
        "Registered agents (v1.0):\n"
        "  Source:   arxiv-searcher, ss-searcher\n"
        "  Curation: candidate-merger, candidate-dedup, candidate-ranker\n"
        "  Process:  paper-processor (fanout)\n"
        "  Persist:  vault-writer, survey-composer\n"
        "  QA:       progress-reflector, answer-synthesizer\n"
        "  REPL:     intent-router"
    )


def handle_show(vault_path: Path, slug: str) -> str:
    for cat in ("articles", "surveys"):
        path = vault_path / cat / f"{slug}.md"
        if path.exists():
            return path.read_text(encoding="utf-8")
    return f"slug {slug!r} not found in articles/ or surveys/."


def handle_help() -> str:
    return (
        "Slash commands:\n"
        "  /distill <topic> [--n N]        — single-pass: search and distill N papers\n"
        "  /ask <question>                 — multi-round QA loop\n"
        "  /resume <session-id>            — continue a paused QA session\n"
        "  /sessions                       — list past QA sessions\n"
        "  /vault                          — show vault stats\n"
        "  /provider                       — show LLM config\n"
        "  /agents                         — list registered agents\n"
        "  /show <slug>                    — display an article/survey from vault\n"
        "  /help                           — this list\n"
        "  /quit                           — exit REPL\n"
        "\n"
        "Natural language: type anything else (e.g. '帮我研究下扩散'),\n"
        "the intent-router will propose a command and confirm with you."
    )
