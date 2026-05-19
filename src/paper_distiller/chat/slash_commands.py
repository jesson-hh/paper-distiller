"""Slash command registry + handlers.

Slash commands bypass the LLM entirely — they're shortcuts for high-frequency
ops (clear history, view cost, peek at vault) and toggle flags (`/auto`).

Each handler takes (args: list[str], loop: AgentLoop) and returns a string to
display to the user. The string EXIT_SIGNAL is special: when returned, the
AgentLoop's run() exits cleanly.
"""

from __future__ import annotations

from typing import Callable


EXIT_SIGNAL = "__pd_exit__"


def parse_slash(line: str) -> tuple[str, list[str]] | None:
    """Parse `/cmd arg1 arg2` → (cmd, [arg1, arg2]). None if not a slash command."""
    if not line.startswith("/"):
        return None
    parts = line[1:].strip().split()
    if not parts:
        return None
    return parts[0].lower(), parts[1:]


def _cmd_clear(args, loop) -> str:
    sys_msg = next(
        (m for m in loop.messages if m.get("role") == "system"), None
    )
    loop.messages.clear()
    if sys_msg is not None:
        loop.messages.append(sys_msg)
    return "(conversation history cleared)"


def _cmd_cost(args, loop) -> str:
    llm = loop.llm
    return (
        f"model: {llm.model}\n"
        f"tokens in:  {llm.total_tokens_in:,}\n"
        f"tokens out: {llm.total_tokens_out:,}\n"
        f"estimated cost: ¥{llm.estimated_cost_cny:.4f}"
    )


def _cmd_help(args, loop) -> str:
    lines = [
        "slash commands (bypass LLM):",
        "  /clear            reset conversation history",
        "  /cost             show session token + ¥ totals",
        "  /help             this message",
        "  /show <slug> [cat]  read a vault entry (default category: articles)",
        "  /history          list 10 most recent vault entries",
        "  /exit, /quit      leave the REPL",
        "  /auto             toggle auto-mode (skip plan-mode previews)",
    ]
    return "\n".join(lines)


def _cmd_show(args, loop) -> str:
    if not args:
        return "usage: /show <slug> [category]"
    slug = args[0]
    category = args[1] if len(args) > 1 else "articles"
    from ..vault.store import VaultStore
    try:
        vault = VaultStore(loop.vault_path)
        entry = vault.read_entry(category, slug)
    except ValueError as e:
        return f"error: {e}"
    if entry is None:
        return f"(no entry {category}/{slug})"
    return f"# {entry.title}\n\n{entry.body}"


def _cmd_history(args, loop) -> str:
    from ..vault.store import VaultStore
    vault = VaultStore(loop.vault_path)
    entries = vault.list_entries()[:10]
    if not entries:
        return "(vault is empty)"
    lines = ["recent vault entries (10 newest):"]
    for e in entries:
        updated = (e.get("updated") or "")[:10]
        lines.append(
            f"  [{e.get('category', '?'):>14}] {e.get('slug', '?')}"
            f"  · {e.get('title', '?')[:50]}  · {updated}"
        )
    return "\n".join(lines)


def _cmd_exit(args, loop) -> str:
    return EXIT_SIGNAL


def _cmd_auto(args, loop) -> str:
    loop.auto_mode = not getattr(loop, "auto_mode", False)
    state = "on" if loop.auto_mode else "off"
    return f"auto-mode: {state} (skips plan-mode previews when on)"


HANDLERS: dict[str, Callable] = {
    "clear": _cmd_clear,
    "cost": _cmd_cost,
    "help": _cmd_help,
    "show": _cmd_show,
    "history": _cmd_history,
    "exit": _cmd_exit,
    "quit": _cmd_exit,
    "auto": _cmd_auto,
}


def dispatch_slash(name: str, args: list, loop) -> str:
    """Run a parsed slash command. Returns a display string or EXIT_SIGNAL."""
    fn = HANDLERS.get(name.lower())
    if fn is None:
        return f"[unknown command /{name}; type /help for the list]"
    try:
        return fn(args, loop)
    except Exception as e:
        return f"[/{name} failed: {type(e).__name__}: {e}]"
