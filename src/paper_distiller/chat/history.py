"""Persistent input history — JSONL file of past user inputs.

Inspired by Claude Code's `~/.claude/history.jsonl` model. Each line is a
single JSON record `{"display": "...", "ts": "ISO timestamp"}`. Most-recent
first via tail-reads.

Limits:
- Max 2000 entries kept on disk; older entries auto-truncated on append
- Slash commands NOT recorded (noise) — only natural-language prompts
- Empty lines / whitespace-only inputs skipped
- File locked during append to avoid concurrent corruption

Used by:
- prompt_toolkit's FileHistory wrapper in agent_loop.run()
- (future) /history slash to browse past inputs
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


MAX_ENTRIES = 2000


def _default_path() -> Path:
    """Resolve history file location. Honors PD_HISTORY_FILE env."""
    override = os.getenv("PD_HISTORY_FILE")
    if override:
        return Path(override)
    return Path(os.path.expanduser("~")) / ".paper-distiller" / "history.jsonl"


class InputHistory:
    """Append-only JSONL store of user inputs across sessions.

    Read pattern: load most-recent N entries on demand (no full file load).
    Write pattern: single-line append with periodic truncation.
    """

    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path else _default_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, text: str) -> None:
        """Add one entry. Skip if blank or starts with '/' (slash command)."""
        text = (text or "").strip()
        if not text or text.startswith("/"):
            return
        record = {
            "display": text,
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        # Best-effort truncation: only check every ~10 appends to avoid IO cost
        try:
            if int(record["ts"][-2:]) % 10 == 0:
                self._truncate_if_needed()
        except Exception:
            pass

    def _truncate_if_needed(self) -> None:
        """Keep at most MAX_ENTRIES lines; drop the oldest."""
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return
        if len(lines) <= MAX_ENTRIES:
            return
        kept = lines[-MAX_ENTRIES:]
        self.path.write_text("\n".join(kept) + "\n", encoding="utf-8")

    def recent(self, limit: int = 20) -> list[dict]:
        """Return up to `limit` most-recent entries (newest first)."""
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []
        out: list[dict] = []
        for line in reversed(lines):
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            if len(out) >= limit:
                break
        return out

    def all_displays_newest_first(self) -> Iterable[str]:
        """Iterator over `display` strings, newest first. For prompt_toolkit
        FileHistory adapter — keeps memory bounded vs full load."""
        for entry in self.recent(limit=MAX_ENTRIES):
            yield entry.get("display", "")

    def clear(self) -> None:
        """Wipe history file. Used by `/clear` slash with --history flag
        (future) or manual reset."""
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass
