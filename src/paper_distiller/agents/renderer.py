"""ConsoleRenderer — receives status events, exposes a rich Table snapshot.

Live rendering (rich.live.Live) is wired up in the CLI layer; this module
only owns state + table construction.

v1.6.2 visual upgrades:
  - Spinner animation on RUNNING rows (so users see "still alive" pulse).
  - Per-agent `activity` field — agents can call ctx.on_status(activity="...")
    to expose what they're currently doing inside a long-running step.
  - Color-coded status (running=cyan, done=green, failed=red, skipped=dim).
"""

from __future__ import annotations

import time
from typing import Any

from rich.table import Table
from rich.text import Text

from .base import Status


# Braille spinner frames — works in most terminals
_SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
# ASCII fallback for terminals without Braille support
_SPINNER_FRAMES_ASCII = "|/-\\"


def _fancy_spinners_supported() -> bool:
    import sys
    enc = (getattr(sys.stdout, "encoding", "") or "").lower()
    return "utf" in enc


_FRAMES = _SPINNER_FRAMES if _fancy_spinners_supported() else _SPINNER_FRAMES_ASCII


_STATUS_STYLE = {
    Status.QUEUED: "dim",
    Status.RUNNING: "bold cyan",
    Status.DONE: "green",
    Status.FAILED: "bold red",
    Status.SKIPPED: "dim yellow",
}


class ConsoleRenderer:
    def __init__(self, title: str = ""):
        self.title = title
        self._rows: dict[str, dict[str, Any]] = {}

    def on_status(self, name: str, status: Status | None = None, **kw) -> None:
        """Status event callback. Updates internal row state.

        `status` is now optional — agents can also call this purely to update
        the activity sub-text without changing status (e.g. mid-LLM-call
        progress reports from candidate-ranker / paper-processor).
        """
        row = self._rows.setdefault(name, {
            "status": Status.QUEUED,
            "started_at": None,
            "elapsed": None,
            "error": None,
            "activity": None,
        })
        if status is not None:
            row["status"] = status
            if status == Status.RUNNING and row["started_at"] is None:
                row["started_at"] = time.monotonic()
            elif status in (Status.DONE, Status.FAILED, Status.SKIPPED):
                if row["started_at"] is not None:
                    row["elapsed"] = time.monotonic() - row["started_at"]
                # Clear transient activity when terminal state reached
                row["activity"] = None
                if "error" in kw:
                    row["error"] = kw["error"]
        if "activity" in kw:
            row["activity"] = kw["activity"]

    def snapshot(self) -> dict[str, dict[str, Any]]:
        """Return a copy of the current row state. For tests."""
        return {k: dict(v) for k, v in self._rows.items()}

    def _spinner_frame(self) -> str:
        """Pick a spinner frame based on wall-clock — 10Hz cycle."""
        idx = int(time.monotonic() * 10) % len(_FRAMES)
        return _FRAMES[idx]

    def build_table(self) -> Table:
        """Return a rich Table reflecting current state."""
        table = Table(title=self.title or None, show_header=True)
        table.add_column("Agent")
        table.add_column("Status")
        table.add_column("Elapsed")
        table.add_column("Activity")

        spinner = self._spinner_frame()
        for name, row in self._rows.items():
            status = row["status"]
            style = _STATUS_STYLE.get(status, "")

            # Prefix RUNNING rows with spinner so users see live motion
            if status == Status.RUNNING:
                status_str = f"{spinner} {status.value}"
            else:
                status_str = status.value
            status_cell = Text(status_str, style=style)

            if row["elapsed"] is not None:
                elapsed_str = f"{row['elapsed']:.1f}s"
            elif row["started_at"] is not None:
                elapsed_str = f"{(time.monotonic() - row['started_at']):.1f}s"
            else:
                elapsed_str = "—"

            activity = row.get("activity") or ""
            activity_cell = Text(activity, style="dim italic") if activity else Text("")

            table.add_row(name, status_cell, elapsed_str, activity_cell)
        return table
