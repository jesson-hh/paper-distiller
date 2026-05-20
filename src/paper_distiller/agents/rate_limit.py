"""Process-global rate limiter + post-429 cooldown for search sources.

Three module-level singletons (ARXIV_LIMITER, SS_LIMITER, OPENALEX_LIMITER)
enforce:
  1. Minimum spacing between any two requests to the same source.
  2. A cooldown window after a transient failure — subsequent acquire() calls
     return False until the window elapses, so the searcher skips the source
     entirely instead of retrying into the wall.

Why module-level singletons rather than per-Context: cooldown state should
survive a single tool_search call (which spins up a fresh Orchestrator),
otherwise the next LLM-driven retry would race straight back into the 429.
For multi-process scenarios this is process-local; out of scope for now.

Spacing values chosen from public docs (May 2026):
  - arxiv: ≥3s between requests (per their API guidelines); we use 3.5s with
    jitter buffer.
  - SS: free tier ~100 req / 5min = 1 per ~3s shared. Be polite: 18s spacing.
    Recommend PD_SS_API_KEY for production — the limiter spacing tightens
    automatically when an API key is configured.
  - OpenAlex: 100 req/s common pool. We're light users: 1s spacing.
"""

from __future__ import annotations

import asyncio
import os
import random
import time
from typing import Optional


class SourceLimiter:
    """Async-safe rate limiter + cooldown for one search source."""

    def __init__(
        self,
        name: str,
        min_spacing_sec: float,
        cooldown_sec: float = 90.0,
    ):
        self.name = name
        self.min_spacing = min_spacing_sec
        self.cooldown_sec = cooldown_sec
        self._last_call_at: float = 0.0
        self._cooldown_until: float = 0.0
        self._lock = asyncio.Lock()

    def is_cooling_down(self) -> bool:
        """Quick non-blocking check — useful for tool_search short-circuit."""
        return time.monotonic() < self._cooldown_until

    def seconds_until_ready(self) -> float:
        """How many more seconds before the cooldown window ends. 0 if free."""
        return max(0.0, self._cooldown_until - time.monotonic())

    async def acquire(self) -> bool:
        """Block until safe to issue next request. False if in cooldown.

        Adds 0-0.5s jitter on top of min_spacing so concurrent searchers don't
        re-synchronize their bursts.
        """
        async with self._lock:
            now = time.monotonic()
            if now < self._cooldown_until:
                return False
            delta = now - self._last_call_at
            wait = self.min_spacing - delta
            if wait > 0:
                # Jitter: ±10% spread to avoid client-side burst sync.
                wait += random.uniform(0, 0.5)
                await asyncio.sleep(wait)
            self._last_call_at = time.monotonic()
            return True

    def mark_429(self, duration_sec: Optional[float] = None) -> None:
        """Engage cooldown for `duration_sec` (default self.cooldown_sec)."""
        dt = duration_sec if duration_sec is not None else self.cooldown_sec
        self._cooldown_until = time.monotonic() + dt

    def reset(self) -> None:
        """Test helper — clear all state."""
        self._last_call_at = 0.0
        self._cooldown_until = 0.0


def _ss_spacing() -> float:
    """SS spacing tightens when an API key is configured (much higher quota)."""
    if os.getenv("PD_SS_API_KEY"):
        return 0.5  # API key → ~100 rps, conservative 2/s here
    return 18.0


ARXIV_LIMITER = SourceLimiter(
    name="arxiv", min_spacing_sec=3.5, cooldown_sec=90.0
)
SS_LIMITER = SourceLimiter(
    name="ss", min_spacing_sec=_ss_spacing(), cooldown_sec=120.0
)
OPENALEX_LIMITER = SourceLimiter(
    name="openalex", min_spacing_sec=1.0, cooldown_sec=60.0
)
