"""Tests for agents.rate_limit — SourceLimiter spacing + cooldown."""

from __future__ import annotations

import asyncio
import time

import pytest


@pytest.mark.asyncio
async def test_acquire_returns_true_when_idle():
    from paper_distiller.agents.rate_limit import SourceLimiter

    limiter = SourceLimiter("test", min_spacing_sec=0.01, cooldown_sec=1.0)
    assert await limiter.acquire() is True


@pytest.mark.asyncio
async def test_acquire_returns_false_when_cooling_down():
    from paper_distiller.agents.rate_limit import SourceLimiter

    limiter = SourceLimiter("test", min_spacing_sec=0.01, cooldown_sec=10.0)
    limiter.mark_429()
    assert await limiter.acquire() is False
    assert limiter.is_cooling_down() is True


@pytest.mark.asyncio
async def test_acquire_enforces_min_spacing():
    """Second acquire after a recent one must sleep before returning."""
    from paper_distiller.agents.rate_limit import SourceLimiter

    limiter = SourceLimiter("test", min_spacing_sec=0.1, cooldown_sec=1.0)
    t0 = time.monotonic()
    await limiter.acquire()  # immediate
    await limiter.acquire()  # must wait ~0.1s
    elapsed = time.monotonic() - t0
    assert elapsed >= 0.1


@pytest.mark.asyncio
async def test_cooldown_elapses():
    """After cooldown_sec passes, acquire works again."""
    from paper_distiller.agents.rate_limit import SourceLimiter

    limiter = SourceLimiter("test", min_spacing_sec=0.01, cooldown_sec=0.1)
    limiter.mark_429()
    assert limiter.is_cooling_down() is True
    await asyncio.sleep(0.15)
    assert limiter.is_cooling_down() is False
    assert await limiter.acquire() is True


def test_seconds_until_ready_zero_when_idle():
    from paper_distiller.agents.rate_limit import SourceLimiter

    limiter = SourceLimiter("test", min_spacing_sec=0.01, cooldown_sec=1.0)
    assert limiter.seconds_until_ready() == 0.0


def test_seconds_until_ready_positive_during_cooldown():
    from paper_distiller.agents.rate_limit import SourceLimiter

    limiter = SourceLimiter("test", min_spacing_sec=0.01, cooldown_sec=5.0)
    limiter.mark_429()
    assert 0 < limiter.seconds_until_ready() <= 5.0


def test_reset_clears_state():
    from paper_distiller.agents.rate_limit import SourceLimiter

    limiter = SourceLimiter("test", min_spacing_sec=0.01, cooldown_sec=10.0)
    limiter.mark_429()
    assert limiter.is_cooling_down() is True
    limiter.reset()
    assert limiter.is_cooling_down() is False


@pytest.mark.asyncio
async def test_mark_429_custom_duration():
    from paper_distiller.agents.rate_limit import SourceLimiter

    limiter = SourceLimiter("test", min_spacing_sec=0.01, cooldown_sec=100.0)
    limiter.mark_429(duration_sec=0.05)
    assert limiter.is_cooling_down() is True
    await asyncio.sleep(0.1)
    assert limiter.is_cooling_down() is False


def test_module_level_singletons_exist():
    from paper_distiller.agents.rate_limit import (
        ARXIV_LIMITER, SS_LIMITER, OPENALEX_LIMITER,
    )
    assert ARXIV_LIMITER.name == "arxiv"
    assert SS_LIMITER.name == "ss"
    assert OPENALEX_LIMITER.name == "openalex"
    # arxiv must respect the 3s guideline
    assert ARXIV_LIMITER.min_spacing >= 3.0


def test_ss_spacing_tightens_with_api_key(monkeypatch):
    """When PD_SS_API_KEY is set, SS_LIMITER spacing should be small."""
    from paper_distiller.agents.rate_limit import _ss_spacing

    monkeypatch.delenv("PD_SS_API_KEY", raising=False)
    no_key = _ss_spacing()
    monkeypatch.setenv("PD_SS_API_KEY", "sk-fake")
    with_key = _ss_spacing()
    assert with_key < no_key
    assert with_key <= 5.0
