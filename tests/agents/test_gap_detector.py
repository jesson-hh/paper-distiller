"""Tests for GapDetector agent."""
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from paper_distiller.agents.base import Context
from paper_distiller.agents.gap_detector import GapDetector
from paper_distiller.qa.research_state import ResearchState


def _state(iterations=1, n_papers=4, themes=None, n_syntheses=2):
    return ResearchState(
        session_id="rs-1", question="why diffusion?",
        config_snapshot={}, started_at="2026-05-19T10:00:00",
        iterations_completed=iterations,
        papers_distilled=[f"a-{i}" for i in range(n_papers)],
        themes=themes if themes is not None else [
            {"name": "Theory", "slugs": ["a-0", "a-1"], "description": ""},
            {"name": "Empirical", "slugs": ["a-2", "a-3"], "description": ""},
        ],
        synthesis_slugs=[f"synth-{i}" for i in range(n_syntheses)],
    )


def _ctx(state):
    return Context(
        cfg=SimpleNamespace(), llm=MagicMock(), vault=MagicMock(),
        shared={"research_state": state},
        on_status=lambda *a, **kw: None,
    )


@pytest.mark.asyncio
async def test_gap_continues_when_llm_says_continue():
    state = _state()
    ctx = _ctx(state)
    ctx.llm.complete.return_value = json.dumps({
        "should_continue": True,
        "missing_aspects": ["缺少 baseline 对比"],
        "next_query": "diffusion baseline comparison 2024",
        "rationale": "Need more empirical work.",
    })
    out = await GapDetector().run(ctx)
    assert out["gap_analysis"]["should_continue"] is True
    assert "baseline" in out["gap_analysis"]["next_query"]
    assert out["gap_analysis"]["missing_aspects"] == ["缺少 baseline 对比"]


@pytest.mark.asyncio
async def test_gap_stops_when_llm_says_stop():
    state = _state(iterations=3)
    ctx = _ctx(state)
    ctx.llm.complete.return_value = json.dumps({
        "should_continue": False,
        "missing_aspects": [],
        "next_query": "",
        "rationale": "Coverage sufficient.",
    })
    out = await GapDetector().run(ctx)
    assert out["gap_analysis"]["should_continue"] is False
    assert out["gap_analysis"]["next_query"] == ""


@pytest.mark.asyncio
async def test_gap_stops_on_malformed_json_after_retry():
    state = _state()
    ctx = _ctx(state)
    ctx.llm.complete.side_effect = ["not json", "still not json"]
    out = await GapDetector().run(ctx)
    # Conservative: stop on failure
    assert out["gap_analysis"]["should_continue"] is False
    assert "failed" in out["gap_analysis"]["rationale"].lower()


@pytest.mark.asyncio
async def test_gap_handles_missing_research_state():
    """If research_state is not in shared (defensive), stop with fallback."""
    ctx = Context(
        cfg=SimpleNamespace(), llm=MagicMock(), vault=MagicMock(),
        shared={},
        on_status=lambda *a, **kw: None,
    )
    out = await GapDetector().run(ctx)
    assert out["gap_analysis"]["should_continue"] is False
    ctx.llm.complete.assert_not_called()


@pytest.mark.asyncio
async def test_gap_detector_deps():
    assert GapDetector().deps == []
