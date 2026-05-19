"""Tests for CandidateDedup agent — filters candidates against qa_state.articles_seen_ids."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from paper_distiller.agents.base import Context
from paper_distiller.agents.dedup import CandidateDedup
from paper_distiller.qa.state import SessionState
from paper_distiller.sources.arxiv import Paper


def _paper(pid, doi=None):
    return Paper(
        source="arxiv", paper_id=pid, arxiv_id=pid, doi=doi,
        title=f"P{pid}", authors=[], abstract="...",
        pdf_url="...", published="2025-01-01", categories=[],
    )


def _ctx(candidates, seen_ids=None):
    state = SessionState(
        session_id="sid-1", question="?", config_snapshot={},
        started_at="2026-05-19T10:00:00",
        articles_seen_ids=set(seen_ids or []),
    )
    return Context(
        cfg=SimpleNamespace(), llm=MagicMock(), vault=MagicMock(),
        shared={"candidates": candidates, "qa_state": state},
        on_status=lambda *a, **kw: None,
    )


@pytest.mark.asyncio
async def test_dedup_passes_through_when_seen_empty():
    cands = [_paper("X1"), _paper("X2")]
    ctx = _ctx(cands, seen_ids=set())
    out = await CandidateDedup().run(ctx)
    assert [p.arxiv_id for p in out["candidates"]] == ["X1", "X2"]


@pytest.mark.asyncio
async def test_dedup_filters_seen_arxiv_ids():
    cands = [_paper("X1"), _paper("X2"), _paper("X3")]
    ctx = _ctx(cands, seen_ids={"X2"})
    out = await CandidateDedup().run(ctx)
    assert [p.arxiv_id for p in out["candidates"]] == ["X1", "X3"]


@pytest.mark.asyncio
async def test_dedup_filters_seen_dois():
    cands = [_paper("X1", doi="10.1/abc"), _paper("X2"), _paper("X3", doi="10.2/def")]
    ctx = _ctx(cands, seen_ids={"10.1/abc"})
    out = await CandidateDedup().run(ctx)
    assert {p.arxiv_id for p in out["candidates"]} == {"X2", "X3"}


@pytest.mark.asyncio
async def test_dedup_noop_when_no_qa_state():
    """In single-pass mode (no qa_state in shared), dedup is a no-op pass-through."""
    cands = [_paper("X1")]
    ctx = Context(
        cfg=SimpleNamespace(), llm=MagicMock(), vault=MagicMock(),
        shared={"candidates": cands},  # no qa_state
        on_status=lambda *a, **kw: None,
    )
    out = await CandidateDedup().run(ctx)
    assert out["candidates"] == cands


@pytest.mark.asyncio
async def test_dedup_deps():
    assert CandidateDedup().deps == ["candidate-merger"]
