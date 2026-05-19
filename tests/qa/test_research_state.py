"""Tests for ResearchState dataclass + persistence."""
import json
from pathlib import Path

import pytest

from paper_distiller.qa.research_state import (
    ResearchState, write_research_state, read_research_state,
)


def _state(**overrides):
    base = dict(
        session_id="rs-2026-05-19-abc",
        question="why diffusion",
        config_snapshot={},
        started_at="2026-05-19T10:00:00",
    )
    base.update(overrides)
    return ResearchState(**base)


def test_state_default_fields():
    s = _state()
    assert s.phase == "seed"
    assert s.papers_distilled == []
    assert s.themes == []
    assert s.synthesis_slugs == []
    assert s.total_cost_cny == 0.0
    assert s.is_done is False
    assert s.stop_reason == ""


def test_state_round_trip_disk(tmp_path):
    s = _state()
    s.papers_distilled = ["a", "b"]
    s.themes = [{"name": "T", "slugs": ["a", "b"], "description": "test"}]
    s.synthesis_slugs = ["synth-1"]
    write_research_state(tmp_path, s)
    s2 = read_research_state(tmp_path, s.session_id)
    assert s2 is not None
    assert s2.papers_distilled == ["a", "b"]
    assert s2.themes[0]["name"] == "T"
    assert s2.synthesis_slugs == ["synth-1"]


def test_read_missing_returns_none(tmp_path):
    assert read_research_state(tmp_path, "no-such-sid") is None
