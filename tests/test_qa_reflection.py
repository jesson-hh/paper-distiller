"""Tests for paper_distiller.qa.reflection — LLM reflection call."""
import json
from unittest.mock import MagicMock

import pytest

from paper_distiller.qa.reflection import reflect, ReflectionError


def _llm_returning(content: str):
    llm = MagicMock()
    llm.complete.return_value = content
    return llm


def test_reflect_parses_valid_json():
    """reflect() returns the parsed JSON dict from the LLM response."""
    payload = {
        "is_done": False,
        "confidence": 4,
        "what_we_know": "diffusion basics",
        "what_is_missing": "volatility clustering",
        "next_query": "volatility clustering diffusion",
        "next_query_rationale": "directly addresses the gap",
        "suggest_stop": False,
    }
    llm = _llm_returning(json.dumps(payload))
    result = reflect(
        question="why diffusion for finance?",
        articles_summary=[],
        prior_queries=["initial query"],
        round_num=1,
        max_rounds=5,
        llm=llm,
    )
    assert result["is_done"] is False
    assert result["confidence"] == 4
    assert result["next_query"] == "volatility clustering diffusion"


def test_reflect_retries_once_on_malformed_json():
    """First call returns garbage; second call returns valid JSON; reflect returns success."""
    llm = MagicMock()
    llm.complete.side_effect = [
        "this is not json",
        json.dumps({
            "is_done": True, "confidence": 9, "what_we_know": "done",
            "what_is_missing": "", "next_query": "", "next_query_rationale": "",
            "suggest_stop": False,
        }),
    ]
    result = reflect(
        question="q",
        articles_summary=[],
        prior_queries=[],
        round_num=1,
        max_rounds=5,
        llm=llm,
    )
    assert result["is_done"] is True
    assert llm.complete.call_count == 2


def test_reflect_raises_after_two_failures():
    """If both attempts return malformed JSON, raise ReflectionError."""
    llm = MagicMock()
    llm.complete.side_effect = ["not json", "still not json"]
    with pytest.raises(ReflectionError, match="malformed"):
        reflect(
            question="q",
            articles_summary=[],
            prior_queries=[],
            round_num=1,
            max_rounds=5,
            llm=llm,
        )
