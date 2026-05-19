"""Tests for IntentRouter — natural language to slash command JSON."""
import json
from unittest.mock import MagicMock

import pytest

from paper_distiller.agents.router import IntentRouter, RoutingError


def test_router_returns_distill_for_topic_query():
    llm = MagicMock()
    llm.complete.return_value = json.dumps({
        "command": "distill",
        "params": {"topic": "diffusion models", "n": 3},
        "missing_params": [],
        "confidence": 9,
    })
    router = IntentRouter(llm=llm)
    out = router.classify("distill 3 papers on diffusion models")
    assert out["command"] == "distill"
    assert out["params"]["topic"] == "diffusion models"


def test_router_returns_ask_for_question():
    llm = MagicMock()
    llm.complete.return_value = json.dumps({
        "command": "ask",
        "params": {"question": "为什么扩散模型在长序列上效果好？"},
        "missing_params": ["max_rounds", "per_round", "max_cost_cny"],
        "confidence": 8,
    })
    router = IntentRouter(llm=llm)
    out = router.classify("为什么扩散模型在长序列上效果好？")
    assert out["command"] == "ask"
    assert "max_rounds" in out["missing_params"]


def test_router_raises_routing_error_on_malformed_json():
    llm = MagicMock()
    llm.complete.return_value = "not json at all"
    router = IntentRouter(llm=llm)
    with pytest.raises(RoutingError, match="malformed"):
        router.classify("anything")


def test_router_raises_on_unknown_command():
    llm = MagicMock()
    llm.complete.return_value = json.dumps({
        "command": "noexist", "params": {}, "missing_params": [], "confidence": 8,
    })
    router = IntentRouter(llm=llm)
    with pytest.raises(RoutingError, match="unknown command"):
        router.classify("anything")
