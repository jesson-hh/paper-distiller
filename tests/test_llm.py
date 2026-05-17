from unittest.mock import MagicMock

import pytest

from paper_distiller.llm.openai_compatible import LLMClient


def _fake_response(content):
    return {
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50},
    }


def test_complete_basic(mocker):
    mock_post = mocker.patch("paper_distiller.llm.openai_compatible.httpx.Client.post")
    mock_post.return_value.json.return_value = _fake_response("hello back")
    mock_post.return_value.raise_for_status = MagicMock()

    client = LLMClient(api_key="sk-test", base_url="https://x.test/v1", model="qwen-plus")
    out = client.complete([{"role": "user", "content": "hi"}])
    assert out == "hello back"


def test_complete_tracks_token_usage(mocker):
    mock_post = mocker.patch("paper_distiller.llm.openai_compatible.httpx.Client.post")
    mock_post.return_value.json.return_value = _fake_response("response")
    mock_post.return_value.raise_for_status = MagicMock()

    client = LLMClient(api_key="sk-test", base_url="https://x.test/v1", model="qwen-plus")
    client.complete([{"role": "user", "content": "hi"}])
    assert client.total_tokens_in == 100
    assert client.total_tokens_out == 50

    client.complete([{"role": "user", "content": "hi again"}])
    assert client.total_tokens_in == 200
    assert client.total_tokens_out == 100


def test_complete_response_format_json(mocker):
    mock_post = mocker.patch("paper_distiller.llm.openai_compatible.httpx.Client.post")
    mock_post.return_value.json.return_value = _fake_response('{"x": 1}')
    mock_post.return_value.raise_for_status = MagicMock()

    client = LLMClient(api_key="sk-test", base_url="https://x.test/v1", model="qwen-plus")
    client.complete([{"role": "user", "content": "hi"}], response_format="json")

    call_kwargs = mock_post.call_args.kwargs
    body = call_kwargs["json"]
    assert body["response_format"] == {"type": "json_object"}
