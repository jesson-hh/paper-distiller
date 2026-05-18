"""Tests for sources/semantic_scholar.py. Uses pytest-mock to stub HTTP."""
from unittest.mock import MagicMock

import pytest

from paper_distiller.sources.semantic_scholar import (
    search,
    lookup_by_arxiv_id,
    lookup_by_doi,
    SSError,
)


def _fake_ss_record(paper_id="abc123", arxiv_id="2501.00001", doi="10.1/x", title="P1"):
    return {
        "paperId": paper_id,
        "title": title,
        "abstract": "abstract text",
        "authors": [{"name": "Alice"}, {"name": "Bob"}],
        "year": 2025,
        "venue": "ICML 2025",
        "externalIds": {"ArXiv": arxiv_id, "DOI": doi},
        "openAccessPdf": {"url": "https://example.com/paper.pdf"},
    }


def _fake_http_ok(json_body):
    r = MagicMock()
    r.status_code = 200
    r.json.return_value = json_body
    r.raise_for_status = MagicMock()
    return r


def _fake_http_404():
    from httpx import HTTPStatusError, Request, Response
    request = Request("GET", "http://test")
    response = Response(404, request=request)
    err = HTTPStatusError("404", request=request, response=response)
    r = MagicMock()
    r.status_code = 404
    r.raise_for_status.side_effect = err
    return r


def test_search_returns_papers(mocker):
    """search() converts SS JSON to Paper dataclasses with correct fields."""
    mock_get = mocker.patch("paper_distiller.sources.semantic_scholar.httpx.Client.get")
    mock_get.return_value = _fake_http_ok({
        "data": [
            _fake_ss_record("abc123", "2501.00001", "10.1/x", "Paper One"),
            _fake_ss_record("def456", "2501.00002", "10.1/y", "Paper Two"),
        ],
        "total": 2,
    })

    papers = search("test query", max_results=5)
    assert len(papers) == 2
    assert papers[0].source == "semanticscholar"
    assert papers[0].paper_id == "abc123"
    assert papers[0].title == "Paper One"
    assert papers[0].arxiv_id == "2501.00001"
    assert papers[0].doi == "10.1/x"
    assert papers[0].ss_paper_id == "abc123"
    assert papers[0].venue == "ICML 2025"
    assert papers[0].open_access_pdf_url == "https://example.com/paper.pdf"
    assert papers[0].pdf_url == "https://example.com/paper.pdf"


def test_search_handles_partial_records(mocker):
    """SS sometimes returns records missing externalIds or openAccessPdf."""
    mock_get = mocker.patch("paper_distiller.sources.semantic_scholar.httpx.Client.get")
    mock_get.return_value = _fake_http_ok({
        "data": [
            # No externalIds, no openAccessPdf
            {"paperId": "ghi789", "title": "Bare paper", "abstract": "x",
             "authors": [], "year": 2020},
        ],
        "total": 1,
    })

    papers = search("anything", max_results=5)
    assert len(papers) == 1
    assert papers[0].arxiv_id is None
    assert papers[0].doi is None
    assert papers[0].open_access_pdf_url is None
    assert papers[0].pdf_url == ""  # no fallback URL


def test_lookup_by_arxiv_id_hit(mocker):
    """lookup_by_arxiv_id returns Paper on 200; URL uses ARXIV: prefix."""
    mock_get = mocker.patch("paper_distiller.sources.semantic_scholar.httpx.Client.get")
    mock_get.return_value = _fake_http_ok(
        _fake_ss_record("xyz", "2503.04164", "10.2/z", "Looked up")
    )

    p = lookup_by_arxiv_id("2503.04164")
    assert p is not None
    assert p.arxiv_id == "2503.04164"
    assert p.title == "Looked up"

    # Verify the URL contains ARXIV: prefix
    call_args = mock_get.call_args
    url = call_args[0][0] if call_args[0] else call_args.kwargs["url"]
    assert "ARXIV:2503.04164" in url


def test_lookup_by_doi_miss_returns_none(mocker):
    """404 from SS lookup returns None, not exception."""
    mock_get = mocker.patch("paper_distiller.sources.semantic_scholar.httpx.Client.get")
    mock_get.return_value = _fake_http_404()

    assert lookup_by_doi("10.999/nonexistent") is None
