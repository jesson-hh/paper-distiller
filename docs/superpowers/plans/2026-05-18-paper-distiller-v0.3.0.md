# paper-distiller v0.3.0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship v0.3.0 — add Semantic Scholar as a second paper source (P1) and a PDF fallback chain (P2). User picks via `--source {arxiv,ss,both}` (default `both`); pipeline merges candidates, dedups by arxiv_id/DOI, and falls back to SS open-access PDF URLs when arxiv's direct PDF fails.

**Architecture:** Unify `ArxivPaper` → generic `Paper` dataclass with cross-source identity fields (arxiv_id, doi, ss_paper_id). New `sources/semantic_scholar.py` module mirrors `sources/arxiv.py`'s interface. Pipeline gathers candidates from configured source(s), dedups via `merge_candidates`, and threads a 3-step PDF fetch (primary → SS open-access → abstract-only). New `VaultStore.find_by_doi` extends vault dedup beyond arxiv ids.

**Tech Stack:** Same as v0.2 — Python 3.10+, httpx, arxiv, pymupdf, python-dotenv, pytest, pytest-mock. No new runtime dependencies (Semantic Scholar API is plain HTTPS via httpx).

**Spec:** [docs/superpowers/specs/2026-05-18-paper-distiller-v0.3.0-design.md](../specs/2026-05-18-paper-distiller-v0.3.0-design.md)

---

## File Structure

| File | Action | LOC delta |
|---|---|---|
| `src/paper_distiller/sources/arxiv.py` | Refactor `ArxivPaper` → `Paper` (with alias); update `search()` to set new fields | ~20 modified |
| `src/paper_distiller/sources/semantic_scholar.py` | NEW: `search`, `lookup_by_arxiv_id`, `lookup_by_doi` | +110 |
| `src/paper_distiller/sources/__init__.py` | Re-export `Paper` + SS functions | +3 |
| `src/paper_distiller/vault/store.py` | Add `find_by_doi` method | +21 |
| `src/paper_distiller/config.py` | Add `source` + `ss_api_key` fields | +6 |
| `src/paper_distiller/cli.py` | Add `--source` flag | +3 |
| `src/paper_distiller/pipeline.py` | New `_gather_candidates` + `merge_candidates` + `_fetch_with_fallback`; extend dedup with DOI | ~80 net |
| `src/paper_distiller/distill/article.py` | Refs injection: arxiv → doi → ss precedence | ~8 modified |
| `pyproject.toml` | version 0.2.0 → 0.3.0 | 1 line |
| `src/paper_distiller/__init__.py` | __version__ 0.2.0 → 0.3.0 | 1 line |
| `tests/test_smoke.py` | __version__ assertion 0.2.0 → 0.3.0 | 1 line |
| `.env.example` | Add `PD_SS_API_KEY=` (optional) | +2 |
| `CHANGELOG.md` | Add [0.3.0] section | ~30 |
| `tests/test_semantic_scholar.py` | NEW: 4 tests | +110 |
| `tests/test_vault_store.py` | Add 2 find_by_doi tests | +25 |
| `tests/test_config.py` | Add 1 source-field test | +12 |
| `tests/test_pipeline.py` | Add 3 source/merge/fallback tests | +90 |

**Test count**: v0.2 had 51 passing. v0.3 adds 10 new (4 SS + 2 vault + 1 config + 3 pipeline). Total after v0.3: **61 passing**.

**Working directory throughout this plan: `G:\paper-distiller\`**

---

## Task 1: Refactor `ArxivPaper` → unified `Paper` dataclass

**Files:**
- Modify: `src/paper_distiller/sources/arxiv.py`
- Modify: `src/paper_distiller/sources/__init__.py`

This task is BACKWARD-COMPATIBLE — `ArxivPaper` is kept as a module-level alias for `Paper`. v0.2 imports continue to work. Existing tests must continue to pass without modification.

- [ ] **Step 1: Run baseline test suite to confirm starting point**

```bash
.venv\Scripts\python.exe -m pytest -v
```

Expected: 51 passed (v0.2 baseline).

- [ ] **Step 2: Replace `ArxivPaper` definition in `src/paper_distiller/sources/arxiv.py`**

In `src/paper_distiller/sources/arxiv.py`, find the existing `ArxivPaper` dataclass (around lines 12–20):

```python
@dataclass
class ArxivPaper:
    arxiv_id: str
    title: str
    authors: list
    abstract: str
    pdf_url: str
    published: str
    categories: list = field(default_factory=list)
```

Replace it with the unified `Paper` definition + alias:

```python
@dataclass
class Paper:
    """A research paper, sourced from arxiv or Semantic Scholar.

    source: which API produced this record ("arxiv" or "semanticscholar")
    paper_id: canonical id within source (arxiv_id for arxiv, paperId for SS)
    arxiv_id / doi / ss_paper_id: cross-source identity (any may be None;
        at least one is always set)
    venue / open_access_pdf_url: SS-only enrichment (None when source="arxiv")
    """
    source: str
    paper_id: str
    title: str
    authors: list
    abstract: str
    published: str
    pdf_url: str
    categories: list = field(default_factory=list)

    arxiv_id: str | None = None
    doi: str | None = None
    ss_paper_id: str | None = None

    venue: str | None = None
    open_access_pdf_url: str | None = None


# Backward-compat alias — v0.2 imports of ArxivPaper continue to work.
ArxivPaper = Paper
```

- [ ] **Step 3: Update `search()` in the same file to populate the new fields**

Find the `search()` function (around lines 33–53). The current implementation builds `ArxivPaper(arxiv_id=arxiv_id, title=..., ..., categories=list(result.categories))`. Update the `papers.append(...)` call to use the unified shape:

```python
        papers.append(Paper(
            source="arxiv",
            paper_id=arxiv_id,
            title=result.title.strip(),
            authors=[a.name for a in result.authors[:10]],
            abstract=result.summary.strip(),
            pdf_url=result.pdf_url,
            published=result.published.isoformat()[:10],
            categories=list(result.categories),
            arxiv_id=arxiv_id,
        ))
```

Note the two new explicit fields: `source="arxiv"`, `paper_id=arxiv_id`, plus `arxiv_id=arxiv_id` (yes, paper_id == arxiv_id for arxiv-sourced papers; the duplication is intentional — `paper_id` is the canonical within-source key, `arxiv_id` is the cross-source identifier).

- [ ] **Step 4: Update `src/paper_distiller/sources/__init__.py` to re-export `Paper`**

Find the current content:

```python
from .arxiv import ArxivPaper, search, download_pdf

__all__ = ["ArxivPaper", "search", "download_pdf"]
```

Replace with:

```python
from .arxiv import Paper, ArxivPaper, search, download_pdf

__all__ = ["Paper", "ArxivPaper", "search", "download_pdf"]
```

- [ ] **Step 5: Run the full test suite — verify no regressions**

```bash
.venv\Scripts\python.exe -m pytest -v
```

Expected: 51 passed. The existing `test_arxiv.py` tests use `ArxivPaper(...)` constructor; since `ArxivPaper = Paper`, those calls now pass through to the new dataclass. **Crucially**, the existing test:

```python
paper = ArxivPaper(
    arxiv_id="2501.00001", title="t", authors=[],
    abstract="a", pdf_url="https://arxiv.org/pdf/2501.00001.pdf",
    published="2024-01-01", categories=[],
)
```

worked in v0.2 because all those fields were positional and required. Now they're still fields on `Paper`, but `source` and `paper_id` are REQUIRED first positional args. **This test will fail.** Fix: the implementer must update `tests/test_arxiv.py:test_download_pdf_writes_file` (and any other call sites) to pass `source="arxiv"` and `paper_id=...`. The other test (`test_search_returns_arxiv_papers`) calls through `search()` which now produces the correct shape via Step 3, so that test still passes.

**Specifically, update `tests/test_arxiv.py:test_download_pdf_writes_file`'s `ArxivPaper(...)` call to:**

```python
    paper = ArxivPaper(
        source="arxiv",
        paper_id="2501.00001",
        arxiv_id="2501.00001",
        title="t", authors=[],
        abstract="a", pdf_url="https://arxiv.org/pdf/2501.00001.pdf",
        published="2024-01-01", categories=[],
    )
```

After this update, re-run pytest. Expected: 51 passed.

- [ ] **Step 6: Commit**

```bash
git add src/paper_distiller/sources/arxiv.py src/paper_distiller/sources/__init__.py tests/test_arxiv.py
git commit -m "refactor(sources): unify ArxivPaper -> Paper with cross-source identity fields

Adds source / paper_id / arxiv_id / doi / ss_paper_id / venue /
open_access_pdf_url fields. ArxivPaper kept as alias for backward compat.

Prepares for v0.3 Semantic Scholar source — every Paper now carries the
identity info needed to dedup across two search APIs."
```

---

## Task 2: New `sources/semantic_scholar.py` module

**Files (create):**
- `src/paper_distiller/sources/semantic_scholar.py`
- `tests/test_semantic_scholar.py`

**Files (modify):**
- `src/paper_distiller/sources/__init__.py` (add SS re-exports)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_semantic_scholar.py`:

```python
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
```

- [ ] **Step 2: Run, confirm fail**

```bash
.venv\Scripts\python.exe -m pytest tests/test_semantic_scholar.py -v
```

Expected: `ModuleNotFoundError: No module named 'paper_distiller.sources.semantic_scholar'`.

- [ ] **Step 3: Implement `src/paper_distiller/sources/semantic_scholar.py`**

```python
"""Semantic Scholar Graph API client.

Free tier: 100 req / 5 min, ~1 req/s. Optional `api_key` parameter sent as
x-api-key header to raise rate limits (apply at https://www.semanticscholar.org/product/api).

Endpoints used:
  GET /paper/search?query=...&limit=...&fields=...    — keyword search
  GET /paper/ARXIV:<id>?fields=...                    — lookup by arxiv id
  GET /paper/DOI:<doi>?fields=...                     — lookup by DOI
"""

from __future__ import annotations

import httpx

from .arxiv import Paper


class SSError(RuntimeError):
    """Raised when SS API returns an error other than 404."""


_BASE_URL = "https://api.semanticscholar.org/graph/v1"
_FIELDS = "title,abstract,authors,year,externalIds,openAccessPdf,venue"
_TIMEOUT = 30.0


def _headers(api_key: str | None) -> dict:
    if api_key:
        return {"x-api-key": api_key}
    return {}


def _record_to_paper(record: dict) -> Paper:
    """Convert an SS API record into the unified Paper dataclass."""
    external = record.get("externalIds") or {}
    open_access = record.get("openAccessPdf") or {}
    authors_field = record.get("authors") or []

    pdf_url_val = open_access.get("url") or ""
    return Paper(
        source="semanticscholar",
        paper_id=record["paperId"],
        title=(record.get("title") or "").strip(),
        authors=[a.get("name", "") for a in authors_field[:10]],
        abstract=(record.get("abstract") or "").strip(),
        published=str(record.get("year") or ""),
        pdf_url=pdf_url_val,
        arxiv_id=external.get("ArXiv"),
        doi=external.get("DOI"),
        ss_paper_id=record["paperId"],
        venue=record.get("venue"),
        open_access_pdf_url=open_access.get("url"),
    )


def search(query: str, max_results: int = 30,
           api_key: str | None = None) -> list[Paper]:
    """Free-text search against Semantic Scholar. Returns up to max_results papers."""
    url = f"{_BASE_URL}/paper/search"
    params = {"query": query, "limit": max_results, "fields": _FIELDS}
    try:
        with httpx.Client(timeout=_TIMEOUT) as client:
            r = client.get(url, params=params, headers=_headers(api_key))
            r.raise_for_status()
    except httpx.HTTPError as e:
        raise SSError(f"SS search failed: {e}") from e

    data = r.json()
    raw_records = data.get("data") or []
    papers = []
    for record in raw_records:
        if not record.get("paperId"):
            continue
        try:
            papers.append(_record_to_paper(record))
        except Exception:
            continue
    return papers


def lookup_by_arxiv_id(arxiv_id: str,
                        api_key: str | None = None) -> Paper | None:
    """Resolve an arxiv id via SS. Returns None on 404 / missing record."""
    return _lookup(f"ARXIV:{arxiv_id}", api_key)


def lookup_by_doi(doi: str, api_key: str | None = None) -> Paper | None:
    """Resolve a DOI via SS. Returns None on 404 / missing record."""
    return _lookup(f"DOI:{doi}", api_key)


def _lookup(id_with_prefix: str, api_key: str | None) -> Paper | None:
    url = f"{_BASE_URL}/paper/{id_with_prefix}"
    params = {"fields": _FIELDS}
    try:
        with httpx.Client(timeout=_TIMEOUT) as client:
            r = client.get(url, params=params, headers=_headers(api_key))
            if r.status_code == 404:
                return None
            r.raise_for_status()
    except httpx.HTTPError as e:
        if "404" in str(e):
            return None
        raise SSError(f"SS lookup {id_with_prefix} failed: {e}") from e

    record = r.json()
    if not record.get("paperId"):
        return None
    try:
        return _record_to_paper(record)
    except Exception:
        return None
```

- [ ] **Step 4: Update `src/paper_distiller/sources/__init__.py`**

Replace with:

```python
from .arxiv import Paper, ArxivPaper, search, download_pdf
from . import semantic_scholar as ss

__all__ = ["Paper", "ArxivPaper", "search", "download_pdf", "ss"]
```

The `ss` module-level re-export lets callers do `from paper_distiller.sources import ss` and then `ss.search(...)`, `ss.lookup_by_arxiv_id(...)`, etc.

- [ ] **Step 5: Run, confirm pass**

```bash
.venv\Scripts\python.exe -m pytest tests/test_semantic_scholar.py -v
```

Expected: 4 passed.

- [ ] **Step 6: Run full suite**

```bash
.venv\Scripts\python.exe -m pytest -v
```

Expected: 55 passed (51 prior + 4 new).

- [ ] **Step 7: Commit**

```bash
git add src/paper_distiller/sources/semantic_scholar.py src/paper_distiller/sources/__init__.py tests/test_semantic_scholar.py
git commit -m "feat(sources): add Semantic Scholar API client

Provides search(), lookup_by_arxiv_id(), lookup_by_doi() returning the
unified Paper dataclass. Free tier (100/5min) is sufficient for typical
runs; optional PD_SS_API_KEY lifts the limit.

Errors: 404 -> None (graceful), other failures -> SSError."
```

---

## Task 3: `VaultStore.find_by_doi`

**Files:**
- Modify: `src/paper_distiller/vault/store.py` (add method)
- Modify: `tests/test_vault_store.py` (add 2 tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_vault_store.py`:

```python
def test_find_by_doi_hit(tmp_vault: Path):
    """find_by_doi mirrors find_by_arxiv_id but matches doi: refs."""
    store = VaultStore(tmp_vault)
    store.save_entry(
        title="A paper with DOI",
        category="articles",
        body="x",
        refs=["doi:10.1234/abcd"],
        slug="doi-paper",
    )
    found = store.find_by_doi("10.1234/abcd")
    assert found is not None
    assert found.slug == "doi-paper"


def test_find_by_doi_miss_and_articles_only(tmp_vault: Path):
    """find_by_doi returns None for unknown DOI and ignores non-articles category."""
    store = VaultStore(tmp_vault)
    # An article with a different DOI
    store.save_entry(title="X", category="articles", body="x",
                     refs=["doi:10.9999/nope"])
    # A survey with the target DOI — must NOT match
    store.save_entry(title="S", category="surveys", body="x",
                     refs=["doi:10.1234/abcd"])
    assert store.find_by_doi("10.1234/abcd") is None
```

- [ ] **Step 2: Run, confirm fail**

```bash
.venv\Scripts\python.exe -m pytest tests/test_vault_store.py::test_find_by_doi_hit tests/test_vault_store.py::test_find_by_doi_miss_and_articles_only -v
```

Expected: `AttributeError: 'VaultStore' object has no attribute 'find_by_doi'`.

- [ ] **Step 3: Implement `find_by_doi`**

In `src/paper_distiller/vault/store.py`, add this method to `VaultStore`. Place it IMMEDIATELY AFTER `find_by_arxiv_id` (and before `list_entries`):

```python
    def find_by_doi(self, doi: str) -> Entry | None:
        """Find an article whose `refs` frontmatter contains `doi:<doi>`.

        Mirrors find_by_arxiv_id semantics — only scans `articles/`, returns
        the first match. Used by the pipeline for DOI-based dedup when a paper
        is sourced from Semantic Scholar (which always returns DOI when known).
        """
        target_ref = f"doi:{doi}"
        folder = self.root / "articles"
        if not folder.exists():
            return None
        for f in folder.glob("*.md"):
            try:
                meta, body = parse_frontmatter(f.read_text(encoding="utf-8"))
                if target_ref in (meta.get("refs") or []):
                    return Entry(
                        slug=meta.get("slug", f.stem),
                        category=meta.get("category", "articles"),
                        title=meta.get("title", f.stem),
                        tags=meta.get("tags") or [],
                        refs=meta.get("refs") or [],
                        links=meta.get("links") or [],
                        created=meta.get("created", ""),
                        updated=meta.get("updated", ""),
                        body=body,
                    )
            except Exception:
                continue
        return None
```

- [ ] **Step 4: Run, confirm pass**

```bash
.venv\Scripts\python.exe -m pytest tests/test_vault_store.py -v
```

Expected: 21 passed (19 prior + 2 new).

- [ ] **Step 5: Run full suite**

```bash
.venv\Scripts\python.exe -m pytest -v
```

Expected: 57 passed.

- [ ] **Step 6: Commit**

```bash
git add src/paper_distiller/vault/store.py tests/test_vault_store.py
git commit -m "feat(vault): add find_by_doi for DOI-based dedup lookup"
```

---

## Task 4: Config + CLI `--source` flag

**Files:**
- Modify: `src/paper_distiller/config.py` (add fields + validation)
- Modify: `src/paper_distiller/cli.py` (add flag)
- Modify: `tests/test_config.py` (add 1 test)
- Modify: `.env.example`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_config.py`:

```python
def test_config_source_field(monkeypatch, tmp_path: Path):
    """Config exposes a `source` field defaulting to 'both', accepts arxiv/ss/both, rejects others."""
    monkeypatch.setenv("PD_API_KEY", "sk-test")
    monkeypatch.setenv("PD_BASE_URL", "https://x/v1")
    monkeypatch.setenv("PD_MODEL", "m")

    # Default
    cfg = load_config(vault_path=tmp_path, topic="x")
    assert cfg.source == "both"

    # Explicit valid values
    for val in ("arxiv", "ss", "both"):
        cfg = load_config(vault_path=tmp_path, topic="x", source=val)
        assert cfg.source == val

    # Invalid value raises
    with pytest.raises(ValueError, match="source"):
        load_config(vault_path=tmp_path, topic="x", source="nonsense")

    # ss_api_key reads from env
    monkeypatch.setenv("PD_SS_API_KEY", "ss-key-123")
    cfg = load_config(vault_path=tmp_path, topic="x")
    assert cfg.ss_api_key == "ss-key-123"
```

- [ ] **Step 2: Run, confirm fail**

```bash
.venv\Scripts\python.exe -m pytest tests/test_config.py::test_config_source_field -v
```

Expected: `TypeError: load_config() got an unexpected keyword argument 'source'`.

- [ ] **Step 3: Update `src/paper_distiller/config.py`**

Add two fields to the `Config` dataclass (insert between `verbose` and `api_key`):

```python
    source: str = "both"
    ss_api_key: str | None = None
```

Update `load_config()` signature to accept `source` and validate it. Find the existing signature:

```python
def load_config(
    vault_path: Path | str,
    topic: str | None = None,
    author: str | None = None,
    n: int = 5,
    pool: int = 30,
    force: bool = False,
    dry_run: bool = False,
    verbose: bool = False,
    model_override: str | None = None,
    provider_override: str | None = None,
) -> Config:
```

Replace with:

```python
def load_config(
    vault_path: Path | str,
    topic: str | None = None,
    author: str | None = None,
    n: int = 5,
    pool: int = 30,
    force: bool = False,
    dry_run: bool = False,
    verbose: bool = False,
    source: str = "both",
    model_override: str | None = None,
    provider_override: str | None = None,
) -> Config:
```

In the function body, add validation right after the existing `if not topic and not author` check:

```python
    if source not in ("arxiv", "ss", "both"):
        raise ValueError(f"source must be one of arxiv/ss/both (got {source!r})")
```

Then in the `Config(...)` constructor call, add the two new field assignments. Find the existing return:

```python
    return Config(
        vault_path=Path(vault_path),
        topic=topic,
        author=author,
        top_n=n,
        pool=pool,
        force=force,
        dry_run=dry_run,
        verbose=verbose,
        api_key=_require("PD_API_KEY"),
        base_url=_require("PD_BASE_URL"),
        model=model_override or _require("PD_MODEL"),
        provider_name=provider_override or os.getenv("PD_PROVIDER_NAME", "unspecified"),
        pdf_timeout_sec=int(os.getenv("PD_PDF_TIMEOUT", "60")),
        min_papers_for_survey=int(os.getenv("PD_MIN_SURVEY", "2")),
    )
```

Replace with:

```python
    return Config(
        vault_path=Path(vault_path),
        topic=topic,
        author=author,
        top_n=n,
        pool=pool,
        force=force,
        dry_run=dry_run,
        verbose=verbose,
        source=source,
        ss_api_key=os.getenv("PD_SS_API_KEY") or None,
        api_key=_require("PD_API_KEY"),
        base_url=_require("PD_BASE_URL"),
        model=model_override or _require("PD_MODEL"),
        provider_name=provider_override or os.getenv("PD_PROVIDER_NAME", "unspecified"),
        pdf_timeout_sec=int(os.getenv("PD_PDF_TIMEOUT", "60")),
        min_papers_for_survey=int(os.getenv("PD_MIN_SURVEY", "2")),
    )
```

- [ ] **Step 4: Update `src/paper_distiller/cli.py` to pass `--source`**

Find the `build_parser()` function. After the `--verbose` argument and before `--model`, add:

```python
    p.add_argument("--source", choices=["arxiv", "ss", "both"], default="both",
                    help="Paper source(s) to search (default both).")
```

In `main()`, update the `load_config(...)` call to pass `source=args.source`. Find:

```python
        cfg = load_config(
            vault_path=args.vault,
            topic=args.topic,
            author=args.author,
            n=args.n,
            pool=args.pool,
            force=args.force,
            dry_run=args.dry_run,
            verbose=args.verbose,
            model_override=args.model,
            provider_override=args.provider,
        )
```

Replace with:

```python
        cfg = load_config(
            vault_path=args.vault,
            topic=args.topic,
            author=args.author,
            n=args.n,
            pool=args.pool,
            force=args.force,
            dry_run=args.dry_run,
            verbose=args.verbose,
            source=args.source,
            model_override=args.model,
            provider_override=args.provider,
        )
```

- [ ] **Step 5: Update `.env.example`**

Append to `.env.example`:

```bash

# Optional: Semantic Scholar API key (free, raises rate limit). Apply at
# https://www.semanticscholar.org/product/api. Without it, the free tier
# (100 req / 5 min) is sufficient for personal use.
PD_SS_API_KEY=
```

- [ ] **Step 6: Run config tests, confirm pass**

```bash
.venv\Scripts\python.exe -m pytest tests/test_config.py -v
```

Expected: 5 passed (4 v0.2 + 1 new).

- [ ] **Step 7: Verify CLI help shows new flag**

```bash
.venv\Scripts\paper-distiller.exe --help
```

Expected: output contains `--source {arxiv,ss,both}` line.

- [ ] **Step 8: Run full suite**

```bash
.venv\Scripts\python.exe -m pytest -v
```

Expected: 58 passed.

- [ ] **Step 9: Commit**

```bash
git add src/paper_distiller/config.py src/paper_distiller/cli.py .env.example tests/test_config.py
git commit -m "feat(cli+config): --source {arxiv,ss,both} flag, default both

Adds Config.source (validated) and Config.ss_api_key (from PD_SS_API_KEY
env var). CLI exposes --source. Pipeline (next task) consumes it to gate
which sources are searched."
```

---

## Task 5: Pipeline source coordination + PDF fallback

**Files:**
- Modify: `src/paper_distiller/pipeline.py`
- Modify: `tests/test_pipeline.py`

This is the largest task — wires source choice, dedup, and PDF fallback together.

- [ ] **Step 1: Write the 3 new integration tests**

Append to `tests/test_pipeline.py`:

```python
def test_pipeline_source_arxiv_only(tmp_path, mocker):
    """--source arxiv: SS search is NOT called; only arxiv candidates rank."""
    from paper_distiller.pipeline import run
    cfg = _config(tmp_path); cfg.vault_path.mkdir()
    cfg.source = "arxiv"

    mock_arxiv = mocker.patch("paper_distiller.pipeline.arxiv_search",
                               return_value=[_paper(1)])
    mock_ss_search = mocker.patch("paper_distiller.pipeline.ss_search")
    mocker.patch("paper_distiller.pipeline.rank", return_value=[_paper(1)])
    mocker.patch("paper_distiller.pipeline.download_pdf",
                 side_effect=lambda p, d, **k: Path(d) / f"{p.paper_id}.pdf")
    mocker.patch("paper_distiller.pipeline.extract_text", return_value="x" * 1000)

    def fake_distill(paper, full_text, wiki_index, llm):
        return ArticleResult(slug=f"art-{paper.paper_id}",
                              title=f"T", body="b", tags=[],
                              refs=[f"arxiv:{paper.arxiv_id}"], depth="full-pdf")
    mocker.patch("paper_distiller.pipeline.distill_article", side_effect=fake_distill)
    mocker.patch("paper_distiller.pipeline.compose_survey")
    mocker.patch("paper_distiller.pipeline.LLMClient")

    run(cfg)
    mock_arxiv.assert_called_once()
    mock_ss_search.assert_not_called()


def test_pipeline_source_both_merges_and_dedups(tmp_path, mocker):
    """--source both: candidates from both APIs are deduped by arxiv_id."""
    from paper_distiller.pipeline import run
    from paper_distiller.sources.arxiv import Paper
    cfg = _config(tmp_path); cfg.vault_path.mkdir()
    cfg.source = "both"

    # Same paper from both sources (matching arxiv_id) — must dedup to 1
    arxiv_paper = Paper(
        source="arxiv", paper_id="2501.00001", arxiv_id="2501.00001",
        title="P1 (arxiv)", authors=[], abstract="a",
        pdf_url="https://arxiv.org/pdf/2501.00001.pdf",
        published="2025-01-01", categories=["math.AT"],
    )
    ss_paper_duplicate = Paper(
        source="semanticscholar", paper_id="ss-xyz",
        arxiv_id="2501.00001", doi="10.1/dup", ss_paper_id="ss-xyz",
        title="P1 (ss view)", authors=[], abstract="a",
        pdf_url="https://example.com/ss.pdf",
        published="2025",
    )
    ss_only_paper = Paper(
        source="semanticscholar", paper_id="ss-abc",
        doi="10.2/unique", ss_paper_id="ss-abc",
        title="P2 (ss only)", authors=[], abstract="b",
        pdf_url="https://example.com/ss2.pdf",
        published="2025",
    )

    mocker.patch("paper_distiller.pipeline.arxiv_search",
                 return_value=[arxiv_paper])
    mocker.patch("paper_distiller.pipeline.ss_search",
                 return_value=[ss_paper_duplicate, ss_only_paper])

    # Capture what rank() actually receives
    captured_candidates = []
    def capture_rank(candidates, *args, **kwargs):
        captured_candidates.append(list(candidates))
        return candidates[:2]  # take both
    mocker.patch("paper_distiller.pipeline.rank", side_effect=capture_rank)

    mocker.patch("paper_distiller.pipeline.download_pdf",
                 side_effect=lambda p, d, **k: Path(d) / f"{p.paper_id}.pdf")
    mocker.patch("paper_distiller.pipeline.extract_text", return_value="x" * 1000)

    def fake_distill(paper, full_text, wiki_index, llm):
        return ArticleResult(slug=f"art-{paper.paper_id}",
                              title="T", body="b", tags=[], refs=[],
                              depth="full-pdf")
    mocker.patch("paper_distiller.pipeline.distill_article", side_effect=fake_distill)
    mocker.patch("paper_distiller.pipeline.compose_survey")
    mocker.patch("paper_distiller.pipeline.LLMClient")

    run(cfg)

    # 2 unique papers after dedup (arxiv copy of P1 wins; ss-only P2 kept)
    assert len(captured_candidates[0]) == 2
    # Confirm the arxiv-sourced version of the duplicate was kept (not SS)
    p1 = [p for p in captured_candidates[0] if p.arxiv_id == "2501.00001"][0]
    assert p1.source == "arxiv"
    # ss-only paper present
    assert any(p.doi == "10.2/unique" for p in captured_candidates[0])


def test_pipeline_pdf_fallback_to_ss(tmp_path, mocker):
    """When arxiv PDF fetch fails for an arxiv-sourced paper with arxiv_id,
    pipeline should call ss.lookup_by_arxiv_id and try its open_access_pdf_url."""
    from paper_distiller.pipeline import run
    from paper_distiller.sources.arxiv import Paper
    cfg = _config(tmp_path); cfg.vault_path.mkdir()
    cfg.source = "arxiv"  # search only arxiv but still uses SS for PDF fallback

    arxiv_paper = Paper(
        source="arxiv", paper_id="2501.00001", arxiv_id="2501.00001",
        title="P1", authors=[], abstract="abstract content " * 50,
        pdf_url="https://arxiv.org/pdf/2501.00001.pdf",
        published="2025-01-01", categories=["math.AT"],
    )
    mocker.patch("paper_distiller.pipeline.arxiv_search",
                 return_value=[arxiv_paper])
    mocker.patch("paper_distiller.pipeline.rank", return_value=[arxiv_paper])

    # Step 1: arxiv PDF fetch fails
    arxiv_fail = mocker.patch("paper_distiller.pipeline.download_pdf",
                                side_effect=Exception("HTTP 503"))

    # Step 2: SS lookup returns paper with open_access_pdf_url
    ss_record = Paper(
        source="semanticscholar", paper_id="ss-1",
        arxiv_id="2501.00001", title="P1 (via SS)", authors=[],
        abstract="x", pdf_url="https://mirror.example.com/p1.pdf",
        published="2025",
        open_access_pdf_url="https://mirror.example.com/p1.pdf",
    )
    mock_ss_lookup = mocker.patch("paper_distiller.pipeline.ss_lookup_by_arxiv_id",
                                    return_value=ss_record)
    mock_download_from_url = mocker.patch(
        "paper_distiller.pipeline.download_pdf_from_url",
        side_effect=lambda url, dest_dir, filename, timeout: Path(dest_dir) / filename,
    )
    mocker.patch("paper_distiller.pipeline.extract_text", return_value="x" * 1000)

    def fake_distill(paper, full_text, wiki_index, llm):
        return ArticleResult(slug="ok", title="T", body="b", tags=[],
                              refs=[], depth="full-pdf")
    mocker.patch("paper_distiller.pipeline.distill_article", side_effect=fake_distill)
    mocker.patch("paper_distiller.pipeline.compose_survey")
    mocker.patch("paper_distiller.pipeline.LLMClient")

    run(cfg)

    # arxiv download attempted, failed
    arxiv_fail.assert_called_once()
    # SS lookup happened
    mock_ss_lookup.assert_called_once_with("2501.00001", api_key=None)
    # SS open-access URL was tried
    mock_download_from_url.assert_called_once()
    call_kwargs = mock_download_from_url.call_args
    url_arg = call_kwargs.kwargs.get("url") or call_kwargs.args[0]
    assert url_arg == "https://mirror.example.com/p1.pdf"
```

- [ ] **Step 2: Run, confirm fail**

```bash
.venv\Scripts\python.exe -m pytest tests/test_pipeline.py -v
```

Expected: 3 failures (the new tests). The other 5 should still pass.

- [ ] **Step 3: Implement the pipeline changes**

This is a structural change. We modify `src/paper_distiller/pipeline.py`. Two phases:

**Phase A — refactor `sources.arxiv.download_pdf` into URL-based primitive.**

In `src/paper_distiller/sources/arxiv.py`, find the existing `download_pdf` function (currently takes `paper: Paper, dest_dir, timeout`):

```python
def download_pdf(paper: ArxivPaper, dest_dir: Path, timeout: float = 60.0) -> Path:
    """Download paper PDF to dest_dir / <arxiv_id>.pdf. Returns the path."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{paper.arxiv_id}.pdf"
    with httpx.stream("GET", paper.pdf_url, timeout=timeout, follow_redirects=True) as r:
        r.raise_for_status()
        with dest.open("wb") as f:
            for chunk in r.iter_bytes():
                f.write(chunk)
    return dest
```

Add a new lower-level utility ABOVE it and refactor `download_pdf` to call it:

```python
def download_pdf_from_url(url: str, dest_dir: Path, filename: str,
                           timeout: float = 60.0) -> Path:
    """Stream a PDF from a URL into dest_dir/filename. Returns the saved path."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / filename
    with httpx.stream("GET", url, timeout=timeout, follow_redirects=True) as r:
        r.raise_for_status()
        with dest.open("wb") as f:
            for chunk in r.iter_bytes():
                f.write(chunk)
    return dest


def download_pdf(paper: Paper, dest_dir: Path, timeout: float = 60.0) -> Path:
    """Download paper PDF to dest_dir/<paper.paper_id>.pdf. Returns the path.

    Thin wrapper retained for backward compat — callers should prefer
    download_pdf_from_url when they need explicit URL/filename control.
    """
    return download_pdf_from_url(
        url=paper.pdf_url,
        dest_dir=dest_dir,
        filename=f"{paper.paper_id}.pdf",
        timeout=timeout,
    )
```

Then update `sources/__init__.py` to export `download_pdf_from_url`:

```python
from .arxiv import Paper, ArxivPaper, search, download_pdf, download_pdf_from_url
from . import semantic_scholar as ss

__all__ = ["Paper", "ArxivPaper", "search", "download_pdf", "download_pdf_from_url", "ss"]
```

**Phase B — rewrite `pipeline.py` orchestration**

In `src/paper_distiller/pipeline.py`, update the top-level imports. Find the existing import block:

```python
from .sources.arxiv import search as arxiv_search, download_pdf
```

Replace with:

```python
from .sources.arxiv import (
    search as arxiv_search,
    download_pdf,
    download_pdf_from_url,
)
from .sources.semantic_scholar import (
    search as ss_search,
    lookup_by_arxiv_id as ss_lookup_by_arxiv_id,
    lookup_by_doi as ss_lookup_by_doi,
)
from .sources.arxiv import Paper
```

Now add three module-level helper functions BEFORE `run(cfg)` (i.e., after `_emit_summary`):

```python
def _gather_candidates(cfg: Config) -> list[Paper]:
    """Search the configured source(s); merge and dedupe."""
    query = _query_for(cfg)
    arxiv_results: list[Paper] = []
    ss_results: list[Paper] = []

    if cfg.source in ("arxiv", "both"):
        try:
            arxiv_results = arxiv_search(query, max_results=cfg.pool)
        except Exception as e:
            if cfg.verbose:
                print(f"  arxiv search failed: {e}")
            if cfg.source == "arxiv":
                raise

    if cfg.source in ("ss", "both"):
        try:
            ss_results = ss_search(query, max_results=cfg.pool,
                                     api_key=cfg.ss_api_key)
        except Exception as e:
            if cfg.verbose:
                print(f"  SS search failed: {e}")
            if cfg.source == "ss":
                raise

    return merge_candidates(arxiv_results, ss_results)


def merge_candidates(arxiv_papers: list[Paper],
                      ss_papers: list[Paper]) -> list[Paper]:
    """Dedupe across sources. Arxiv-sourced papers win when both have the same paper."""
    seen_keys: set[str] = set()

    def keys_for(p: Paper) -> list[str]:
        ks = []
        if p.arxiv_id:
            ks.append(f"arxiv:{p.arxiv_id}")
        if p.doi:
            ks.append(f"doi:{p.doi}")
        if not ks:
            ks.append(f"{p.source}:{p.paper_id}")
        return ks

    out: list[Paper] = []
    for p in list(arxiv_papers) + list(ss_papers):
        ks = keys_for(p)
        if any(k in seen_keys for k in ks):
            continue
        out.append(p)
        for k in ks:
            seen_keys.add(k)
    return out


def _fetch_with_fallback(paper: Paper, cfg: Config, tmpdir: Path) -> str:
    """Try paper.pdf_url; on failure, fall through to SS open-access; finally to ''."""
    pdf_path = None

    # Step 1: primary URL
    if paper.pdf_url:
        try:
            pdf_path = download_pdf_from_url(
                url=paper.pdf_url,
                dest_dir=tmpdir,
                filename=f"{paper.paper_id}.pdf",
                timeout=cfg.pdf_timeout_sec,
            )
        except Exception as e:
            if cfg.verbose:
                print(f"  primary PDF fetch failed for {paper.paper_id}: {e}")

    # Step 2: SS open-access fallback (only when primary failed AND paper was arxiv-sourced
    # with cross-source identity available)
    if pdf_path is None and paper.source == "arxiv" and (paper.arxiv_id or paper.doi):
        try:
            ss_record = None
            if paper.arxiv_id:
                ss_record = ss_lookup_by_arxiv_id(paper.arxiv_id,
                                                    api_key=cfg.ss_api_key)
            elif paper.doi:
                ss_record = ss_lookup_by_doi(paper.doi,
                                              api_key=cfg.ss_api_key)
            if ss_record and ss_record.open_access_pdf_url:
                if cfg.verbose:
                    print(f"  trying SS open-access PDF for {paper.paper_id}")
                pdf_path = download_pdf_from_url(
                    url=ss_record.open_access_pdf_url,
                    dest_dir=tmpdir,
                    filename=f"{paper.paper_id}-ss.pdf",
                    timeout=cfg.pdf_timeout_sec,
                )
                if cfg.verbose:
                    print(f"  SS open-access PDF fetched OK for {paper.paper_id}")
        except Exception as e:
            if cfg.verbose:
                print(f"  SS PDF fallback also failed: {e}")

    if pdf_path is None:
        return ""

    try:
        return extract_text(pdf_path)
    except Exception as e:
        if cfg.verbose:
            print(f"  PDF parse failed for {paper.paper_id}: {e}")
        return ""
```

Now replace the body of `run(cfg)` step "3. arxiv search" with a call to `_gather_candidates`. Find:

```python
    # 3. arxiv search
    candidates = arxiv_search(_query_for(cfg), max_results=cfg.pool)
    summary["candidates_found"] = len(candidates)
```

Replace with:

```python
    # 3. gather candidates from configured source(s)
    candidates = _gather_candidates(cfg)
    summary["candidates_found"] = len(candidates)
```

In the per-paper loop, **extend the dedup block** to also check DOI. Find the existing v0.2 block:

```python
    for paper in top:
        # arxiv-id-based dedup
        if not cfg.force:
            existing = store.find_by_arxiv_id(paper.arxiv_id)
            if existing is not None:
                ...
```

Wait — that's wrong. v0.2's check unconditionally calls `find_by_arxiv_id(paper.arxiv_id)`, but for SS-only papers `paper.arxiv_id` is None. Update to:

```python
    for paper in top:
        # ID-based dedup (arxiv_id first, then doi). Skips both if --force.
        if not cfg.force:
            existing = None
            if paper.arxiv_id:
                existing = store.find_by_arxiv_id(paper.arxiv_id)
                if existing is not None and cfg.verbose:
                    print(f"  skipping arxiv:{paper.arxiv_id} — "
                          f"already in articles/{existing.slug}.md")
            if existing is None and paper.doi:
                existing = store.find_by_doi(paper.doi)
                if existing is not None and cfg.verbose:
                    print(f"  skipping doi:{paper.doi} — "
                          f"already in articles/{existing.slug}.md")
            if existing is not None:
                summary["skipped_dedup"] += 1
                continue

        # Slug-based fallback (legacy entries without refs)
        from .vault.store import slugify
        if paper.arxiv_id:
            arxiv_slug = f"paper-{paper.arxiv_id}"
        else:
            arxiv_slug = f"paper-{paper.paper_id}"
        title_slug = slugify(paper.title)
        if (
            store.slug_exists("articles", arxiv_slug)
            or store.slug_exists("articles", title_slug)
        ) and not cfg.force:
            summary["skipped_dedup"] += 1
            continue
```

Finally, **replace the PDF fetch + extract block** inside the per-paper loop with a single `_fetch_with_fallback` call. Find:

```python
            full_text = ""
            try:
                pdf_path = download_pdf(paper, Path(tmpdir),
                                          timeout=cfg.pdf_timeout_sec)
                full_text = extract_text(pdf_path)
            except Exception as e:
                if cfg.verbose:
                    print(f"  PDF fetch/parse failed for {paper.arxiv_id}: {e}; using abstract.")
```

Replace with:

```python
            full_text = _fetch_with_fallback(paper, cfg, Path(tmpdir))
```

Leave the rest of the per-paper loop (distill_article call, save_entry, depth_breakdown update) UNCHANGED.

- [ ] **Step 4: Run pipeline tests, confirm pass**

```bash
.venv\Scripts\python.exe -m pytest tests/test_pipeline.py -v
```

Expected: 8 passed (5 v0.2 + 3 new).

- [ ] **Step 5: Run full suite**

```bash
.venv\Scripts\python.exe -m pytest -v
```

Expected: 61 passed (58 prior + 3 new).

- [ ] **Step 6: Commit**

```bash
git add src/paper_distiller/pipeline.py src/paper_distiller/sources/arxiv.py src/paper_distiller/sources/__init__.py tests/test_pipeline.py
git commit -m "feat(pipeline): two-source candidate gather + PDF fallback chain

_gather_candidates dispatches to arxiv + SS based on cfg.source, then
merge_candidates dedupes by arxiv_id/doi (arxiv source wins on conflict).
Per-paper loop now also checks find_by_doi for vault dedup.

_fetch_with_fallback wraps the PDF fetch: primary URL -> SS lookup +
open_access URL -> abstract-only. Verbose mode logs each fallback step.

Refactors download_pdf into download_pdf_from_url (URL-based primitive)
+ download_pdf (backward-compat wrapper)."
```

---

## Task 6: Update `distill/article.py` refs injection

**Files:**
- Modify: `src/paper_distiller/distill/article.py`

This is a small change to the refs-injection logic so SS-sourced papers (without arxiv_id) get a sensible canonical ref.

- [ ] **Step 1: Locate the existing refs injection**

In `src/paper_distiller/distill/article.py`, find the existing logic (currently in the `distill()` function, around lines 94–99):

```python
    refs = parsed.get("refs", []) or []
    if f"arxiv:{paper.arxiv_id}" not in refs:
        refs.insert(0, f"arxiv:{paper.arxiv_id}")
```

This breaks for SS-only papers (where `paper.arxiv_id` is None — `f"arxiv:{None}"` would be inserted).

- [ ] **Step 2: Replace with multi-id-aware injection**

Replace the two lines above with:

```python
    refs = parsed.get("refs", []) or []
    # Inject canonical ref(s). Priority: arxiv -> doi -> ss_paper_id.
    canonical_refs: list[str] = []
    if paper.arxiv_id:
        canonical_refs.append(f"arxiv:{paper.arxiv_id}")
    if paper.doi:
        canonical_refs.append(f"doi:{paper.doi}")
    if not canonical_refs and paper.ss_paper_id:
        canonical_refs.append(f"ss:{paper.ss_paper_id}")
    for ref in canonical_refs:
        if ref not in refs:
            refs.insert(0, ref)
    # Preserve insertion order: most-specific first (arxiv:..., doi:..., ss:...).
    # Reverse the prepended order so arxiv appears first if both arxiv+doi are added.
    refs = canonical_refs + [r for r in refs if r not in canonical_refs]
```

- [ ] **Step 3: Run existing article tests, verify no regression**

```bash
.venv\Scripts\python.exe -m pytest tests/test_distill_article.py -v
```

The existing v0.2 test `test_distill_returns_article_result` asserts `result.refs == ["arxiv:2501.00001"]`. Our `_paper()` helper in that test has `arxiv_id="2501.00001"` set, and no doi/ss_paper_id, so `canonical_refs == ["arxiv:2501.00001"]`. The final `refs = canonical_refs + [r for r in refs if r not in canonical_refs]` produces `["arxiv:2501.00001"]` (since the LLM's mocked output already includes that ref). Test still passes.

Expected: 4 passed.

- [ ] **Step 4: Run full suite**

```bash
.venv\Scripts\python.exe -m pytest -v
```

Expected: 61 passed (unchanged from Task 5).

- [ ] **Step 5: Commit**

```bash
git add src/paper_distiller/distill/article.py
git commit -m "feat(distill): refs injection respects arxiv/doi/ss priority

Previously hardcoded f'arxiv:{paper.arxiv_id}' which crashed on SS-only
papers (arxiv_id=None). Now picks the best available canonical id and
always prepends arxiv (if known) ahead of doi/ss in the refs list."
```

---

## Task 7: Version bump + CHANGELOG + tag

**Files:**
- Modify: `src/paper_distiller/__init__.py`
- Modify: `pyproject.toml`
- Modify: `tests/test_smoke.py`
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Bump `__version__`**

In `src/paper_distiller/__init__.py`, change:

```python
__version__ = "0.2.0"
```

to:

```python
__version__ = "0.3.0"
```

- [ ] **Step 2: Bump `pyproject.toml` version**

```toml
version = "0.3.0"
```

- [ ] **Step 3: Update `tests/test_smoke.py`**

Change:

```python
    assert paper_distiller.__version__ == "0.2.0"
```

to:

```python
    assert paper_distiller.__version__ == "0.3.0"
```

- [ ] **Step 4: Update `CHANGELOG.md`**

Prepend a new `[0.3.0]` section above the existing `[0.2.0]` entry:

```markdown
## [0.3.0] — 2026-05-18

### Added
- **Semantic Scholar as second paper source.** New `sources/semantic_scholar.py` module exposes `search`, `lookup_by_arxiv_id`, `lookup_by_doi` returning the unified `Paper` dataclass.
- **`--source {arxiv,ss,both}` CLI flag** (default `both`). When `both`, pipeline searches both APIs in series, then `merge_candidates` dedupes by `arxiv_id` and `doi` (arxiv-sourced wins on conflict). When `arxiv` or `ss` solo, only that source is searched and errors propagate.
- **PDF fallback chain**: if a paper's primary PDF download fails AND the paper has an arxiv id or DOI, pipeline queries SS for `openAccessPdf` and tries that URL before falling back to abstract-only.
- **`VaultStore.find_by_doi`** mirrors `find_by_arxiv_id` semantics for DOI-based vault dedup.
- **`Config.source` and `Config.ss_api_key`** (the latter read from optional `PD_SS_API_KEY` env var).
- **`download_pdf_from_url(url, dest_dir, filename, timeout)`** as the URL-based primitive used by both arxiv direct fetch and SS fallback.

### Changed
- **`ArxivPaper` → unified `Paper` dataclass.** Adds `source`, `paper_id`, `arxiv_id`, `doi`, `ss_paper_id`, `venue`, `open_access_pdf_url` fields. `ArxivPaper` is kept as a module-level alias so v0.2 imports continue to work.
- **`distill/article.py` refs injection** now prefers arxiv id, falls back to DOI, then to SS paper id — fixes a NoneType bug that would have shipped if v0.3 hadn't generalized the dataclass.
- **Pipeline dedup** checks both `find_by_arxiv_id` and `find_by_doi` (in that priority order) before the slug-based fallback.

### Internal
- 10 new unit/integration tests; total now 61 (was 51 in v0.2).
- No new runtime dependencies (Semantic Scholar API is plain HTTPS via existing `httpx`).
- `.env.example` now documents the optional `PD_SS_API_KEY`.

```

- [ ] **Step 5: Verify version consistency**

```bash
findstr /R "^version" pyproject.toml
findstr "__version__" src\paper_distiller\__init__.py
findstr "0.3.0" tests\test_smoke.py
```

All should show `0.3.0`.

- [ ] **Step 6: Run full suite one more time**

```bash
.venv\Scripts\python.exe -m pytest -v
```

Expected: 61 passed.

- [ ] **Step 7: Commit + tag**

```bash
git add src/paper_distiller/__init__.py pyproject.toml tests/test_smoke.py CHANGELOG.md
git commit -m "chore: bump version to 0.3.0 + changelog"
git tag -a v0.3.0 -m "v0.3.0 — Semantic Scholar source + PDF fallback chain"
```

- [ ] **Step 8: Verify the tag**

```bash
git tag --list -n
git show v0.3.0 --stat | head -25
git log --oneline | head -10
```

Expected:
- `v0.3.0` in tag list with message
- Recent log shows v0.3.0 chore commit at HEAD, plus the 6 task commits above it

---

## Task 8 (optional): Commit spec + plan to docs/

After tag, if the v0.3 spec and plan files (`docs/superpowers/specs/2026-05-18-paper-distiller-v0.3.0-design.md` and `docs/superpowers/plans/2026-05-18-paper-distiller-v0.3.0.md`) are still untracked, commit them:

```bash
git add docs/superpowers/specs/2026-05-18-paper-distiller-v0.3.0-design.md docs/superpowers/plans/2026-05-18-paper-distiller-v0.3.0.md
git commit -m "docs: add v0.3.0 spec + plan"
```

(This mirrors how v0.2's docs were committed in a follow-up commit after the tag.)

---

## Acceptance criteria (from spec §9)

After all 7 tasks complete and v0.3.0 is tagged:

- [ ] `pytest -v` from `G:\paper-distiller\`: 61 tests pass
- [ ] `paper-distiller --help` shows `--source {arxiv,ss,both}` flag
- [ ] `--source` accepts only `arxiv`, `ss`, `both`; invalid value → ValueError (from `load_config`)
- [ ] `VaultStore.find_by_doi` returns Entry on hit, None on miss, only scans `articles/`
- [ ] Pipeline `--source arxiv` does NOT call SS search
- [ ] Pipeline `--source both` merges arxiv + SS results, deduping by arxiv_id/doi (arxiv source wins)
- [ ] PDF fallback: arxiv-sourced paper with PDF fail + arxiv_id → SS lookup → try open_access_pdf_url
- [ ] SS-only paper with empty pdf_url → abstract-only mode (no PDF attempts)
- [ ] `ArxivPaper` alias still works for backward-compat imports
- [ ] Refs ordering: arxiv first, then doi, then ss (for papers with multiple ids)
- [ ] `__version__` and `pyproject.toml` both show `0.3.0`
- [ ] `CHANGELOG.md` has `[0.3.0]` section
- [ ] Annotated tag `v0.3.0` exists

## Self-review notes

**Spec coverage**:
- Task 1 implements §4 (unified Paper dataclass) + §11 (backward-compat alias).
- Task 2 implements §5 (semantic_scholar.py module).
- Task 3 implements §8 (find_by_doi).
- Task 4 implements §6 Config/CLI portions.
- Task 5 implements §3 architecture + §6 pipeline orchestration + §7 PDF fallback chain.
- Task 6 implements the §6 refs precedence detail.
- Task 7 implements §12 release path.
- Task 8 (optional) handles the spec/plan docs commit (analogous to v0.2's separate docs commit).

**Spec gaps covered**: §10 (limitations) is documented in the CHANGELOG, no implementation needed.

**No placeholders detected**. Every step has runnable code or commands with expected outputs.

**Type/name consistency**:
- `Paper` dataclass — used consistently across all tasks (Task 1 defines, Tasks 2/5/6 consume).
- `ArxivPaper` alias — Task 1 establishes, Task 2 module re-exports it.
- `_gather_candidates`, `merge_candidates`, `_fetch_with_fallback` — function names consistent in Task 5.
- `cfg.source`, `cfg.ss_api_key` — Task 4 defines, Task 5 consumes.
- `download_pdf_from_url` — Task 5 Phase A introduces, then Tasks 5/test fixtures use.
- `ss_lookup_by_arxiv_id`, `ss_lookup_by_doi`, `ss_search` — aliased imports in pipeline.py (Task 5), matched by mocker.patch targets in tests (Task 5 Step 1).

**Estimated total effort**: 4–6 hours across 7 tasks (+ optional 8).
