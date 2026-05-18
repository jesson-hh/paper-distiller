# paper-distiller v0.3.0 — Design

**Date**: 2026-05-18
**Author**: brainstorm session (post-v0.2.0 ship)
**Status**: design (pending implementation plan)
**Target**: `G:\paper-distiller\` (existing repo, `main` branch, baseline tag `v0.2.0`)

---

## 1. Goal

Add **Semantic Scholar** as a second paper source. Two coupled goals (P1 + P2 from brainstorm Q1):

**(P1) Extend coverage to non-arxiv papers.** arxiv indexes ~2.5M preprints (heavy ML/math/CS bias). Semantic Scholar indexes ~200M papers — covering older math journal articles, statistics-only venues, applied work that never hit arxiv. User invokes `--source ss` or `--source both` to broaden the candidate pool.

**(P2) PDF fallback chain.** When arxiv's PDF download fails (503/429, withdrawn paper, network), if the paper has an arxiv id or DOI, query Semantic Scholar's `openAccessPdf` field and try the alternative URL before falling back to abstract-only mode.

---

## 2. Context

### What just shipped (v0.2.0, tag `v0.2.0`)

- 51 tests passing, 5 in `tests/test_pipeline.py`
- `VaultStore.find_by_arxiv_id` enables precise dedup by arxiv id in vault frontmatter `refs`
- Pipeline `--force` flag bypasses both arxiv-id and slug-based dedup
- Distill threshold restored: `len(full_text) > 500` for full-pdf mode

### What v0.3.0 changes

- New `sources/semantic_scholar.py` module
- Unify `sources/arxiv.py`'s `ArxivPaper` → generic `Paper` (alias kept for backward compat)
- Pipeline coordinates two sources behind `--source {arxiv,ss,both}` (default `both`)
- New `VaultStore.find_by_doi` for DOI-based dedup (alongside existing `find_by_arxiv_id`)
- Per-paper PDF fetch tries arxiv first, then SS open-access, then abstract-only

### Out of scope for v0.3.0

| Not doing | Reason / where |
|---|---|
| Venue / citation count in article frontmatter (P3) | Defer until we know how Dataview consumes it — separate v0.x |
| `find_by_ss_paper_id` | DOI covers most non-arxiv papers; SS-only ids are rare |
| Caching SS API responses | Each run is ~5–10 API calls; well under rate limits without caching |
| Filtering candidates by venue / citation count | Out of pipeline scope; future "advanced filter" feature |
| LEANN integration | v0.4.0 |
| L3 multi-round research loop | v0.5.0 |

---

## 3. Architecture summary

```
                    ┌────────────────────┐
   --source arxiv → │ sources/arxiv.py   │ ─┐
                    └────────────────────┘  │
                                            ├→ merge_candidates() → ranker → loop
                    ┌────────────────────┐  │     (dedup by arxiv_id / doi)
   --source ss   → │ sources/semantic_  │ ─┘
                    │   scholar.py       │
                    └────────────────────┘
                              ↑
                              │ also used in PDF fallback (per-paper)
                              │ via lookup_by_arxiv_id() / lookup_by_doi()
```

The merge step happens only when `--source both`. Each source returns the same `Paper` dataclass.

---

## 4. Unified `Paper` dataclass

Replaces `ArxivPaper` in `src/paper_distiller/sources/arxiv.py`. The old name `ArxivPaper` is kept as a module-level alias so v0.2 tests / downstream code don't break.

```python
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class Paper:
    """A research paper, sourced from arxiv or Semantic Scholar.

    Field semantics:
      - source: which API produced this record. "arxiv" or "semanticscholar".
      - paper_id: canonical id within source (arxiv_id for arxiv, paperId for SS).
      - arxiv_id / doi / ss_paper_id: cross-source identity. Set when known;
        at least one is always set. Used for dedup and PDF fallback lookups.
      - venue / open_access_pdf_url: SS-provided enrichment. None when source="arxiv".
    """
    source: str
    paper_id: str
    title: str
    authors: list
    abstract: str
    published: str
    pdf_url: str
    categories: list = field(default_factory=list)

    # Cross-source identity (at least one set)
    arxiv_id: str | None = None
    doi: str | None = None
    ss_paper_id: str | None = None

    # SS-only enrichment
    venue: str | None = None
    open_access_pdf_url: str | None = None


# Backward-compat alias
ArxivPaper = Paper
```

For arxiv-sourced papers:
- `source = "arxiv"`, `paper_id = arxiv_id = "2503.04164"` (same value)
- `ss_paper_id`, `doi`, `venue`, `open_access_pdf_url` all None

For SS-sourced papers:
- `source = "semanticscholar"`, `paper_id = ss_paper_id = <SS's paperId>`
- `arxiv_id` set if SS's `externalIds.ArXiv` present
- `doi` set if SS's `externalIds.DOI` present
- `venue` and `open_access_pdf_url` populated from SS response when available

---

## 5. `sources/semantic_scholar.py` design

### API endpoint summary

Base URL: `https://api.semanticscholar.org/graph/v1`

| Operation | Endpoint | Fields requested |
|---|---|---|
| Search | `GET /paper/search?query={q}&limit={n}&fields=title,abstract,authors,year,externalIds,openAccessPdf,venue` | as listed |
| Lookup by arxiv id | `GET /paper/ARXIV:{arxiv_id}?fields=title,abstract,authors,year,externalIds,openAccessPdf,venue` | as listed |
| Lookup by DOI | `GET /paper/DOI:{doi}?fields=title,abstract,authors,year,externalIds,openAccessPdf,venue` | as listed |

Free tier: 100 req / 5 min, ~1 req/s. Our 5-paper run uses ~5–10 SS calls.

Optional `PD_SS_API_KEY` env var: if set, sent as `x-api-key` header to raise rate limit.

### Public surface

```python
def search(query: str, max_results: int = 30, api_key: str | None = None) -> list[Paper]:
    """Search Semantic Scholar by free-text query. Returns up to max_results papers."""

def lookup_by_arxiv_id(arxiv_id: str, api_key: str | None = None) -> Paper | None:
    """Resolve an arxiv id via SS. Returns None on miss or API error."""

def lookup_by_doi(doi: str, api_key: str | None = None) -> Paper | None:
    """Resolve a DOI via SS. Returns None on miss or API error."""
```

### Internal: SS response → `Paper` conversion

```python
def _ss_record_to_paper(record: dict) -> Paper:
    external = record.get("externalIds") or {}
    open_access = record.get("openAccessPdf") or {}
    authors_field = record.get("authors") or []
    return Paper(
        source="semanticscholar",
        paper_id=record["paperId"],
        title=(record.get("title") or "").strip(),
        authors=[a.get("name", "") for a in authors_field[:10]],
        abstract=(record.get("abstract") or "").strip(),
        published=str(record.get("year") or ""),
        pdf_url=open_access.get("url", ""),
        arxiv_id=external.get("ArXiv"),
        doi=external.get("DOI"),
        ss_paper_id=record["paperId"],
        venue=record.get("venue"),
        open_access_pdf_url=open_access.get("url"),
    )
```

### Error handling

| Failure mode | Behavior |
|---|---|
| HTTP 404 on lookup_by_arxiv_id/doi | Return None |
| HTTP 429 (rate limit) | Single retry after 5s; if still 429, return None / raise (search uses raise) |
| HTTP 5xx | Single retry after 3s; if still failing, raise SSError (caught by caller) |
| Malformed JSON / missing required fields | Skip that record (in search), or return None (in lookup) |
| Network timeout | Same as 5xx |

Default timeout: 30 s per request.

---

## 6. Pipeline source coordination

### `Config` extension

```python
@dataclass
class Config:
    # ... existing fields ...
    source: str = "both"           # "arxiv" | "ss" | "both"
    ss_api_key: str | None = None  # from PD_SS_API_KEY env var, optional
```

`load_config()` validates `source in {"arxiv", "ss", "both"}` and raises ValueError otherwise.

### CLI flag

```python
p.add_argument("--source", choices=["arxiv", "ss", "both"], default="both",
               help="Which source(s) to search. Default both.")
```

### Pipeline orchestration

Replace v0.2 pipeline's single `arxiv_search(...)` call with:

```python
def _gather_candidates(cfg: Config) -> list[Paper]:
    """Search the configured source(s) and return deduped candidate list."""
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
                raise  # arxiv-only mode: fail loudly

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
```

Notes:
- `--source both` is **tolerant**: if one source errors, the other still flows. If both fail, summary records `candidates_found=0` and pipeline exits cleanly (existing v0.1 behavior).
- `--source arxiv` and `--source ss` are **strict**: a source error propagates.

### `merge_candidates`

```python
def merge_candidates(arxiv_papers: list[Paper],
                      ss_papers: list[Paper]) -> list[Paper]:
    """Combine + dedupe. Arxiv-sourced papers take precedence when the same paper
    appears in both (their direct pdf_url is usually faster than the SS mirror)."""
    seen: dict[str, Paper] = {}

    def keys_for(p: Paper) -> list[str]:
        ks = []
        if p.arxiv_id:
            ks.append(f"arxiv:{p.arxiv_id}")
        if p.doi:
            ks.append(f"doi:{p.doi}")
        if not ks:
            ks.append(f"{p.source}:{p.paper_id}")
        return ks

    for p in arxiv_papers:
        for k in keys_for(p):
            seen.setdefault(k, p)
    for p in ss_papers:
        ks = keys_for(p)
        if any(k in seen for k in ks):
            continue  # already covered by arxiv
        seen[ks[0]] = p

    # Preserve insertion order: arxiv first (in the order arxiv returned),
    # then SS-only additions.
    out = []
    seen_objs = set()
    for p in list(arxiv_papers) + list(ss_papers):
        oid = id(p)
        if oid in seen_objs:
            continue
        ks = keys_for(p)
        if any(seen.get(k) is p for k in ks):
            out.append(p)
            seen_objs.add(oid)
    return out
```

The dedup picks the arxiv version when both sources return the same paper. Output order: arxiv block first, then SS-only papers.

### Per-paper loop dedup (extends v0.2)

```python
for paper in top:
    if not cfg.force:
        # arxiv-id check (v0.2)
        if paper.arxiv_id:
            existing = store.find_by_arxiv_id(paper.arxiv_id)
            if existing is not None:
                summary["skipped_dedup"] += 1
                if cfg.verbose:
                    print(f"  skipping arxiv:{paper.arxiv_id} — "
                          f"already in articles/{existing.slug}.md")
                continue
        # NEW: doi-id check (v0.3)
        if paper.doi:
            existing = store.find_by_doi(paper.doi)
            if existing is not None:
                summary["skipped_dedup"] += 1
                if cfg.verbose:
                    print(f"  skipping doi:{paper.doi} — "
                          f"already in articles/{existing.slug}.md")
                continue
    # ... slug-based fallback unchanged ...
    # ... PDF fetch (now with SS fallback, §7) ...
```

### Refs in saved articles

The distillation step currently injects `refs=["arxiv:<id>"]` if the LLM forgot. v0.3 extends:

```python
# In distill/article.py — after parsing LLM JSON
if paper.arxiv_id and f"arxiv:{paper.arxiv_id}" not in refs:
    refs.insert(0, f"arxiv:{paper.arxiv_id}")
elif paper.doi and f"doi:{paper.doi}" not in refs:
    refs.insert(0, f"doi:{paper.doi}")
elif paper.ss_paper_id and f"ss:{paper.ss_paper_id}" not in refs:
    refs.insert(0, f"ss:{paper.ss_paper_id}")

# Also append the secondary id if present and not already there
if paper.arxiv_id and paper.doi:
    if f"doi:{paper.doi}" not in refs:
        refs.append(f"doi:{paper.doi}")
```

A paper with both arxiv id and DOI ends up with `refs=["arxiv:X", "doi:Y"]`. Future dedup by either id works.

---

## 7. PDF fallback chain

Replace v0.2's `try download; on fail use abstract` with a 3-step chain:

```python
def _fetch_with_fallback(paper: Paper, cfg: Config, tmpdir: Path) -> str:
    """Returns full text (possibly empty if all paths fail or short).

    Sequence:
      1. Try paper.pdf_url (arxiv direct, or SS open_access for SS-sourced).
      2. If failed AND we have an alternative identifier:
         - arxiv-sourced paper → try SS lookup_by_arxiv_id → its open_access_pdf
         - ss-sourced paper → already used open_access, no further fallback
      3. If still no PDF → return "" (caller treats as abstract-only mode).
    """
    pdf_path = None

    # Step 1: primary PDF URL
    if paper.pdf_url:
        try:
            pdf_path = download_pdf_from_url(paper.pdf_url, tmpdir,
                                              filename=f"{paper.paper_id}.pdf",
                                              timeout=cfg.pdf_timeout_sec)
        except Exception as e:
            if cfg.verbose:
                print(f"  primary PDF fetch failed for {paper.paper_id}: {e}")

    # Step 2: SS fallback (only for arxiv-sourced papers with alt id)
    if pdf_path is None and paper.source == "arxiv" and (paper.arxiv_id or paper.doi):
        try:
            ss_record = None
            if paper.arxiv_id:
                ss_record = ss_lookup_by_arxiv_id(paper.arxiv_id,
                                                    api_key=cfg.ss_api_key)
            elif paper.doi:
                ss_record = ss_lookup_by_doi(paper.doi, api_key=cfg.ss_api_key)

            if ss_record and ss_record.open_access_pdf_url:
                if cfg.verbose:
                    print(f"  trying SS open-access PDF for {paper.paper_id}")
                pdf_path = download_pdf_from_url(
                    ss_record.open_access_pdf_url, tmpdir,
                    filename=f"{paper.paper_id}-ss.pdf",
                    timeout=cfg.pdf_timeout_sec,
                )
                if cfg.verbose and pdf_path:
                    print(f"  SS open-access PDF fetched OK")
        except Exception as e:
            if cfg.verbose:
                print(f"  SS PDF fallback also failed: {e}")

    # Step 3: extract or abstract-only
    if pdf_path is None:
        return ""
    try:
        return extract_text(pdf_path)
    except Exception as e:
        if cfg.verbose:
            print(f"  PDF parse failed for {paper.paper_id}: {e}")
        return ""
```

### `download_pdf_from_url` (refactor of v0.2's `download_pdf`)

Current `download_pdf(paper: ArxivPaper, dest_dir: Path, timeout: float)` takes a `Paper` and uses `paper.pdf_url`. v0.3 splits into:
- `download_pdf_from_url(url: str, dest_dir: Path, filename: str, timeout: float) -> Path` — the new lower-level utility
- `download_pdf(paper: Paper, dest_dir: Path, timeout: float) -> Path` — kept as thin wrapper for backward compat in tests

### Behavior summary

| Scenario | Step 1 | Step 2 | Result |
|---|---|---|---|
| arxiv paper, arxiv PDF works | OK | (skipped) | full-pdf |
| arxiv paper, arxiv PDF 503 + SS has open-access | fails | OK | full-pdf (via SS) |
| arxiv paper, arxiv PDF 503 + SS no open-access | fails | None | abstract-only |
| SS-only paper, open-access URL works | OK | (skipped) | full-pdf |
| SS-only paper, open-access URL 404 | fails | (skipped, no alt id) | abstract-only |
| SS-only paper, no open-access URL at all | (skipped, empty url) | (skipped) | abstract-only |

---

## 8. `find_by_doi` (new VaultStore method)

Mirror of v0.2's `find_by_arxiv_id`:

```python
def find_by_doi(self, doi: str) -> Entry | None:
    """Find an article whose `refs` frontmatter contains `doi:<doi>`.

    Mirrors find_by_arxiv_id semantics — only scans articles/, returns first match.
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

A future refactor might extract a generic `find_by_ref(ref_prefix, value)` — defer; two methods is fine.

---

## 9. Acceptance criteria

After v0.3.0 ships:

- [ ] `paper-distiller --vault X --topic Y --source ss --n 1 -v` runs end-to-end on a real vault, distilling a single paper found via SS
- [ ] `paper-distiller --vault X --topic Y --source both --n 3 -v` shows the dedup happen in verbose log (when arxiv and SS overlap on at least one paper)
- [ ] `--source` accepts only `arxiv`, `ss`, `both`; invalid value → ValueError before any API call
- [ ] When `--source both` and the same paper appears in both, only one distillation happens (LLM ranker sees the arxiv version)
- [ ] An arxiv paper whose PDF returns 503 falls through to SS open-access URL (verify via verbose log on a paper known to be off-server)
- [ ] An SS-only paper with `pdf_url=""` falls through to abstract-only mode (no PDF attempts)
- [ ] `find_by_doi` returns the correct article when refs contain `doi:<doi>`; None otherwise
- [ ] Pipeline dedup catches `paper.doi` matches in vault (not just arxiv_id)
- [ ] DataView in Obsidian still renders correctly with multi-ref articles (`refs=["arxiv:X", "doi:Y"]`) — should "just work" since refs is a plain list
- [ ] All 61 tests pass (51 v0.2 + 10 new)
- [ ] `.env.example` includes optional `PD_SS_API_KEY`
- [ ] CHANGELOG `[0.3.0]` entry committed
- [ ] Annotated tag `v0.3.0`

---

## 10. Known v3 limitations + future work

| Limitation | Effect | Workaround / future |
|---|---|---|
| SS free tier (100/5min) | Heavy use could throttle | Get free API key from SS; set `PD_SS_API_KEY` |
| `find_by_ss_paper_id` not implemented | SS-only papers with no DOI / arxiv id aren't found by ref-based dedup | Slug-based fallback still applies; rare in practice |
| No caching of SS responses | Same paper looked up twice in nearby runs costs 2 API calls | Acceptable at current scale; v0.x could add disk cache |
| Venue / citation count not yet in frontmatter | Can't sort articles by venue in Obsidian | Separate v0.x (P3); needs Dataview schema design |
| Two source-search calls double the network time when `--source both` | A few extra seconds per run | Could parallelize with `asyncio.gather`; defer until measured pain |
| Refs ordering convention assumed by reader | `refs=["arxiv:X", "doi:Y"]` vs `["doi:Y", "arxiv:X"]` semantically equivalent but visually different | Document in README; canonical order = arxiv first, then doi, then ss |

---

## 11. Migration / compatibility

- `ArxivPaper` is preserved as an alias for `Paper`. v0.2 imports (`from paper_distiller.sources.arxiv import ArxivPaper`) keep working.
- `download_pdf(paper: Paper, dest_dir, timeout)` retains its old signature; only the internal implementation moves to call `download_pdf_from_url(url, dest_dir, filename, timeout)`.
- v0.2 `find_by_arxiv_id` unchanged; v0.3 adds `find_by_doi` alongside.
- Existing articles with only `refs=["arxiv:X"]` continue to dedup correctly. v0.3 dedup also catches DOI when present.

No breaking changes for any vault written by v0.2.

---

## 12. Implementation roadmap (for writing-plans skill)

Approximate task decomposition (~7 tasks):

1. Refactor `Paper` dataclass + `ArxivPaper` alias + update `sources/arxiv.py` `search()` to return new shape (+ unchanged tests)
2. Add `sources/semantic_scholar.py` with `search`, `lookup_by_arxiv_id`, `lookup_by_doi` + 4 new tests
3. Add `VaultStore.find_by_doi` + 2 new tests
4. Add `Config.source` + `Config.ss_api_key` + CLI `--source` flag + 1 new config test
5. Replace pipeline single-source candidate fetch with `_gather_candidates` + `merge_candidates`; extend per-paper dedup to also check DOI; refactor PDF fetch to `_fetch_with_fallback` + 3 new pipeline tests
6. Update `distill/article.py` refs injection for arxiv/doi/ss precedence (no new tests needed; existing tests still pass)
7. Version bump 0.2.0 → 0.3.0, CHANGELOG entry, tag v0.3.0

Estimated effort: 4–6 hours.
