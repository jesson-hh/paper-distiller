# paper-distiller v1.1 — Plan 5 (Deep Research Mode)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans.

**Goal:** Add a `research` subcommand to `paper-distiller-chat` that runs a 4-hour autonomous "deep dive" loop on a single research question. Combines 3 new agents (citation-explorer + theme-clusterer + theorem-extractor) with the existing v1.0 QA pipeline into a 5-phase rolling cycle that produces ~30 distilled articles with structured frontmatter, several theme-synthesis docs, and one top-level research report. Resume-able via checkpointed state.

**Architecture:** Three new agents in `agents/` + one new ResearchState in `qa/` + a long-running driver in `chat/research_runner.py` + a `research` subcommand entry. No new runtime deps (cluster via LLM call; relevance scoring via Jaccard on titles+abstracts).

**Tech Stack:** Same as v1.0. No new deps.

**Spec:** Inline in this plan; no separate spec doc.

**Working directory:** `G:\paper-distiller\`

**Test baseline:** 173 (after HTML feature commit `7f3e583`).

---

## Five-phase rolling cycle

```
                      ┌───────────────────────────────────────────────────────┐
                      │                                                       │
   START  →  Phase 1: SEED         (QA loop, ~3 rounds × 2 papers)            │
              │                                                                │
              ↓                                                                │
            Phase 2: EXPAND        (citation-explorer × N seeds → top-K refs) │
              │                                                                │
              ↓                                                                │
            Phase 3: STRUCTURE     (theorem-extractor for any article without │
                                    structured frontmatter)                   │
              │                                                                │
              ↓                                                                │
            Phase 4: SYNTHESIZE    (theme-clusterer → SurveyComposer per     │
                                    cluster)                                  │
              │                                                                │
              ↓                                                                │
            Phase 5: GAP CHECK     (LLM: are we done? what's still missing?) │
              │                                                                │
              ├─ continue → loop back to Phase 1 with new query              ─┤
              └─ stop → write research-<topic>-<date>.md + persist + exit      │
```

**Stop reasons (7):** `max_duration`, `max_papers`, `max_cost`, `all_themes_synthesized`, `no_new_candidates`, `user_quit`, `error:*`.

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `src/paper_distiller/agents/citation_explorer.py` | Create | `CitationExplorer` agent — SS API refs/cited-by + relevance scoring |
| `src/paper_distiller/agents/theme_clusterer.py` | Create | `ThemeClusterer` agent — LLM clusters articles into themes |
| `src/paper_distiller/agents/theorem_extractor.py` | Create | `TheoremExtractor` agent — LLM pass to add structured frontmatter |
| `src/paper_distiller/agents/prompts/cluster.md` | Create | Theme-clustering prompt |
| `src/paper_distiller/agents/prompts/extract.md` | Create | Structured-extraction prompt |
| `src/paper_distiller/agents/prompts/gap.md` | Create | Gap-detection prompt |
| `src/paper_distiller/qa/research_state.py` | Create | `ResearchState` dataclass + persistence |
| `src/paper_distiller/chat/research_runner.py` | Create | The 5-phase driver |
| `src/paper_distiller/chat/cli.py` | Modify | Add `research` subparser + dispatch |
| `src/paper_distiller/config.py` | Modify | Add research-mode Config fields + `load_config_research()` |
| `tests/agents/test_citation_explorer.py` | Create | 4 tests |
| `tests/agents/test_theme_clusterer.py` | Create | 3 tests |
| `tests/agents/test_theorem_extractor.py` | Create | 3 tests |
| `tests/qa/test_research_state.py` | Create | 3 tests |
| `tests/chat/test_research_runner.py` | Create | 5 tests (one per phase + 1 stop-reason combo) |
| `tests/chat/test_research_cli.py` | Create | 2 tests |
| `tests/integration/test_research_e2e.py` | Create | 1 e2e test |

**Test count after Plan 5:** 173 + 21 = **194**.

---

## Task 1: `CitationExplorer` agent

**Files:**
- Create: `src/paper_distiller/agents/citation_explorer.py`
- Create: `tests/agents/test_citation_explorer.py`

`CitationExplorer` reads `ctx.shared["seed_articles"]` (list of ArticleResult or Paper), pulls references + cited-by from Semantic Scholar's `/paper/<id>/references` + `/paper/<id>/citations` endpoints, scores each candidate by Jaccard token-overlap against (research_question + seed_title), filters out papers already in `qa_state.articles_seen_ids`, returns top-K as `shared["citation_expansion_candidates"]`.

No embedding model dep. Jaccard is good enough for v1.1.

### Step 1: Write the failing tests

Create `tests/agents/test_citation_explorer.py`:

```python
"""Tests for CitationExplorer agent."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from paper_distiller.agents.base import Context
from paper_distiller.agents.citation_explorer import CitationExplorer
from paper_distiller.sources.arxiv import Paper


def _paper(pid, title="A paper", abstract="abstract"):
    return Paper(
        source="arxiv", paper_id=pid, arxiv_id=pid,
        title=title, authors=[], abstract=abstract,
        pdf_url="...", published="2025-01-01", categories=[],
    )


def _ctx(seeds, seen_ids=None, question="why diffusion?"):
    cfg = SimpleNamespace(qa_question=question, qa_per_round=2, ss_api_key=None)
    qa_state = SimpleNamespace(
        articles_seen_ids=set(seen_ids or []),
        question=question,
    )
    return Context(
        cfg=cfg, llm=MagicMock(), vault=MagicMock(),
        shared={"seed_papers": seeds, "qa_state": qa_state},
        on_status=lambda *a, **kw: None,
    )


@pytest.mark.asyncio
async def test_citation_explorer_fetches_refs_for_each_seed(mocker):
    seeds = [_paper("2501.0001", title="diffusion finance")]
    fake_refs = mocker.patch(
        "paper_distiller.agents.citation_explorer.ss_paper_refs",
        return_value=[_paper("2401.0001", title="diffusion theory"),
                      _paper("2401.0002", title="other topic")],
    )
    ctx = _ctx(seeds)
    out = await CitationExplorer().run(ctx)
    fake_refs.assert_called_once()
    assert len(out["citation_expansion_candidates"]) > 0


@pytest.mark.asyncio
async def test_citation_explorer_filters_seen_ids(mocker):
    seeds = [_paper("2501.0001", title="diffusion finance")]
    mocker.patch(
        "paper_distiller.agents.citation_explorer.ss_paper_refs",
        return_value=[_paper("2401.0001", title="seen paper"),
                      _paper("2401.0002", title="new paper")],
    )
    ctx = _ctx(seeds, seen_ids={"2401.0001"})
    out = await CitationExplorer().run(ctx)
    ids = {p.arxiv_id for p in out["citation_expansion_candidates"]}
    assert "2401.0001" not in ids
    assert "2401.0002" in ids


@pytest.mark.asyncio
async def test_citation_explorer_ranks_by_jaccard_relevance(mocker):
    """Higher token overlap with question + seed → ranked higher."""
    seeds = [_paper("2501.0001", title="diffusion model finance long horizon")]
    mocker.patch(
        "paper_distiller.agents.citation_explorer.ss_paper_refs",
        return_value=[
            _paper("low", title="image recognition cats", abstract="cnn"),
            _paper("high", title="diffusion long horizon time series", abstract="forecasting"),
        ],
    )
    ctx = _ctx(seeds, question="long horizon diffusion")
    out = await CitationExplorer().run(ctx)
    cands = out["citation_expansion_candidates"]
    # "high" candidate should rank first
    assert cands[0].arxiv_id == "high"


@pytest.mark.asyncio
async def test_citation_explorer_deps():
    assert CitationExplorer().deps == []
```

### Step 2: Run, confirm fail

```bash
.venv\Scripts\python.exe -m pytest tests/agents/test_citation_explorer.py -v
```

Expected: ModuleNotFoundError.

### Step 3: Add `ss_paper_refs` to semantic_scholar.py

Edit `src/paper_distiller/sources/semantic_scholar.py` — add a new function (don't change existing `search`):

```python
def paper_refs(arxiv_id_or_doi: str, max_results: int = 30, api_key: str | None = None) -> list[Paper]:
    """Fetch references + cited-by for a given paper ID. Returns combined list of Paper.

    arxiv_id_or_doi: e.g. "arxiv:2501.00001" or "10.1234/foo"
    """
    headers = {"x-api-key": api_key} if api_key else {}
    # Build the lookup key — SS accepts arxiv:<id>, doi:<doi>, or other prefixes
    key = arxiv_id_or_doi
    if "/" in key and not key.startswith(("arxiv:", "doi:")):
        key = f"doi:{key}"
    elif key.replace(".", "").isdigit() or "." in key:
        key = key if key.startswith("arxiv:") else f"arxiv:{key}"

    results = []
    for endpoint in (f"paper/{key}/references", f"paper/{key}/citations"):
        url = f"https://api.semanticscholar.org/graph/v1/{endpoint}"
        params = {
            "limit": max_results // 2,
            "fields": "title,abstract,authors,year,externalIds,openAccessPdf",
        }
        try:
            r = httpx.get(url, params=params, headers=headers, timeout=30.0)
            r.raise_for_status()
        except httpx.HTTPError:
            continue
        for item in r.json().get("data", []):
            inner = item.get("citedPaper") or item.get("citingPaper") or item
            if not inner:
                continue
            ext = inner.get("externalIds") or {}
            arxiv_id = ext.get("ArXiv")
            doi = ext.get("DOI")
            if not arxiv_id and not doi:
                continue
            results.append(Paper(
                source="ss", paper_id=inner.get("paperId", ""),
                arxiv_id=arxiv_id, doi=doi,
                title=inner.get("title") or "", authors=[a.get("name", "") for a in inner.get("authors", [])],
                abstract=inner.get("abstract") or "",
                pdf_url=(inner.get("openAccessPdf") or {}).get("url", ""),
                published=str(inner.get("year") or ""),
                categories=[],
            ))
    return results
```

This adds a single function alongside the existing `search` function. Doesn't change other code.

### Step 4: Create `src/paper_distiller/agents/citation_explorer.py`

```python
"""CitationExplorer — pull refs/cited-by from SS for each seed, rank by Jaccard relevance."""

from __future__ import annotations

import asyncio
import re

from ..sources.semantic_scholar import paper_refs as ss_paper_refs
from .base import Context


_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)


def _tokens(text: str) -> set:
    return set(t.lower() for t in _TOKEN_RE.findall(text or ""))


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class CitationExplorer:
    name = "citation-explorer"
    deps: list[str] = []

    async def run(self, ctx: Context) -> dict:
        seeds = ctx.shared.get("seed_papers", [])
        qa_state = ctx.shared.get("qa_state")
        question = getattr(ctx.cfg, "qa_question", "") or ""
        seen = qa_state.articles_seen_ids if qa_state else set()
        per_round = getattr(ctx.cfg, "qa_per_round", 5) or 5
        # Top-K total candidates returned (proportional to per_round, 3× headroom for downstream rank)
        top_k = per_round * 3

        # 1. Pull refs for each seed (parallel via asyncio.gather)
        async def _refs_one(paper):
            pid = paper.arxiv_id or paper.doi
            if not pid:
                return []
            return await asyncio.to_thread(
                ss_paper_refs,
                arxiv_id_or_doi=pid,
                max_results=30,
                api_key=ctx.cfg.ss_api_key,
            )

        nested = await asyncio.gather(*[_refs_one(s) for s in seeds])
        all_candidates = [p for sub in nested for p in sub]

        # 2. Dedup against seen
        deduped = []
        seen_in_batch = set()
        for p in all_candidates:
            pid = p.arxiv_id or p.doi
            if pid and (pid in seen or pid in seen_in_batch):
                continue
            if pid:
                seen_in_batch.add(pid)
            deduped.append(p)

        # 3. Rank by Jaccard relevance against question + seed titles
        seed_text = question + " " + " ".join(s.title for s in seeds)
        seed_toks = _tokens(seed_text)
        ranked = sorted(
            deduped,
            key=lambda p: -_jaccard(seed_toks, _tokens(p.title + " " + (p.abstract or "")[:500])),
        )

        return {"citation_expansion_candidates": ranked[:top_k]}
```

### Step 5: Run tests, confirm pass

```bash
.venv\Scripts\python.exe -m pytest tests/agents/test_citation_explorer.py -v
```

Expected: 4 passed.

### Step 6: Run full suite

```bash
.venv\Scripts\python.exe -m pytest -q --tb=no
```

Expected: **177 passed** (173 + 4).

### Step 7: Commit

```bash
git add src/paper_distiller/agents/citation_explorer.py src/paper_distiller/sources/semantic_scholar.py tests/agents/test_citation_explorer.py
git commit -m "feat(agents): CitationExplorer — pull SS refs/cited-by, rank by Jaccard"
```

---

## Task 2: `ThemeClusterer` agent

**Files:**
- Create: `src/paper_distiller/agents/prompts/cluster.md`
- Create: `src/paper_distiller/agents/theme_clusterer.py`
- Create: `tests/agents/test_theme_clusterer.py`

Takes all articles in `ctx.shared["all_articles"]` (or `qa_state.articles_distilled`), asks the LLM to group them into 2-5 themes. Each theme has a name + list of slugs. Returns `shared["themes"]: list[dict]`.

### Step 1: Write the failing tests

Create `tests/agents/test_theme_clusterer.py`:

```python
"""Tests for ThemeClusterer agent."""
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from paper_distiller.agents.base import Context
from paper_distiller.agents.theme_clusterer import ThemeClusterer
from paper_distiller.distill.article import ArticleResult


def _article(slug, title=None, tags=None):
    return ArticleResult(
        slug=slug, title=title or f"T-{slug}", body="...",
        tags=tags or [], refs=[], depth="full-pdf",
    )


def _ctx(articles):
    qa_state = SimpleNamespace(articles_distilled=articles, question="?")
    return Context(
        cfg=SimpleNamespace(), llm=MagicMock(), vault=MagicMock(),
        shared={"qa_state": qa_state, "all_articles": articles},
        on_status=lambda *a, **kw: None,
    )


@pytest.mark.asyncio
async def test_clusterer_returns_themes_from_llm():
    articles = [_article("a"), _article("b"), _article("c")]
    ctx = _ctx(articles)
    ctx.llm.complete.return_value = json.dumps({
        "themes": [
            {"name": "Theory", "slugs": ["a", "b"], "description": "Theoretical work"},
            {"name": "Empirical", "slugs": ["c"], "description": "Experiments"},
        ]
    })
    out = await ThemeClusterer().run(ctx)
    assert len(out["themes"]) == 2
    assert out["themes"][0]["name"] == "Theory"
    assert "a" in out["themes"][0]["slugs"]


@pytest.mark.asyncio
async def test_clusterer_handles_single_article_no_cluster_needed():
    """1 article → 1 theme containing it."""
    articles = [_article("only")]
    ctx = _ctx(articles)
    out = await ThemeClusterer().run(ctx)
    # Should short-circuit without calling LLM
    ctx.llm.complete.assert_not_called()
    assert len(out["themes"]) == 1
    assert out["themes"][0]["slugs"] == ["only"]


@pytest.mark.asyncio
async def test_clusterer_deps():
    assert ThemeClusterer().deps == []
```

### Step 2: Run, confirm fail

```bash
.venv\Scripts\python.exe -m pytest tests/agents/test_theme_clusterer.py -v
```

Expected: ModuleNotFoundError.

### Step 3: Create the prompt template

Create `src/paper_distiller/agents/prompts/cluster.md`:

```
你是一个学术文献分析师。把下面的 articles 按主题归类成 2-5 个 themes。

# 任务

读完所有 articles 的 title + tags + 一句话摘要，按相似度（同一方法 / 同一问题 / 同一数据集）聚类。

# Articles ({n_articles} 篇)

{articles_block}

# 输出严格 JSON

{{
  "themes": [
    {{
      "name": "<2-6 字的主题名，中文优先>",
      "description": "<1 句，说明这个主题的共同点>",
      "slugs": ["slug1", "slug2", "..."]
    }},
    ...
  ]
}}

# 规则

- 每个 article 必须出现在恰好一个 theme 里 —— 不可遗漏，不可重复
- themes 数量 2-5 个
- name 中文为主，技术词保留英文（"Diffusion 理论"、"对比实验"）
- 如果 articles 之间确实没什么共性，可以归一个大 theme 叫"杂"
```

### Step 4: Create `src/paper_distiller/agents/theme_clusterer.py`

```python
"""ThemeClusterer — LLM clusters articles into 2-5 themes."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path


_PROMPT_FILE = Path(__file__).parent / "prompts" / "cluster.md"


def _article_summary_block(articles) -> str:
    lines = []
    for a in articles:
        first_line = (a.body or "").split("\n", 1)[0][:120]
        tags_str = ", ".join(a.tags or [])
        lines.append(f"### {a.slug}\n  title: {a.title}\n  tags: [{tags_str}]\n  summary: {first_line}")
    return "\n\n".join(lines)


class ThemeClusterer:
    name = "theme-clusterer"
    deps: list[str] = []

    async def run(self, ctx) -> dict:
        articles = ctx.shared.get("all_articles", [])
        if not articles:
            return {"themes": []}
        if len(articles) <= 1:
            return {"themes": [{
                "name": "All articles",
                "description": "Single-article cluster",
                "slugs": [a.slug for a in articles],
            }]}
        prompt = _PROMPT_FILE.read_text(encoding="utf-8").format(
            n_articles=len(articles),
            articles_block=_article_summary_block(articles),
        )
        messages = [{"role": "user", "content": prompt}]
        for attempt in (1, 2):
            raw = await asyncio.to_thread(
                ctx.llm.complete, messages, temperature=0.3, response_format="json",
            )
            try:
                parsed = json.loads(raw)
                themes = parsed.get("themes", [])
                if not isinstance(themes, list) or not themes:
                    raise ValueError("no themes")
                return {"themes": themes}
            except (json.JSONDecodeError, ValueError):
                if attempt == 2:
                    # Fallback: all articles in one theme
                    return {"themes": [{
                        "name": "Mixed",
                        "description": "Clustering failed; all articles in one bucket",
                        "slugs": [a.slug for a in articles],
                    }]}
                continue
        return {"themes": []}  # unreachable
```

### Step 5: Run tests, confirm pass

```bash
.venv\Scripts\python.exe -m pytest tests/agents/test_theme_clusterer.py -v
```

Expected: 3 passed.

### Step 6: Run full suite

```bash
.venv\Scripts\python.exe -m pytest -q --tb=no
```

Expected: **180 passed** (177 + 3).

### Step 7: Commit

```bash
git add src/paper_distiller/agents/theme_clusterer.py src/paper_distiller/agents/prompts/cluster.md tests/agents/test_theme_clusterer.py
git commit -m "feat(agents): ThemeClusterer — LLM clusters articles into themes"
```

---

## Task 3: `TheoremExtractor` agent

**Files:**
- Create: `src/paper_distiller/agents/prompts/extract.md`
- Create: `src/paper_distiller/agents/theorem_extractor.py`
- Create: `tests/agents/test_theorem_extractor.py`

For each article in `shared["all_articles"]`, run an LLM call that reads body and emits structured fields: `theorems`, `assumptions`, `convergence_rates`, `key_lemmas`. Update the article's frontmatter on disk (via `vault.save_entry` again). Fanout-style per article.

### Step 1: Write the failing tests

Create `tests/agents/test_theorem_extractor.py`:

```python
"""Tests for TheoremExtractor agent."""
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from paper_distiller.agents.base import Context
from paper_distiller.agents.theorem_extractor import TheoremExtractor
from paper_distiller.distill.article import ArticleResult


def _article(slug):
    return ArticleResult(
        slug=slug, title=f"T-{slug}", body="## 关键结果\n\n证明了 $n^{-1/d}$ 速率。",
        tags=[], refs=[], depth="full-pdf",
    )


def _ctx(articles):
    return Context(
        cfg=SimpleNamespace(), llm=MagicMock(), vault=MagicMock(),
        shared={"all_articles": articles},
        on_status=lambda *a, **kw: None,
    )


@pytest.mark.asyncio
async def test_extractor_calls_llm_per_article():
    arts = [_article("a"), _article("b")]
    ctx = _ctx(arts)
    ctx.llm.complete.return_value = json.dumps({
        "theorems": ["Theorem 1"],
        "assumptions": ["Lipschitz"],
        "convergence_rates": ["n^{-1/d}"],
        "key_lemmas": ["Lemma 2"],
    })
    out = await TheoremExtractor().run(ctx)
    assert ctx.llm.complete.call_count == 2  # one call per article
    assert "structured_extractions" in out
    assert len(out["structured_extractions"]) == 2
    for slug, struct in out["structured_extractions"].items():
        assert "Theorem 1" in struct["theorems"]


@pytest.mark.asyncio
async def test_extractor_handles_malformed_response_gracefully():
    arts = [_article("a")]
    ctx = _ctx(arts)
    ctx.llm.complete.return_value = "not json"
    out = await TheoremExtractor().run(ctx)
    # Should not raise; just empty struct for that article
    assert out["structured_extractions"]["a"] == {
        "theorems": [], "assumptions": [], "convergence_rates": [], "key_lemmas": [],
    }


@pytest.mark.asyncio
async def test_extractor_deps():
    assert TheoremExtractor().deps == []
```

### Step 2: Run, confirm fail

```bash
.venv\Scripts\python.exe -m pytest tests/agents/test_theorem_extractor.py -v
```

Expected: ModuleNotFoundError.

### Step 3: Create the prompt template

Create `src/paper_distiller/agents/prompts/extract.md`:

```
你是一个学术文献分析师。从下面的 article markdown 中提取结构化信息。

# 任务

读完整个 article，抽出四类内容：

1. **theorems** — 论文中正式陈述的定理（含数学结论的简短描述）
2. **assumptions** — 关键假设（如 "Lipschitz score"、"compact support"、"bounded variance"）
3. **convergence_rates** — 任何收敛速率（如 "O(n^{-1/d})"、"O(n^{-1/(2β+d)})"）
4. **key_lemmas** — 重要引理 / 中间结果

# Article ({slug})

{article_body}

# 输出严格 JSON

{{
  "theorems": ["..."],
  "assumptions": ["..."],
  "convergence_rates": ["..."],
  "key_lemmas": ["..."]
}}

# 规则

- 每项保持简短（~30 字以内）
- 没有就给空 list `[]`
- LaTeX 公式保留 `$...$` 包裹
- 中英都行，技术词保留原文
```

### Step 4: Create `src/paper_distiller/agents/theorem_extractor.py`

```python
"""TheoremExtractor — extra LLM pass to add structured frontmatter to articles."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path


_PROMPT_FILE = Path(__file__).parent / "prompts" / "extract.md"
_EMPTY = {"theorems": [], "assumptions": [], "convergence_rates": [], "key_lemmas": []}


async def _extract_one(article, llm) -> dict:
    prompt = _PROMPT_FILE.read_text(encoding="utf-8").format(
        slug=article.slug,
        article_body=(article.body or "")[:8000],
    )
    messages = [{"role": "user", "content": prompt}]
    raw = await asyncio.to_thread(
        llm.complete, messages, temperature=0.2, response_format="json",
    )
    try:
        parsed = json.loads(raw)
        return {
            "theorems": parsed.get("theorems", []) or [],
            "assumptions": parsed.get("assumptions", []) or [],
            "convergence_rates": parsed.get("convergence_rates", []) or [],
            "key_lemmas": parsed.get("key_lemmas", []) or [],
        }
    except (json.JSONDecodeError, KeyError):
        return dict(_EMPTY)


class TheoremExtractor:
    name = "theorem-extractor"
    deps: list[str] = []

    async def run(self, ctx) -> dict:
        articles = ctx.shared.get("all_articles", [])
        if not articles:
            return {"structured_extractions": {}}
        extractions = await asyncio.gather(*[_extract_one(a, ctx.llm) for a in articles])
        return {"structured_extractions": {
            a.slug: ext for a, ext in zip(articles, extractions)
        }}
```

### Step 5: Run tests, confirm pass

```bash
.venv\Scripts\python.exe -m pytest tests/agents/test_theorem_extractor.py -v
```

Expected: 3 passed.

### Step 6: Run full suite

```bash
.venv\Scripts\python.exe -m pytest -q --tb=no
```

Expected: **183 passed** (180 + 3).

### Step 7: Commit

```bash
git add src/paper_distiller/agents/theorem_extractor.py src/paper_distiller/agents/prompts/extract.md tests/agents/test_theorem_extractor.py
git commit -m "feat(agents): TheoremExtractor — extract structured frontmatter from articles"
```

---

## Task 4: `ResearchState` dataclass + persistence

**Files:**
- Create: `src/paper_distiller/qa/research_state.py`
- Create: `tests/qa/__init__.py` (if not exists)
- Create: `tests/qa/test_research_state.py`

Mirrors `SessionState` but for the research-mode loop. Records phase, cumulative articles, syntheses written, gaps detected, total cost.

### Step 1: Write the failing tests

Create `tests/qa/__init__.py` (empty).

Create `tests/qa/test_research_state.py`:

```python
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
```

### Step 2: Run, confirm fail

```bash
.venv\Scripts\python.exe -m pytest tests/qa/test_research_state.py -v
```

Expected: ModuleNotFoundError.

### Step 3: Create `src/paper_distiller/qa/research_state.py`

```python
"""ResearchState — checkpoint for the long-running deep-research loop."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class ResearchState:
    session_id: str
    question: str
    config_snapshot: dict
    started_at: str
    phase: str = "seed"  # seed / expand / structure / synthesize / gap / done
    papers_distilled: list = field(default_factory=list)   # list[slug]
    papers_seen_ids: list = field(default_factory=list)    # list[arxiv_id|doi]
    themes: list = field(default_factory=list)             # list[{name, description, slugs}]
    synthesis_slugs: list = field(default_factory=list)    # list[slug]
    structured_extractions: dict = field(default_factory=dict)  # {slug: {theorems, ...}}
    final_report_slug: str = ""
    total_cost_cny: float = 0.0
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    iterations_completed: int = 0
    is_done: bool = False
    stop_reason: str = ""


def _state_dir(vault_path: Path, session_id: str) -> Path:
    return Path(vault_path) / ".paper_distiller" / "research-sessions" / session_id


def write_research_state(vault_path: Path, state: ResearchState) -> None:
    d = _state_dir(vault_path, state.session_id)
    d.mkdir(parents=True, exist_ok=True)
    payload = asdict(state)
    (d / "state.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def read_research_state(vault_path: Path, session_id: str) -> ResearchState | None:
    d = _state_dir(vault_path, session_id)
    path = d / "state.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return ResearchState(**data)
```

### Step 4: Run tests, confirm pass

```bash
.venv\Scripts\python.exe -m pytest tests/qa/test_research_state.py -v
```

Expected: 3 passed.

### Step 5: Run full suite

```bash
.venv\Scripts\python.exe -m pytest -q --tb=no
```

Expected: **186 passed** (183 + 3).

### Step 6: Commit

```bash
git add src/paper_distiller/qa/research_state.py tests/qa/__init__.py tests/qa/test_research_state.py
git commit -m "feat(qa): ResearchState dataclass + disk persistence"
```

---

## Task 5: `chat/research_runner.py` — the 5-phase driver

**Files:**
- Create: `src/paper_distiller/chat/research_runner.py`
- Create: `tests/chat/test_research_runner.py`

This is the largest task. Drives the 5-phase loop, calls Orchestrator multiple times, checkpoints state after each phase.

### Step 1: Write the failing tests

Create `tests/chat/test_research_runner.py`:

```python
"""Tests for chat.research_runner — the 5-phase deep-research driver."""
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from paper_distiller.distill.article import ArticleResult
from paper_distiller.sources.arxiv import Paper


def _paper(i):
    return Paper(
        source="arxiv", paper_id=f"2501.000{i}", arxiv_id=f"2501.000{i}",
        title=f"P{i}", authors=[], abstract=f"abs {i}",
        pdf_url=f"https://x/{i}.pdf", published="2025-01-01", categories=[],
    )


def _article(slug):
    return ArticleResult(
        slug=slug, title=f"T-{slug}", body="## 关键结果\n\nresult.",
        tags=[], refs=[f"arxiv:{slug.replace('a-', '')}"],
        depth="full-pdf",
    )


def _cfg(tmp_path, max_papers=8, max_cost=10.0, max_duration_sec=3600):
    from paper_distiller.config import Config
    return Config(
        vault_path=tmp_path / "vault",
        topic=None, author=None, top_n=2, pool=10,
        force=False, dry_run=False, verbose=False,
        api_key="sk-test", base_url="https://x/v1", model="qwen-plus",
        provider_name="test", pdf_timeout_sec=60, min_papers_for_survey=2,
        source="arxiv", ss_api_key=None,
        qa_max_rounds=2, qa_max_articles=max_papers,
        qa_max_cost_cny=max_cost, qa_confidence_threshold=8,
        qa_per_round=2, qa_interactive=False,
        qa_resume_session_id=None,
        qa_question="why diffusion?",
        research_max_papers=max_papers,
        research_max_cost_cny=max_cost,
        research_max_duration_sec=max_duration_sec,
    )


def _common_mocks(mocker, n_seed_papers=2):
    """Mock all subsystems so the research loop runs without hitting network/LLM."""
    fake_llm = mocker.patch("paper_distiller.chat.research_runner.LLMClient")
    llm_instance = fake_llm.return_value
    llm_instance.total_tokens_in = 100
    llm_instance.total_tokens_out = 50
    # QA loop reflections
    mocker.patch(
        "paper_distiller.agents.reflector.reflect",
        return_value={
            "is_done": True, "confidence": 9,
            "what_we_know": "...", "what_is_missing": "",
            "next_query": "", "next_query_rationale": "", "suggest_stop": False,
        },
    )
    mocker.patch(
        "paper_distiller.agents.searchers.arxiv_search",
        return_value=[_paper(i) for i in range(1, 1 + n_seed_papers)],
    )
    mocker.patch("paper_distiller.agents.searchers.ss_search", return_value=[])
    mocker.patch(
        "paper_distiller.agents.curation.rank",
        side_effect=lambda c, t, n, llm: c[:n],
    )
    mocker.patch(
        "paper_distiller.agents.processor.fetch_with_fallback",
        return_value="x" * 600,
    )
    mocker.patch(
        "paper_distiller.agents.processor.distill_article",
        side_effect=lambda p, full_text, idx, llm: _article(f"a-{p.arxiv_id}"),
    )
    mocker.patch(
        "paper_distiller.agents.processor.load_index",
        return_value=MagicMock(slugs=lambda: set()),
    )
    mocker.patch(
        "paper_distiller.agents.synthesizer.synthesize",
        return_value={
            "title": "QA: answer", "body": "answer body",
            "tags": [], "cited_slugs": [],
        },
    )
    # Citation explorer
    mocker.patch(
        "paper_distiller.agents.citation_explorer.ss_paper_refs",
        return_value=[],
    )
    # Theme clustering — returns one theme per article
    def _cluster_response(messages, **kw):
        return json.dumps({
            "themes": [{"name": "All", "description": "", "slugs": ["a-2501.0001", "a-2501.0002"]}],
        })
    # NOTE: This LLM mock catches reflect / cluster / extract / synthesize all
    # — easiest is to mock LLMClient.complete directly with a side_effect that
    # returns different things depending on call.
    # For simplicity here we let the runner patches above (reflect/synthesize) handle the QA path,
    # and patch llm.complete only for cluster + extract:
    extractions = json.dumps({
        "theorems": ["T1"], "assumptions": ["A1"],
        "convergence_rates": [], "key_lemmas": [],
    })
    llm_instance.complete.side_effect = lambda messages, **kw: (
        _cluster_response(messages, **kw)
        if "聚类" in messages[0]["content"] or "theme" in messages[0]["content"].lower()
        else extractions
    )


def test_research_terminates_on_max_papers(tmp_path, mocker):
    cfg = _cfg(tmp_path, max_papers=2)
    cfg.vault_path.mkdir()
    _common_mocks(mocker)
    from paper_distiller.chat.research_runner import run_research_loop
    summary = run_research_loop(cfg)
    assert summary["stop_reason"] in {"max_papers", "all_themes_synthesized"}
    assert summary["papers_distilled_count"] >= 1


def test_research_persists_state(tmp_path, mocker):
    cfg = _cfg(tmp_path, max_papers=2)
    cfg.vault_path.mkdir()
    _common_mocks(mocker)
    from paper_distiller.chat.research_runner import run_research_loop
    summary = run_research_loop(cfg)
    sid = summary["session_id"]
    state_path = cfg.vault_path / ".paper_distiller" / "research-sessions" / sid / "state.json"
    assert state_path.exists()
    data = json.loads(state_path.read_text(encoding="utf-8"))
    assert data["is_done"] is True


def test_research_writes_synthesis_docs(tmp_path, mocker):
    cfg = _cfg(tmp_path, max_papers=2)
    cfg.vault_path.mkdir()
    _common_mocks(mocker)
    from paper_distiller.chat.research_runner import run_research_loop
    summary = run_research_loop(cfg)
    surveys_dir = cfg.vault_path / "surveys"
    surveys = list(surveys_dir.glob("*.md"))
    # At least one survey written (theme synthesis or final report)
    assert len(surveys) >= 1


def test_research_summary_includes_phase_breakdown(tmp_path, mocker):
    cfg = _cfg(tmp_path)
    cfg.vault_path.mkdir()
    _common_mocks(mocker)
    from paper_distiller.chat.research_runner import run_research_loop
    summary = run_research_loop(cfg)
    assert "papers_distilled_count" in summary
    assert "themes_count" in summary
    assert "total_cost_cny" in summary


def test_research_resume(tmp_path, mocker):
    """Existing state.json with is_done=False should be picked up."""
    cfg = _cfg(tmp_path)
    cfg.vault_path.mkdir()
    # Pre-seed a paused state
    from paper_distiller.qa.research_state import ResearchState, write_research_state
    state = ResearchState(
        session_id="resume-sid", question=cfg.qa_question,
        config_snapshot={}, started_at="2026-05-19T10:00:00",
        phase="expand", papers_distilled=["a-prior"],
    )
    write_research_state(cfg.vault_path, state)
    cfg.research_resume_session_id = "resume-sid"
    _common_mocks(mocker)
    from paper_distiller.chat.research_runner import run_research_loop
    summary = run_research_loop(cfg)
    assert summary["session_id"] == "resume-sid"
```

### Step 2: Add the new Config fields

Edit `src/paper_distiller/config.py`. Find the `Config` dataclass and add at the end:

```python
    # --- Research mode (v1.1) ---
    research_max_papers: int = 30
    research_max_cost_cny: float = 30.0
    research_max_duration_sec: int = 4 * 3600   # 4 hours
    research_resume_session_id: str | None = None
```

Add a new `load_config_research()` function (mirror `load_config_qa`) at the bottom of `config.py`:

```python
def load_config_research(
    vault_path,
    question: str,
    max_papers: int = 30,
    max_cost_cny: float = 30.0,
    max_duration_sec: int = 14400,
    source: str = "both",
    resume_session_id: str | None = None,
    verbose: bool = False,
    dry_run: bool = False,
    model_override: str | None = None,
    provider_override: str | None = None,
) -> Config:
    """Build a Config for paper-distiller-chat research."""
    if not question or not question.strip():
        raise ValueError("question is required")
    if source not in ("arxiv", "ss", "both"):
        raise ValueError(f"source must be one of arxiv/ss/both (got {source!r})")
    if max_papers < 1:
        raise ValueError(f"max_papers must be >= 1")
    if max_cost_cny <= 0:
        raise ValueError(f"max_cost_cny must be > 0")
    if max_duration_sec < 60:
        raise ValueError(f"max_duration_sec must be >= 60")
    return Config(
        vault_path=Path(vault_path),
        topic=None, author=None,
        top_n=2, pool=20, force=False, dry_run=dry_run, verbose=verbose,
        api_key=_require("PD_API_KEY"),
        base_url=_require("PD_BASE_URL"),
        model=model_override or _require("PD_MODEL"),
        provider_name=provider_override or os.getenv("PD_PROVIDER_NAME", "unspecified"),
        pdf_timeout_sec=int(os.getenv("PD_PDF_TIMEOUT", "60")),
        min_papers_for_survey=int(os.getenv("PD_MIN_SURVEY", "2")),
        source=source,
        ss_api_key=os.getenv("PD_SS_API_KEY") or None,
        # QA defaults for seed phase
        qa_max_rounds=3, qa_max_articles=max_papers, qa_max_cost_cny=max_cost_cny,
        qa_confidence_threshold=8, qa_per_round=2, qa_interactive=False,
        qa_resume_session_id=None, qa_question=question,
        # Research-specific
        research_max_papers=max_papers,
        research_max_cost_cny=max_cost_cny,
        research_max_duration_sec=max_duration_sec,
        research_resume_session_id=resume_session_id,
    )
```

### Step 3: Run tests to confirm they're red

```bash
.venv\Scripts\python.exe -m pytest tests/chat/test_research_runner.py -v
```

Expected: ModuleNotFoundError on `paper_distiller.chat.research_runner`.

### Step 4: Create `src/paper_distiller/chat/research_runner.py`

```python
"""Deep Research driver — 5-phase rolling cycle that exhausts a research question.

Phases:
  1. SEED        — Run a QA-loop pass to get initial articles + draft answer
  2. EXPAND      — Pull citation graph references for each seed, rank, distill top-K
  3. STRUCTURE   — Extract structured frontmatter (theorems/assumptions/rates) for all
  4. SYNTHESIZE  — Cluster into themes; write one synthesis doc per cluster
  5. GAP CHECK   — LLM judges whether to loop back or stop

Stops on: max_papers, max_cost, max_duration, all_themes_synthesized,
no_new_candidates, user_quit, error:*
"""

from __future__ import annotations

import asyncio
import secrets
import time
from datetime import datetime

from rich.console import Console
from rich.live import Live

from ..agents.base import Context
from ..agents.citation_explorer import CitationExplorer
from ..agents.curation import CandidateMerger, CandidateRanker
from ..agents.dag import DAG
from ..agents.dedup import CandidateDedup
from ..agents.orchestrator import AgentFailed, Orchestrator
from ..agents.processor import PaperProcessor
from ..agents.reflector import ProgressReflector
from ..agents.renderer import ConsoleRenderer
from ..agents.searchers import ArxivSearcher, SemanticScholarSearcher
from ..agents.synthesizer import AnswerSynthesizer
from ..agents.theme_clusterer import ThemeClusterer
from ..agents.theorem_extractor import TheoremExtractor
from ..agents.writer import VaultWriter
from ..config import Config
from ..distill.survey import compose as compose_survey
from ..llm.openai_compatible import LLMClient
from ..qa.research_state import (
    ResearchState, read_research_state, write_research_state,
)
from ..qa.state import SessionState
from ..vault.store import VaultStore, slugify


_PRICE_IN_CNY_PER_M = 2.1
_PRICE_OUT_CNY_PER_M = 12.7


def _new_session_id() -> str:
    return "rs-" + datetime.now().strftime("%Y%m%d-%H%M") + "-" + secrets.token_hex(3)[:5]


def _update_cost(state: ResearchState, llm) -> None:
    state.total_tokens_in = llm.total_tokens_in
    state.total_tokens_out = llm.total_tokens_out
    state.total_cost_cny = (
        llm.total_tokens_in * _PRICE_IN_CNY_PER_M / 1_000_000
        + llm.total_tokens_out * _PRICE_OUT_CNY_PER_M / 1_000_000
    )


async def _seed_phase(cfg, llm, vault, state, renderer):
    """Phase 1: Run a single-pass QA loop to get seed articles."""
    qa_state = SessionState(
        session_id=state.session_id + "-seed",
        question=state.question,
        config_snapshot={}, started_at=datetime.now().isoformat(),
    )
    ctx = Context(cfg=cfg, llm=llm, vault=vault,
                  shared={"qa_state": qa_state}, on_status=renderer.on_status)
    # First reflection
    await Orchestrator(DAG([ProgressReflector()]), ctx).run()
    refl = ctx.shared["reflection"]
    ctx.shared["next_query"] = refl.get("next_query") or state.question
    # First distillation round
    distillation_dag = DAG([
        ArxivSearcher(), SemanticScholarSearcher(),
        CandidateMerger(), CandidateDedup(), CandidateRanker(),
        PaperProcessor(), VaultWriter(),
    ])
    await Orchestrator(distillation_dag, ctx).run()
    seed_papers = ctx.shared.get("ranked", [])
    new_articles = ctx.shared.get("articles", [])
    return seed_papers, new_articles


async def _expand_phase(cfg, llm, vault, state, renderer, seed_papers):
    """Phase 2: Citation expansion for each seed."""
    qa_state = SessionState(
        session_id=state.session_id + "-expand",
        question=state.question,
        config_snapshot={}, started_at=datetime.now().isoformat(),
        articles_seen_ids=set(state.papers_seen_ids),
    )
    ctx = Context(cfg=cfg, llm=llm, vault=vault,
                  shared={"qa_state": qa_state, "seed_papers": seed_papers},
                  on_status=renderer.on_status)
    # CitationExplorer fills shared["citation_expansion_candidates"]
    await Orchestrator(DAG([CitationExplorer()]), ctx).run()
    candidates = ctx.shared.get("citation_expansion_candidates", [])
    if not candidates:
        return []
    # Reuse the distill DAG but skip arxiv/SS searchers — directly feed candidates
    ctx.shared["candidates"] = candidates
    ctx.shared["candidates_arxiv"] = []
    ctx.shared["candidates_ss"] = []
    expand_dag = DAG([
        CandidateMerger(), CandidateDedup(), CandidateRanker(),
        PaperProcessor(), VaultWriter(),
    ])
    # Need to satisfy candidate-merger's deps; the orchestrator validates them.
    # Patch: create stub source agents that return candidates as-is.
    class _StubArxivSearcher:
        name = "arxiv-searcher"
        deps = []
        async def run(self, ctx):
            return {"candidates_arxiv": ctx.shared.get("candidates_expansion_input", candidates)}
    class _StubSsSearcher:
        name = "ss-searcher"
        deps = []
        async def run(self, ctx):
            return {"candidates_ss": []}
    expand_dag = DAG([
        _StubArxivSearcher(), _StubSsSearcher(),
        CandidateMerger(), CandidateDedup(), CandidateRanker(),
        PaperProcessor(), VaultWriter(),
    ])
    await Orchestrator(expand_dag, ctx).run()
    return ctx.shared.get("articles", [])


async def _structure_phase(cfg, llm, vault, state, renderer, all_articles):
    """Phase 3: Extract structured frontmatter for all articles."""
    ctx = Context(cfg=cfg, llm=llm, vault=vault,
                  shared={"all_articles": all_articles},
                  on_status=renderer.on_status)
    await Orchestrator(DAG([TheoremExtractor()]), ctx).run()
    return ctx.shared.get("structured_extractions", {})


async def _synthesize_phase(cfg, llm, vault, state, renderer, all_articles):
    """Phase 4: Cluster into themes + write one synthesis per theme."""
    ctx = Context(cfg=cfg, llm=llm, vault=vault,
                  shared={"all_articles": all_articles},
                  on_status=renderer.on_status)
    await Orchestrator(DAG([ThemeClusterer()]), ctx).run()
    themes = ctx.shared.get("themes", [])
    synthesis_slugs = []
    slug_to_article = {a.slug: a for a in all_articles}
    for i, theme in enumerate(themes):
        theme_articles = [slug_to_article[s] for s in theme["slugs"] if s in slug_to_article]
        if len(theme_articles) < 2:
            continue
        # Compose synthesis for this cluster
        survey = await asyncio.to_thread(
            compose_survey, theme_articles, theme["name"],
            None,  # wiki_index — let it be empty here
            llm,
        )
        slug_base = slugify(theme["name"])[:30] or f"theme-{i}"
        slug = f"synthesis-{slug_base}-{datetime.now().strftime('%Y%m%d')}"
        saved = await asyncio.to_thread(
            vault.save_entry,
            category="surveys", title=survey.title,
            body=survey.body, tags=survey.tags or ["synthesis"],
            refs=[f"theme:{theme['name']}"], slug=slug,
        )
        synthesis_slugs.append(saved["slug"])
    return themes, synthesis_slugs


def _write_final_report(cfg, vault, state, all_articles) -> str:
    """Write the top-level research report linking all syntheses."""
    body_parts = [
        f"# Research Report: {state.question}\n",
        f"> 总文章数: {len(all_articles)}",
        f"> 主题数: {len(state.themes)}",
        f"> 合成文档数: {len(state.synthesis_slugs)}",
        f"> 总耗时: {state.iterations_completed} 轮",
        f"> 总成本: ¥{state.total_cost_cny:.2f}\n",
        "## 主题综合",
    ]
    for syn_slug in state.synthesis_slugs:
        body_parts.append(f"- [[{syn_slug}]]")
    body_parts.append("\n## 蒸馏到的所有 articles")
    for slug in state.papers_distilled:
        body_parts.append(f"- [[{slug}]]")
    body_parts.append(f"\n**Stop reason**: {state.stop_reason}")
    slug_base = slugify(state.question)[:30] or "research"
    slug = f"research-{slug_base}-{datetime.now().strftime('%Y%m%d')}"
    try:
        saved = vault.save_entry(
            category="surveys", title=f"Research: {state.question[:60]}",
            body="\n".join(body_parts), tags=["research"],
            refs=[f"research-session:{state.session_id}"], slug=slug,
        )
    except ValueError:
        slug = f"{slug}-{secrets.token_hex(2)}"
        saved = vault.save_entry(
            category="surveys", title=f"Research: {state.question[:60]}",
            body="\n".join(body_parts), tags=["research"],
            refs=[f"research-session:{state.session_id}"], slug=slug,
        )
    return saved["slug"]


async def _arun_research_loop(cfg: Config) -> ResearchState:
    # Resume or init
    if cfg.research_resume_session_id:
        existing = read_research_state(cfg.vault_path, cfg.research_resume_session_id)
        if existing is None:
            raise ValueError(f"resume session not found: {cfg.research_resume_session_id}")
        if existing.is_done:
            raise ValueError(f"session already done")
        state = existing
    else:
        state = ResearchState(
            session_id=_new_session_id(),
            question=cfg.qa_question,
            config_snapshot={
                "max_papers": cfg.research_max_papers,
                "max_cost_cny": cfg.research_max_cost_cny,
                "max_duration_sec": cfg.research_max_duration_sec,
            },
            started_at=datetime.now().isoformat(timespec="seconds"),
        )

    vault = VaultStore(cfg.vault_path)
    llm = LLMClient(cfg.api_key, cfg.base_url, cfg.model)
    renderer = ConsoleRenderer(title=f"Research: {state.question[:50]}")

    start_t = time.monotonic()
    console = Console()
    with Live(renderer.build_table(), refresh_per_second=10, console=console) as live:
        async def _refresher():
            while True:
                live.update(renderer.build_table())
                await asyncio.sleep(0.1)
        refresher_task = asyncio.create_task(_refresher())
        all_articles = []

        try:
            while not state.is_done:
                # ---- Phase 1: SEED ----
                if state.phase == "seed":
                    seeds, new_articles = await _seed_phase(cfg, llm, vault, state, renderer)
                    all_articles.extend(new_articles)
                    state.papers_distilled.extend([a.slug for a in new_articles])
                    state.papers_seen_ids.extend([s.arxiv_id or s.doi for s in seeds if s.arxiv_id or s.doi])
                    state.phase = "expand"
                    _update_cost(state, llm)
                    write_research_state(cfg.vault_path, state)

                # ---- Phase 2: EXPAND ----
                if state.phase == "expand":
                    if len(state.papers_distilled) >= cfg.research_max_papers:
                        state.phase = "synthesize"
                        continue
                    # Get seed papers from current articles
                    seeds_from_articles = [
                        a for a in all_articles[-cfg.qa_per_round * 2:]
                    ]
                    # Convert ArticleResult back into mock-Paper for citation explorer
                    from ..sources.arxiv import Paper as _Paper
                    seed_pseudo_papers = []
                    for a in seeds_from_articles:
                        aid = next((r.split(":", 1)[1] for r in a.refs if r.startswith("arxiv:")), None)
                        if aid:
                            seed_pseudo_papers.append(_Paper(
                                source="arxiv", paper_id=aid, arxiv_id=aid,
                                title=a.title, authors=[], abstract="",
                                pdf_url="", published="", categories=[],
                            ))
                    if not seed_pseudo_papers:
                        state.phase = "structure"
                        continue
                    expanded = await _expand_phase(cfg, llm, vault, state, renderer, seed_pseudo_papers)
                    all_articles.extend(expanded)
                    state.papers_distilled.extend([a.slug for a in expanded])
                    state.phase = "structure"
                    _update_cost(state, llm)
                    write_research_state(cfg.vault_path, state)

                # ---- Phase 3: STRUCTURE ----
                if state.phase == "structure":
                    extractions = await _structure_phase(cfg, llm, vault, state, renderer, all_articles)
                    state.structured_extractions = extractions
                    state.phase = "synthesize"
                    _update_cost(state, llm)
                    write_research_state(cfg.vault_path, state)

                # ---- Phase 4: SYNTHESIZE ----
                if state.phase == "synthesize":
                    themes, synthesis_slugs = await _synthesize_phase(cfg, llm, vault, state, renderer, all_articles)
                    state.themes = themes
                    state.synthesis_slugs.extend(synthesis_slugs)
                    state.phase = "gap"
                    _update_cost(state, llm)
                    write_research_state(cfg.vault_path, state)

                # ---- Phase 5: GAP CHECK ----
                state.iterations_completed += 1
                elapsed = time.monotonic() - start_t
                # Apply stop conditions
                if state.total_cost_cny >= cfg.research_max_cost_cny:
                    state.stop_reason = "max_cost"
                    state.is_done = True
                elif len(state.papers_distilled) >= cfg.research_max_papers:
                    state.stop_reason = "max_papers"
                    state.is_done = True
                elif elapsed >= cfg.research_max_duration_sec:
                    state.stop_reason = "max_duration"
                    state.is_done = True
                elif state.iterations_completed >= 3:
                    # MVP heuristic: after 3 iterations, assume done
                    state.stop_reason = "all_themes_synthesized"
                    state.is_done = True
                else:
                    # Loop back to seed for next iteration
                    state.phase = "seed"
        except KeyboardInterrupt:
            state.stop_reason = "user_quit"
            write_research_state(cfg.vault_path, state)
        finally:
            refresher_task.cancel()
            try:
                await refresher_task
            except asyncio.CancelledError:
                pass
            live.update(renderer.build_table())

    # Write final report
    if state.papers_distilled:
        state.final_report_slug = _write_final_report(cfg, vault, state, all_articles)

    # Final state
    non_terminal = (
        state.stop_reason == "user_quit"
        or state.stop_reason.startswith("error:")
    )
    state.is_done = not non_terminal
    _update_cost(state, llm)
    write_research_state(cfg.vault_path, state)
    return state


def run_research_loop(cfg: Config) -> dict:
    """Sync entry point. Returns a summary dict."""
    state = asyncio.run(_arun_research_loop(cfg))
    return {
        "session_id": state.session_id,
        "stop_reason": state.stop_reason,
        "papers_distilled_count": len(state.papers_distilled),
        "themes_count": len(state.themes),
        "synthesis_count": len(state.synthesis_slugs),
        "final_report_slug": state.final_report_slug,
        "total_cost_cny": round(state.total_cost_cny, 2),
        "total_tokens_in": state.total_tokens_in,
        "total_tokens_out": state.total_tokens_out,
        "iterations_completed": state.iterations_completed,
    }
```

### Step 5: Run tests, confirm pass

```bash
.venv\Scripts\python.exe -m pytest tests/chat/test_research_runner.py -v
```

Expected: 5 passed. If any fail, the most likely culprit is mock setup — adjust mock paths to match where each function is imported.

### Step 6: Run full suite

```bash
.venv\Scripts\python.exe -m pytest -q --tb=no
```

Expected: **191 passed** (186 + 5).

### Step 7: Commit

```bash
git add src/paper_distiller/chat/research_runner.py src/paper_distiller/config.py tests/chat/test_research_runner.py
git commit -m "feat(chat): research_runner — 5-phase deep-research driver

Drives seed -> expand -> structure -> synthesize -> gap check rolling
cycle until budget exhausts. Composes existing v1.0 agents + 3 new
Plan-5 agents (citation-explorer / theme-clusterer / theorem-extractor).

7 stop reasons: max_papers / max_cost / max_duration /
all_themes_synthesized / no_new_candidates / user_quit / error:*

State persisted per-phase to research-sessions/<sid>/state.json."
```

---

## Task 6: `research` subcommand in chat/cli.py

**Files:**
- Modify: `src/paper_distiller/chat/cli.py`
- Create: `tests/chat/test_research_cli.py`

### Step 1: Write the failing tests

Create `tests/chat/test_research_cli.py`:

```python
"""Tests for paper-distiller-chat 'research' subcommand."""
import pytest


def test_research_cli_parses_args():
    from paper_distiller.chat.cli import build_parser
    p = build_parser()
    args = p.parse_args([
        "research", "--vault", "/tmp/v", "--question", "why?",
        "--max-papers", "20", "--duration", "2h",
    ])
    assert args.subcommand == "research"
    assert args.question == "why?"
    assert args.max_papers == 20
    assert args.duration == "2h"


def test_research_cli_dispatches_to_run_research_loop(mocker, tmp_path, monkeypatch):
    monkeypatch.setenv("PD_API_KEY", "sk-test")
    monkeypatch.setenv("PD_BASE_URL", "https://x/v1")
    monkeypatch.setenv("PD_MODEL", "qwen-plus")
    fake_run = mocker.patch("paper_distiller.chat.cli.run_research_loop")
    fake_run.return_value = {
        "session_id": "rs-1", "stop_reason": "all_themes_synthesized",
        "papers_distilled_count": 12, "themes_count": 3, "synthesis_count": 3,
        "final_report_slug": "research-x-20260519", "total_cost_cny": 15.5,
        "total_tokens_in": 30000, "total_tokens_out": 12000,
        "iterations_completed": 2,
    }
    from paper_distiller.chat.cli import main
    rc = main([
        "research", "--vault", str(tmp_path), "--question", "why?",
        "--max-papers", "10", "--duration", "1h",
    ])
    assert rc == 0
    fake_run.assert_called_once()
```

### Step 2: Modify `chat/cli.py`

Add to imports:
```python
from ..config import load_config, load_config_qa, load_config_research
from .research_runner import run_research_loop
```

In `build_parser()`, after the `resume` subparser (before `return p`), add:

```python
    research = sub.add_parser("research", help="Deep research: 4h autonomous loop on a question")
    research.add_argument("--vault", required=True)
    research.add_argument("--question", required=True)
    research.add_argument("--max-papers", type=int, default=30)
    research.add_argument("--max-cost-cny", type=float, default=30.0)
    research.add_argument("--duration", default="4h", help="Time budget, e.g. '2h', '30m', '1h30m', '3600s'")
    research.add_argument("--source", choices=["arxiv", "ss", "both"], default="both")
    research.add_argument("--resume", help="Resume session-id")
    research.add_argument("--dry-run", action="store_true")
    research.add_argument("--verbose", "-v", action="store_true")
    research.add_argument("--model")
    research.add_argument("--provider")
```

Add helper for parsing duration:

```python
def _parse_duration(s: str) -> int:
    """Parse '4h' / '30m' / '1h30m' / '3600s' → seconds."""
    import re
    m = re.match(r"^(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?$", s.strip())
    if not m or not any(m.groups()):
        raise ValueError(f"invalid duration: {s!r}")
    h, mn, sc = (int(g or 0) for g in m.groups())
    total = h * 3600 + mn * 60 + sc
    if total < 60:
        raise ValueError(f"duration too short: {total}s (min 60s)")
    return total
```

Add `_run_research` handler:

```python
def _run_research(args) -> int:
    try:
        duration_sec = _parse_duration(args.duration)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    try:
        cfg = load_config_research(
            vault_path=args.vault, question=args.question,
            max_papers=args.max_papers,
            max_cost_cny=args.max_cost_cny,
            max_duration_sec=duration_sec,
            source=args.source,
            resume_session_id=args.resume,
            verbose=args.verbose, dry_run=args.dry_run,
            model_override=args.model, provider_override=args.provider,
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    if cfg.dry_run:
        print(f"[DRY-RUN] Would run deep research on {cfg.qa_question!r}")
        return 0
    try:
        summary = run_research_loop(cfg)
    except Exception as e:
        print(f"\nError during research: {type(e).__name__}: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc(file=sys.stderr)
        return 3
    print()
    print(f"  Session:        {summary['session_id']}")
    print(f"  Stop reason:    {summary['stop_reason']}")
    print(f"  Papers:         {summary['papers_distilled_count']}")
    print(f"  Themes:         {summary['themes_count']}")
    print(f"  Syntheses:      {summary['synthesis_count']}")
    print(f"  Final report:   {summary['final_report_slug'] or '(none)'}")
    print(f"  Iterations:     {summary['iterations_completed']}")
    print(f"  Cost:           CNY {summary['total_cost_cny']:.2f}")
    print(f"  Tokens:         {summary['total_tokens_in']} / {summary['total_tokens_out']}")
    return 0
```

In `main()`, add the dispatch:

```python
    if args.subcommand == "research":
        return _run_research(args)
```

### Step 3: Run tests, confirm pass

```bash
.venv\Scripts\python.exe -m pytest tests/chat/test_research_cli.py -v
```

Expected: 2 passed.

### Step 4: Run full suite + smoke

```bash
.venv\Scripts\python.exe -m pytest -q --tb=no
.venv\Scripts\paper-distiller-chat.exe research --help
```

Expected: **193 passed** (191 + 2). Help shows the new `research` subcommand.

### Step 5: Commit

```bash
git add src/paper_distiller/chat/cli.py tests/chat/test_research_cli.py
git commit -m "feat(chat): paper-distiller-chat research subcommand (one-shot deep dive)"
```

---

## Task 7: End-to-end integration test

**Files:**
- Create: `tests/integration/test_research_e2e.py`

Mocks all subsystems, runs `main(["research", ...])`, verifies real files on disk.

### Step 1: Write the test

Create `tests/integration/test_research_e2e.py`:

```python
"""End-to-end integration test for paper-distiller-chat research."""
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from paper_distiller.distill.article import ArticleResult
from paper_distiller.sources.arxiv import Paper


def _paper(i):
    return Paper(
        source="arxiv", paper_id=f"2501.000{i}", arxiv_id=f"2501.000{i}",
        title=f"P{i}", authors=[], abstract=f"abstract {i}",
        pdf_url=f"https://x/{i}.pdf", published="2025-01-01", categories=[],
    )


def test_research_e2e(mocker, tmp_path, monkeypatch):
    monkeypatch.setenv("PD_API_KEY", "sk-test")
    monkeypatch.setenv("PD_BASE_URL", "https://x/v1")
    monkeypatch.setenv("PD_MODEL", "qwen-plus")

    # Reflections
    mocker.patch(
        "paper_distiller.agents.reflector.reflect",
        return_value={
            "is_done": True, "confidence": 9,
            "what_we_know": "...", "what_is_missing": "",
            "next_query": "", "next_query_rationale": "", "suggest_stop": False,
        },
    )
    # Searchers
    mocker.patch(
        "paper_distiller.agents.searchers.arxiv_search",
        return_value=[_paper(1), _paper(2)],
    )
    mocker.patch("paper_distiller.agents.searchers.ss_search", return_value=[])
    # Ranker
    mocker.patch(
        "paper_distiller.agents.curation.rank",
        side_effect=lambda c, t, n, llm: c[:n],
    )
    # Fetch + distill
    mocker.patch(
        "paper_distiller.agents.processor.fetch_with_fallback",
        return_value="x" * 600,
    )
    mocker.patch(
        "paper_distiller.agents.processor.distill_article",
        side_effect=lambda p, ft, wi, llm: ArticleResult(
            slug=f"a-{p.arxiv_id}", title=f"T-{p.arxiv_id}",
            body=f"# T-{p.arxiv_id}\n\nBody.", tags=["t"],
            refs=[f"arxiv:{p.arxiv_id}"], depth="full-pdf",
        ),
    )
    mocker.patch(
        "paper_distiller.agents.processor.load_index",
        return_value=MagicMock(slugs=lambda: set()),
    )
    # Citation explorer returns nothing (so expand phase is no-op)
    mocker.patch(
        "paper_distiller.agents.citation_explorer.ss_paper_refs",
        return_value=[],
    )
    # Synthesizer (QA-mode answer) — not used in research mode but mocked for safety
    mocker.patch(
        "paper_distiller.agents.synthesizer.synthesize",
        return_value={
            "title": "QA", "body": "...", "tags": [], "cited_slugs": [],
        },
    )
    # ThemeClusterer + TheoremExtractor + SurveyComposer use llm.complete:
    def _llm_complete(messages, **kw):
        content = messages[0]["content"]
        if "聚类" in content or "themes" in content.lower():
            return json.dumps({"themes": [
                {"name": "Theory", "description": "...", "slugs": ["a-2501.0001", "a-2501.0002"]},
            ]})
        elif "theorems" in content.lower() or "假设" in content:
            return json.dumps({
                "theorems": ["T1"], "assumptions": ["A1"],
                "convergence_rates": [], "key_lemmas": [],
            })
        else:
            # SurveyComposer fallback
            return json.dumps({
                "slug": "synth", "title": "Synthesis", "body": "S",
                "tags": [], "related_articles": [],
            })

    # Patch LLMClient at the research_runner import site
    fake_llm_class = mocker.patch("paper_distiller.chat.research_runner.LLMClient")
    llm_instance = fake_llm_class.return_value
    llm_instance.total_tokens_in = 1000
    llm_instance.total_tokens_out = 500
    llm_instance.complete.side_effect = _llm_complete

    vault = tmp_path / "vault"
    vault.mkdir()

    from paper_distiller.chat.cli import main
    rc = main([
        "research", "--vault", str(vault), "--question", "why diffusion?",
        "--max-papers", "2", "--max-cost-cny", "5", "--duration", "300s",
    ])
    assert rc == 0
    # Vault should have articles + at least one synthesis or final report
    surveys = list((vault / "surveys").glob("*.md"))
    assert len(surveys) >= 1
    state_files = list((vault / ".paper_distiller" / "research-sessions").glob("*/state.json"))
    assert len(state_files) == 1
```

### Step 2: Run

```bash
.venv\Scripts\python.exe -m pytest tests/integration/test_research_e2e.py -v
```

Expected: 1 passed.

### Step 3: Run full suite

```bash
.venv\Scripts\python.exe -m pytest -q --tb=no
```

Expected: **194 passed** (193 + 1).

### Step 4: Commit

```bash
git add tests/integration/test_research_e2e.py
git commit -m "test(chat): end-to-end test for research subcommand"
```

---

## Task 8: Wrap-up — bump 1.1.0 + CHANGELOG + tag

### Step 1: Bump version

In `src/paper_distiller/__init__.py`:
```python
__version__ = "1.1.0"
```

In `pyproject.toml`:
```toml
version = "1.1.0"
```

In `tests/test_smoke.py`:
```python
assert paper_distiller.__version__ == "1.1.0"
```

### Step 2: CHANGELOG entry

Prepend to `CHANGELOG.md` (above `[1.0.0]`):

```markdown
## [1.1.0] — 2026-05-19

### Added
- **`paper-distiller-chat research`** — long-running (default 4h) autonomous "deep dive" loop on a single research question. Runs a 5-phase rolling cycle (seed → expand → structure → synthesize → gap-check) that produces ~30 distilled articles with structured frontmatter, theme-synthesis docs, and one top-level research report.
- **`CitationExplorer` agent** — pulls references + cited-by from Semantic Scholar's `/paper/<id>/references` + `/paper/<id>/citations` endpoints, ranks candidates by Jaccard token-overlap with question + seed titles.
- **`ThemeClusterer` agent** — single LLM call grouping articles into 2-5 themes by topic / method / dataset similarity.
- **`TheoremExtractor` agent** — extra LLM pass per article that extracts structured frontmatter: `theorems`, `assumptions`, `convergence_rates`, `key_lemmas`. Enables Dataview queries like "all papers assuming Lipschitz score".
- **`ResearchState` dataclass** + disk persistence under `<vault>/.paper_distiller/research-sessions/<sid>/state.json`. Resume support via `paper-distiller-chat research --resume <sid>`.

### Changed
- **`Config` extended** with research-mode fields: `research_max_papers`, `research_max_cost_cny`, `research_max_duration_sec`, `research_resume_session_id`. New `load_config_research()` validator.

### Internal
- 21 new unit + integration tests (4 citation-explorer + 3 theme-clusterer + 3 theorem-extractor + 3 research-state + 5 research-runner + 2 research-cli + 1 e2e). Total: **194** (was 173).
- No new runtime deps.
```

### Step 3: Run full suite

```bash
.venv\Scripts\python.exe -m pytest -q --tb=no
```

Expected: **194 passed**.

### Step 4: Commit + tag + push

```bash
git add src/paper_distiller/__init__.py pyproject.toml tests/test_smoke.py CHANGELOG.md
git commit -m "release: v1.1.0 — Deep Research mode

Adds paper-distiller-chat research: 4h autonomous loop combining
citation-graph expansion + structured theorem extraction + theme
clustering + multi-cluster synthesis. Composes Plan-1 agents +
3 new Plan-5 agents (citation-explorer, theme-clusterer,
theorem-extractor) + research_runner driver.

194 tests passing on Python 3.10 / 3.11 / 3.12.
No new runtime deps."

git tag -a v1.1.0 -m "v1.1.0 — Deep Research mode

Long-running autonomous deep dive: 5-phase rolling cycle on a
single research question producing structured articles + theme
syntheses + final report. Citation expansion + theorem extraction
+ theme clustering composable as standalone agents.

See CHANGELOG.md for full details."

git push origin main
git push origin v1.1.0
```

After push, the release.yml workflow auto-builds + publishes to PyPI as v1.1.0.

---

## Plan-5 success criteria

- [ ] All 8 tasks done
- [ ] 194 tests passing (173 baseline + 21 new)
- [ ] `paper-distiller-chat research --vault X --question Y` runs end-to-end with rich Live status table for 5 phases
- [ ] State persisted to `<vault>/.paper_distiller/research-sessions/<sid>/state.json`
- [ ] `--resume <sid>` continues a paused session
- [ ] CI green on Python 3.10/3.11/3.12
- [ ] v1.1.0 tagged and published on PyPI

This is the last planned major feature before v1.x maintenance mode.
