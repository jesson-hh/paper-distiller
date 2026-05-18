# paper-distiller v0.5.0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship v0.5.0 — a question-driven multi-round research loop (`paper-distiller-qa`). Given a research question, the agent autonomously plans search queries, distills relevant papers across multiple rounds, and synthesizes a cited answer document. Bounded by hard budget (rounds / articles / cost) + LLM "is_done" judgment + diminishing-returns detection.

**Architecture:** New `qa/` package alongside existing `distill/`. Composes existing L2 primitives (`pipeline.gather_candidates`, `pipeline.rank`, `pipeline.fetch_with_fallback`, `distill_article`, `vault.save_entry`) into a state-machine loop. Two new LLM prompts (reflect + answer). Persistent SessionState enables `--resume` after crash/Ctrl+C. Final output is a `surveys/qa-...md` doc with audit trail.

**Tech Stack:** Same as v0.3 — Python 3.10+, httpx, arxiv, pymupdf, python-dotenv, pytest, pytest-mock. No new runtime dependencies.

**Spec:** [docs/superpowers/specs/2026-05-18-paper-distiller-v0.5.0-design.md](../specs/2026-05-18-paper-distiller-v0.5.0-design.md)

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `src/paper_distiller/pipeline.py` | Modify | Rename `_gather_candidates` → `gather_candidates`, `_fetch_with_fallback` → `fetch_with_fallback` (still importable from old names via alias for v0.3 test back-compat) |
| `src/paper_distiller/config.py` | Modify | Add qa-related fields to `Config`; new `load_config_qa()` |
| `src/paper_distiller/qa/__init__.py` | Create | Package marker |
| `src/paper_distiller/qa/state.py` | Create | `SessionState`, `RoundRecord` dataclasses + persistence helpers |
| `src/paper_distiller/qa/reflection.py` | Create | LLM reflection call wrapper (JSON-out) |
| `src/paper_distiller/qa/answer.py` | Create | LLM answer-synthesis call wrapper (JSON-out) |
| `src/paper_distiller/qa/loop.py` | Create | Main orchestrator — state machine + termination |
| `src/paper_distiller/qa/cli.py` | Create | `paper-distiller-qa` argparse entry |
| `src/paper_distiller/qa/prompts/reflect.md` | Create | Reflection prompt template |
| `src/paper_distiller/qa/prompts/answer.md` | Create | Answer synthesis prompt template |
| `pyproject.toml` | Modify | Add `paper-distiller-qa` script entry + bump version 0.3.0 → 0.5.0 |
| `src/paper_distiller/__init__.py` | Modify | Bump `__version__` to 0.5.0 |
| `tests/test_smoke.py` | Modify | Update version assertion to 0.5.0 |
| `CHANGELOG.md` | Modify | Add `[0.5.0]` section (mention v0.4 gap) |
| `tests/test_qa_state.py` | Create | 3 tests for SessionState serialize/persist/resume |
| `tests/test_qa_reflection.py` | Create | 3 tests for reflect() call |
| `tests/test_qa_answer.py` | Create | 3 tests for synthesize() call |
| `tests/test_qa_loop.py` | Create | 5 integration tests for the loop (mocked subsystems) |
| `tests/test_qa_cli.py` | Create | 2 tests for argparse + dispatch |

**Test count**: v0.3 had 61 passing. v0.5 adds 16. Total after v0.5: **77 passing**.

**Working directory throughout this plan: `G:\paper-distiller\`.**

---

## Task 1: Pipeline helper rename + Config additions

**Files:**
- Modify: `src/paper_distiller/pipeline.py`
- Modify: `src/paper_distiller/config.py`

This task does NO new tests — it's a refactor + additive Config change. Existing v0.3 tests must continue to pass unchanged.

- [ ] **Step 1: Run baseline test suite to confirm starting point**

```bash
.venv\Scripts\python.exe -m pytest -q --tb=no
```

Expected: 61 passed.

- [ ] **Step 2: Rename pipeline helpers — promote to public**

In `src/paper_distiller/pipeline.py`, locate the three helper functions: `_gather_candidates`, `merge_candidates` (already public — no change), `_fetch_with_fallback`. Add public-name aliases below their definitions, keeping the underscore-prefixed names intact:

Find this block (around the helpers' definitions, immediately AFTER `_fetch_with_fallback`):

```python
def _fetch_with_fallback(paper: Paper, cfg: Config, tmpdir: Path) -> str:
    """..."""
    # ... existing body ...
```

Immediately AFTER that function definition (before `def run(cfg: Config) -> dict:`), add the two public aliases:

```python
# Public aliases for qa/ package (v0.5+). Old underscore names retained for
# v0.3 internal callers; both refer to the same callable.
gather_candidates = _gather_candidates
fetch_with_fallback = _fetch_with_fallback
```

`merge_candidates` is already public (no underscore prefix) — leave it.

- [ ] **Step 3: Add qa fields to Config dataclass**

In `src/paper_distiller/config.py`, find the `@dataclass class Config:` block. After the last existing field, append the qa-specific fields:

```python
    # QA loop (v0.5) — only used by paper-distiller-qa entry
    qa_max_rounds: int = 5
    qa_max_articles: int = 15
    qa_max_cost_cny: float = 20.0
    qa_confidence_threshold: int = 8
    qa_per_round: int = 2
    qa_interactive: bool = False
    qa_resume_session_id: str | None = None
    qa_question: str | None = None
```

- [ ] **Step 4: Add `load_config_qa()` function**

In the same file, AFTER the existing `load_config()` function, append:

```python
def load_config_qa(
    vault_path: Path | str,
    question: str,
    max_rounds: int = 5,
    max_articles: int = 15,
    max_cost_cny: float = 20.0,
    confidence_threshold: int = 8,
    per_round: int = 2,
    source: str = "both",
    interactive: bool = False,
    resume_session_id: str | None = None,
    verbose: bool = False,
    dry_run: bool = False,
    model_override: str | None = None,
    provider_override: str | None = None,
) -> Config:
    """Build a Config for paper-distiller-qa. Mirrors load_config() but with
    the QA-loop specific fields populated."""
    if not question or not question.strip():
        raise ValueError("question is required and must be non-empty")
    if source not in ("arxiv", "ss", "both"):
        raise ValueError(f"source must be one of arxiv/ss/both (got {source!r})")
    if max_rounds < 1:
        raise ValueError(f"max_rounds must be >= 1 (got {max_rounds})")
    if max_articles < 1:
        raise ValueError(f"max_articles must be >= 1 (got {max_articles})")
    if max_cost_cny <= 0:
        raise ValueError(f"max_cost_cny must be > 0 (got {max_cost_cny})")
    if not (0 <= confidence_threshold <= 10):
        raise ValueError(f"confidence_threshold must be in [0, 10] (got {confidence_threshold})")
    if per_round < 1:
        raise ValueError(f"per_round must be >= 1 (got {per_round})")

    return Config(
        vault_path=Path(vault_path),
        topic=None,
        author=None,
        top_n=per_round,           # repurposed: ranker top-N per round
        pool=30,
        force=False,
        dry_run=dry_run,
        verbose=verbose,
        api_key=_require("PD_API_KEY"),
        base_url=_require("PD_BASE_URL"),
        model=model_override or _require("PD_MODEL"),
        provider_name=provider_override or os.getenv("PD_PROVIDER_NAME", "unspecified"),
        pdf_timeout_sec=int(os.getenv("PD_PDF_TIMEOUT", "60")),
        min_papers_for_survey=int(os.getenv("PD_MIN_SURVEY", "2")),
        source=source,
        ss_api_key=os.getenv("PD_SS_API_KEY") or None,
        qa_max_rounds=max_rounds,
        qa_max_articles=max_articles,
        qa_max_cost_cny=max_cost_cny,
        qa_confidence_threshold=confidence_threshold,
        qa_per_round=per_round,
        qa_interactive=interactive,
        qa_resume_session_id=resume_session_id,
        qa_question=question,
    )
```

- [ ] **Step 5: Run baseline test suite — verify no regression**

```bash
.venv\Scripts\python.exe -m pytest -q --tb=no
```

Expected: 61 passed. The rename is alias-only (old names still work); the Config additions are new fields with defaults (existing constructions still work).

- [ ] **Step 6: Commit**

```bash
git add src/paper_distiller/pipeline.py src/paper_distiller/config.py
git commit -m "refactor(pipeline+config): promote helpers public + add qa Config fields

pipeline._gather_candidates / _fetch_with_fallback are aliased to public
names gather_candidates / fetch_with_fallback for v0.5 qa/ package reuse.
Old underscore names retained for v0.3 back-compat.

Config gains qa_* fields (max_rounds / max_articles / max_cost_cny /
confidence_threshold / per_round / interactive / resume_session_id /
question), all defaulting to safe values. New load_config_qa() builds
the same Config with QA-specific validation. No behavior change for
v0.3 callers."
```

---

## Task 2: `qa/state.py` — SessionState dataclass + persistence

**Files:**
- Create: `src/paper_distiller/qa/__init__.py`
- Create: `src/paper_distiller/qa/state.py`
- Create: `tests/test_qa_state.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_qa_state.py`:

```python
"""Tests for paper_distiller.qa.state — SessionState dataclass + persistence."""
import json
from pathlib import Path

import pytest

from paper_distiller.qa.state import (
    SessionState,
    RoundRecord,
    write_state,
    read_state,
)


def _make_state(question="why diffusion?"):
    return SessionState(
        session_id="20260518-2143-abc12",
        question=question,
        config_snapshot={"max_rounds": 5, "source": "both"},
        started_at="2026-05-18T21:43:00",
    )


def test_session_state_roundtrip(tmp_path: Path):
    """write_state followed by read_state returns equivalent SessionState."""
    vault = tmp_path / "vault"
    vault.mkdir()
    state = _make_state()
    state.rounds_completed = 2
    state.articles_seen_ids = {"2503.04164", "10.1/abc"}
    state.cost_cny = 0.42
    state.tokens_in_total = 5000
    state.history.append(RoundRecord(
        round=1, query="diffusion finance", rationale="seed query",
        candidates_found=10, new_articles=2,
        article_slugs=["a", "b"],
        what_we_know="some", what_is_missing="more",
        confidence=4, timestamp="2026-05-18T21:43:30",
    ))

    write_state(vault, state)
    restored = read_state(vault, state.session_id)
    assert restored.session_id == state.session_id
    assert restored.question == state.question
    assert restored.rounds_completed == 2
    assert restored.articles_seen_ids == {"2503.04164", "10.1/abc"}
    assert restored.cost_cny == 0.42
    assert len(restored.history) == 1
    assert restored.history[0].query == "diffusion finance"


def test_session_state_missing_returns_none(tmp_path: Path):
    """read_state returns None for unknown session_id."""
    vault = tmp_path / "vault"
    vault.mkdir()
    assert read_state(vault, "nonexistent-session") is None


def test_session_state_persists_articles_seen_ids_as_list(tmp_path: Path):
    """The set field is serialized as a JSON list and restored as a set."""
    vault = tmp_path / "vault"
    vault.mkdir()
    state = _make_state()
    state.articles_seen_ids = {"id1", "id2", "id3"}

    write_state(vault, state)
    on_disk_path = vault / ".paper_distiller" / "qa-sessions" / state.session_id / "state.json"
    raw = json.loads(on_disk_path.read_text(encoding="utf-8"))
    assert isinstance(raw["articles_seen_ids"], list)
    assert set(raw["articles_seen_ids"]) == {"id1", "id2", "id3"}

    restored = read_state(vault, state.session_id)
    assert isinstance(restored.articles_seen_ids, set)
    assert restored.articles_seen_ids == {"id1", "id2", "id3"}
```

- [ ] **Step 2: Run, confirm fail**

```bash
.venv\Scripts\python.exe -m pytest tests/test_qa_state.py -v
```

Expected: `ModuleNotFoundError: No module named 'paper_distiller.qa'`.

- [ ] **Step 3: Create `qa/__init__.py`**

Create `src/paper_distiller/qa/__init__.py` with content:

```python
"""Question-driven multi-round research loop (v0.5)."""
```

- [ ] **Step 4: Create `qa/state.py`**

Create `src/paper_distiller/qa/state.py`:

```python
"""SessionState dataclass + on-disk persistence for the QA loop."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class RoundRecord:
    round: int
    query: str
    rationale: str
    candidates_found: int
    new_articles: int
    article_slugs: list
    what_we_know: str
    what_is_missing: str
    confidence: int
    timestamp: str


@dataclass
class SessionState:
    session_id: str
    question: str
    config_snapshot: dict
    started_at: str

    rounds_completed: int = 0
    articles_distilled: list = field(default_factory=list)  # list of dicts (article kwargs)
    articles_seen_ids: set = field(default_factory=set)     # arxiv_id ∪ doi
    history: list = field(default_factory=list)             # list[RoundRecord]
    last_reflection: dict | None = None

    cost_cny: float = 0.0
    tokens_in_total: int = 0
    tokens_out_total: int = 0

    is_done: bool = False
    stop_reason: str = ""


def _session_dir(vault_path: Path, session_id: str) -> Path:
    return vault_path / ".paper_distiller" / "qa-sessions" / session_id


def write_state(vault_path: Path, state: SessionState) -> None:
    """Persist the latest SessionState snapshot to <vault>/.paper_distiller/qa-sessions/<sid>/state.json."""
    session_dir = _session_dir(vault_path, state.session_id)
    session_dir.mkdir(parents=True, exist_ok=True)

    payload = asdict(state)
    payload["articles_seen_ids"] = sorted(state.articles_seen_ids)
    # history items are RoundRecord dataclasses; asdict() handles nested dataclasses recursively

    (session_dir / "state.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def read_state(vault_path: Path, session_id: str) -> SessionState | None:
    """Read a previously persisted SessionState. Returns None if not found."""
    state_path = _session_dir(vault_path, session_id) / "state.json"
    if not state_path.exists():
        return None
    raw = json.loads(state_path.read_text(encoding="utf-8"))
    raw["articles_seen_ids"] = set(raw.get("articles_seen_ids") or [])
    history_raw = raw.get("history") or []
    raw["history"] = [RoundRecord(**r) for r in history_raw]
    return SessionState(**raw)
```

- [ ] **Step 5: Run tests, confirm pass**

```bash
.venv\Scripts\python.exe -m pytest tests/test_qa_state.py -v
```

Expected: 3 passed.

- [ ] **Step 6: Run full suite**

```bash
.venv\Scripts\python.exe -m pytest -q --tb=no
```

Expected: 64 passed (61 + 3).

- [ ] **Step 7: Commit**

```bash
git add src/paper_distiller/qa/__init__.py src/paper_distiller/qa/state.py tests/test_qa_state.py
git commit -m "feat(qa): SessionState + on-disk persistence (state.py)

SessionState and RoundRecord dataclasses + write_state / read_state
helpers writing to <vault>/.paper_distiller/qa-sessions/<sid>/state.json.
The set field articles_seen_ids round-trips as a sorted JSON list.
3 unit tests cover the roundtrip, missing-session, and set/list
serialization invariants."
```

---

## Task 3: `qa/prompts/reflect.md` + `qa/reflection.py`

**Files:**
- Create: `src/paper_distiller/qa/prompts/reflect.md`
- Create: `src/paper_distiller/qa/reflection.py`
- Create: `tests/test_qa_reflection.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_qa_reflection.py`:

```python
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
    llm = _llm_returning("still not json")
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
```

- [ ] **Step 2: Run, confirm fail**

```bash
.venv\Scripts\python.exe -m pytest tests/test_qa_reflection.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Create the reflection prompt template**

Create `src/paper_distiller/qa/prompts/reflect.md`:

```markdown
你是一个研究助手，正在帮用户回答一个研究问题。这是第 {round_num} 轮（最多 {max_rounds} 轮）。

# 原始问题
{question}

# 当前已蒸馏的论文（共 {n_articles} 篇）
{articles_summary}

# 前几轮搜过的 query（避免重复）
{prior_queries}

# 你的任务

判断：
1. 现在的信息**够回答原始问题吗**？给一个 0-10 的 confidence。
2. 如果够：is_done=true，简述能回答的核心点。
3. 如果不够：缺哪方面？设计下一轮搜索的 query。
4. 如果你怀疑这个问题搜不出更多有意义的内容（比如已经把领域翻烂了），suggest_stop=true。

# 输出严格 JSON（无 markdown 围栏，无前导文字）

{{
  "is_done": false,
  "confidence": 4,
  "what_we_know": "...",
  "what_is_missing": "...",
  "next_query": "...",
  "next_query_rationale": "...",
  "suggest_stop": false
}}

约束：
- next_query 必须**跟原始问题相关**——不要追求跟问题不沾边的有趣方向。
- next_query 不能跟 prior_queries 重复。
- confidence 要谨慎；只有 ≥ 8 我才会真停。宁可低一点。
- 如果 is_done=true，next_query 可以留空字符串。
```

- [ ] **Step 4: Create `qa/reflection.py`**

Create `src/paper_distiller/qa/reflection.py`:

```python
"""LLM reflection call for the QA loop.

Wraps a single LLM invocation that produces structured JSON describing the
loop's progress: whether the question is answered, what's missing, and the
next query to try if not.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..llm.openai_compatible import LLMClient


class ReflectionError(RuntimeError):
    pass


_PROMPT_FILE = Path(__file__).parent / "prompts" / "reflect.md"

_REQUIRED_KEYS = {
    "is_done", "confidence", "what_we_know", "what_is_missing",
    "next_query", "next_query_rationale", "suggest_stop",
}


def _render_prompt(
    question: str,
    articles_summary: list,
    prior_queries: list,
    round_num: int,
    max_rounds: int,
) -> str:
    if articles_summary:
        articles_block = "\n".join(f"- {s}" for s in articles_summary)
    else:
        articles_block = "(尚无已蒸馏的论文)"
    if prior_queries:
        queries_block = "\n".join(f"- {q}" for q in prior_queries)
    else:
        queries_block = "(本轮是第一次搜索)"
    return _PROMPT_FILE.read_text(encoding="utf-8").format(
        round_num=round_num,
        max_rounds=max_rounds,
        question=question,
        n_articles=len(articles_summary),
        articles_summary=articles_block,
        prior_queries=queries_block,
    )


def _parse_response(raw: str) -> dict:
    parsed = json.loads(raw)
    missing = _REQUIRED_KEYS - set(parsed.keys())
    if missing:
        raise ValueError(f"reflection JSON missing keys: {missing}")
    return parsed


def reflect(
    question: str,
    articles_summary: list,
    prior_queries: list,
    round_num: int,
    max_rounds: int,
    llm: LLMClient,
) -> dict:
    """One reflection call. Retries once on malformed JSON; raises on second failure."""
    prompt = _render_prompt(question, articles_summary, prior_queries,
                            round_num, max_rounds)
    messages = [{"role": "user", "content": prompt}]
    for attempt in (1, 2):
        raw = llm.complete(messages, temperature=0.3, response_format="json")
        try:
            return _parse_response(raw)
        except (json.JSONDecodeError, ValueError):
            if attempt == 2:
                raise ReflectionError(
                    f"reflection returned malformed JSON twice: {raw[:200]}"
                )
            # retry once
            continue
    raise ReflectionError("unreachable")  # pragma: no cover
```

- [ ] **Step 5: Run tests, confirm pass**

```bash
.venv\Scripts\python.exe -m pytest tests/test_qa_reflection.py -v
```

Expected: 3 passed.

- [ ] **Step 6: Run full suite**

```bash
.venv\Scripts\python.exe -m pytest -q --tb=no
```

Expected: 67 passed (64 + 3).

- [ ] **Step 7: Commit**

```bash
git add src/paper_distiller/qa/prompts/reflect.md src/paper_distiller/qa/reflection.py tests/test_qa_reflection.py
git commit -m "feat(qa): reflection call (prompts/reflect.md + reflection.py)

reflect() takes the current loop state (question, distilled article
summaries, prior queries, round number) and asks the LLM for a structured
JSON verdict: is_done / confidence / what_we_know / what_is_missing /
next_query / next_query_rationale / suggest_stop.

JSON parse retries once on malformed output; second failure raises
ReflectionError. The loop will then exit with stop_reason='error:...'.

3 unit tests cover happy-path parse, single-retry recovery, and the
two-failure exception."
```

---

## Task 4: `qa/prompts/answer.md` + `qa/answer.py`

**Files:**
- Create: `src/paper_distiller/qa/prompts/answer.md`
- Create: `src/paper_distiller/qa/answer.py`
- Create: `tests/test_qa_answer.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_qa_answer.py`:

```python
"""Tests for paper_distiller.qa.answer — LLM answer synthesis."""
import json
from unittest.mock import MagicMock

import pytest

from paper_distiller.qa.answer import synthesize, AnswerError
from paper_distiller.distill.article import ArticleResult


def _article(slug, body="..."):
    return ArticleResult(
        slug=slug, title=f"Title {slug}", body=body,
        tags=["t"], refs=[f"arxiv:x-{slug}"], depth="full-pdf",
    )


def test_synthesize_returns_answer_result():
    """synthesize() returns the parsed JSON answer."""
    llm = MagicMock()
    llm.complete.return_value = json.dumps({
        "title": "QA: 测试问题",
        "body": "# QA: 测试\n\n答案 [[a]] [[b]]",
        "tags": ["test"],
        "cited_slugs": ["a", "b"],
    })
    result = synthesize(
        question="为什么?",
        articles=[_article("a"), _article("b")],
        llm=llm,
    )
    assert result["title"] == "QA: 测试问题"
    assert "[[a]]" in result["body"]
    assert set(result["cited_slugs"]) == {"a", "b"}


def test_synthesize_strips_invented_wikilinks():
    """[[slug]] in body referencing slugs NOT in the articles list are stripped."""
    llm = MagicMock()
    llm.complete.return_value = json.dumps({
        "title": "T", "body": "see [[real]] and [[fake|Display]]",
        "tags": [], "cited_slugs": ["real", "fake"],
    })
    result = synthesize(
        question="q",
        articles=[_article("real")],
        llm=llm,
    )
    assert "[[real]]" in result["body"]
    assert "[[fake|Display]]" not in result["body"]
    # Display text preserved as plain text
    assert "Display" in result["body"]


def test_synthesize_raises_on_malformed_json():
    """Non-JSON LLM response raises AnswerError after one retry."""
    llm = MagicMock()
    llm.complete.side_effect = ["not json", "also not json"]
    with pytest.raises(AnswerError, match="malformed"):
        synthesize(
            question="q",
            articles=[_article("a")],
            llm=llm,
        )
```

- [ ] **Step 2: Run, confirm fail**

```bash
.venv\Scripts\python.exe -m pytest tests/test_qa_answer.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Create the answer prompt template**

Create `src/paper_distiller/qa/prompts/answer.md`:

```markdown
你正在为用户回答一个研究问题。所有相关的论文已经蒸馏好。

# 问题
{question}

# 可用的 articles（共 {n_articles} 篇）

{articles_full}

# 你的任务

综合所有 articles，写一个**完整、引用充分**的答案。

# 输出严格 JSON（无 markdown 围栏，无前导文字）

{{
  "title": "QA: <对问题的简短重述，中文>",
  "body": "<完整 markdown 答案>",
  "tags": ["...", "...", "3-7 个"],
  "cited_slugs": ["...实际在 body 里 [[link]] 引用的 slug..."]
}}

# body 结构要求

```
# QA: {{title}}

> **问题**: {question}

## 答案

[2-5 段，每段 200-400 字。引用 [[slug]] 形式 — slug 必须来自上面 articles 列表。]

## 关键发现要点

- 发现 1（来源: [[slug]]）
- 发现 2（来源: ...）
...

## 不确定 / 仍待研究

[列出还没回答的子问题。如果没有，就写"暂无明显未答的关键子问题"。]
```

约束：
- 答案的每个 claim 应有 [[link]] 支持。
- [[slug]] 必须来自上面 articles 列表 — **不要发明 slug**。
- 中文为主，技术词可保留英文。
- body 长度目标 800-2500 中文字。
```

- [ ] **Step 4: Create `qa/answer.py`**

Create `src/paper_distiller/qa/answer.py`:

```python
"""LLM answer-synthesis call for the QA loop.

Given the question + all distilled articles, produces a final markdown
answer with [[wikilink]] citations. Invented slugs (not in the articles
set) are stripped post-LLM.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from ..llm.openai_compatible import LLMClient


class AnswerError(RuntimeError):
    pass


_PROMPT_FILE = Path(__file__).parent / "prompts" / "answer.md"
_LINK_RE = re.compile(r"\[\[([^\]\|]+)(?:\|([^\]]+))?\]\]")
_REQUIRED_KEYS = {"title", "body", "tags", "cited_slugs"}


def _render_prompt(question: str, articles: list) -> str:
    if articles:
        # Cap each article body to ~12K chars to fit total under ~80K tokens
        blocks = []
        for a in articles:
            body_capped = (a.body or "")[:12000]
            blocks.append(f"### slug: {a.slug}\n### title: {a.title}\n\n{body_capped}")
        articles_full = "\n\n---\n\n".join(blocks)
    else:
        articles_full = "(没有可用 articles —— 这种情况你应在 body 中说明无法回答)"
    return _PROMPT_FILE.read_text(encoding="utf-8").format(
        question=question,
        n_articles=len(articles),
        articles_full=articles_full,
    )


def _scrub_invented_links(body: str, valid_slugs: set) -> str:
    """Strip [[slug]] / [[slug|Display]] when slug is not in valid_slugs.

    For invented links: if a display text exists, keep it as plain text;
    otherwise keep the bare slug as plain text.
    """
    def repl(m):
        slug = m.group(1).strip()
        display = m.group(2)
        if slug in valid_slugs:
            return m.group(0)
        return display if display else slug
    return _LINK_RE.sub(repl, body)


def _parse_response(raw: str) -> dict:
    parsed = json.loads(raw)
    missing = _REQUIRED_KEYS - set(parsed.keys())
    if missing:
        raise ValueError(f"answer JSON missing keys: {missing}")
    return parsed


def synthesize(question: str, articles: list, llm: LLMClient) -> dict:
    """One answer synthesis call. Retries once on malformed JSON.

    `articles` is a list of ArticleResult-like objects (must have .slug,
    .title, .body). Returns dict with keys: title, body, tags, cited_slugs.
    Invented [[wikilinks]] are scrubbed from body before return.
    """
    prompt = _render_prompt(question, articles)
    messages = [{"role": "user", "content": prompt}]
    valid_slugs = {a.slug for a in articles}

    for attempt in (1, 2):
        raw = llm.complete(messages, temperature=0.5, response_format="json")
        try:
            parsed = _parse_response(raw)
            parsed["body"] = _scrub_invented_links(parsed["body"], valid_slugs)
            return parsed
        except (json.JSONDecodeError, ValueError):
            if attempt == 2:
                raise AnswerError(
                    f"answer synthesis returned malformed JSON twice: {raw[:200]}"
                )
            continue
    raise AnswerError("unreachable")  # pragma: no cover
```

- [ ] **Step 5: Run tests, confirm pass**

```bash
.venv\Scripts\python.exe -m pytest tests/test_qa_answer.py -v
```

Expected: 3 passed.

- [ ] **Step 6: Run full suite**

```bash
.venv\Scripts\python.exe -m pytest -q --tb=no
```

Expected: 70 passed (67 + 3).

- [ ] **Step 7: Commit**

```bash
git add src/paper_distiller/qa/prompts/answer.md src/paper_distiller/qa/answer.py tests/test_qa_answer.py
git commit -m "feat(qa): answer synthesis (prompts/answer.md + answer.py)

synthesize() takes the question + all distilled articles and asks the
LLM for a JSON-shaped answer with [[wikilink]] citations. Invented
slugs (not in the article set) are scrubbed post-LLM (mirrors
distill.article._scrub_invented_links).

Bodies capped to 12K chars per article to keep total under ~80K tokens.

3 unit tests cover happy-path, hallucination scrub, malformed JSON."
```

---

## Task 5: `qa/loop.py` — the orchestrator

**Files:**
- Create: `src/paper_distiller/qa/loop.py`
- Create: `tests/test_qa_loop.py`

The largest task. Wires reflection + search + distill + answer into a state-machine loop with 7 stop reasons.

- [ ] **Step 1: Write the 5 integration tests**

Create `tests/test_qa_loop.py`:

```python
"""Integration tests for the QA loop. All subsystems mocked."""
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from paper_distiller.config import Config
from paper_distiller.qa.loop import run
from paper_distiller.distill.article import ArticleResult
from paper_distiller.sources.arxiv import Paper


def _config(tmp_path, max_rounds=5, max_articles=15, max_cost_cny=20.0,
            per_round=2, interactive=False, resume_id=None):
    return Config(
        vault_path=tmp_path / "vault",
        topic=None, author=None,
        top_n=per_round, pool=10, force=False, dry_run=False, verbose=False,
        api_key="sk-test", base_url="https://x/v1", model="qwen-plus",
        provider_name="test", pdf_timeout_sec=60, min_papers_for_survey=2,
        source="arxiv", ss_api_key=None,
        qa_max_rounds=max_rounds, qa_max_articles=max_articles,
        qa_max_cost_cny=max_cost_cny, qa_confidence_threshold=8,
        qa_per_round=per_round, qa_interactive=interactive,
        qa_resume_session_id=resume_id, qa_question="why diffusion?",
    )


def _paper(i, arxiv_id=None):
    aid = arxiv_id or f"2501.0000{i}"
    return Paper(
        source="arxiv", paper_id=aid, arxiv_id=aid,
        title=f"P{i}", authors=[], abstract=f"abstract {i}",
        pdf_url=f"https://arxiv.org/pdf/{aid}.pdf",
        published="2025-01-01", categories=[],
    )


def _article(slug):
    return ArticleResult(
        slug=slug, title=f"T-{slug}", body="b",
        tags=[], refs=[f"arxiv:{slug}"], depth="full-pdf",
    )


def _mock_reflection(rounds_data):
    """Build a side_effect list that returns the given reflection sequence."""
    def factory(*args, **kwargs):
        return rounds_data.pop(0)
    return factory


def _common_mocks(mocker, reflection_responses, distill_factory=None):
    """Mock all the subsystems used by qa.loop.run."""
    mocker.patch("paper_distiller.qa.loop.LLMClient")
    mocker.patch(
        "paper_distiller.qa.loop.reflect",
        side_effect=_mock_reflection(list(reflection_responses)),
    )
    mocker.patch(
        "paper_distiller.qa.loop.gather_candidates",
        return_value=[_paper(1), _paper(2), _paper(3)],
    )
    mocker.patch(
        "paper_distiller.qa.loop.rank",
        side_effect=lambda candidates, topic, top_n, llm: candidates[:top_n],
    )
    mocker.patch(
        "paper_distiller.qa.loop.fetch_with_fallback",
        return_value="x" * 1000,
    )

    if distill_factory is None:
        def distill_factory(paper, full_text, wiki_index, llm):
            return _article(slug=f"a-{paper.arxiv_id}")
    mocker.patch(
        "paper_distiller.qa.loop.distill_article",
        side_effect=distill_factory,
    )
    mocker.patch(
        "paper_distiller.qa.loop.synthesize",
        return_value={
            "title": "QA: Answer",
            "body": "# QA\n\nAnswer body [[a-2501.00001]]",
            "tags": ["qa"],
            "cited_slugs": ["a-2501.00001"],
        },
    )


def test_loop_terminates_on_llm_done(tmp_path, mocker):
    """Loop exits when reflection.is_done=True with confidence >= threshold."""
    cfg = _config(tmp_path)
    cfg.vault_path.mkdir()

    reflection_seq = [
        {"is_done": False, "confidence": 4, "what_we_know": "...",
         "what_is_missing": "...", "next_query": "q1",
         "next_query_rationale": "...", "suggest_stop": False},
        {"is_done": True, "confidence": 9, "what_we_know": "all clear",
         "what_is_missing": "", "next_query": "",
         "next_query_rationale": "", "suggest_stop": False},
    ]
    _common_mocks(mocker, reflection_seq)

    summary = run(cfg)
    assert summary["stop_reason"] == "llm_done"
    assert summary["rounds_completed"] == 1  # one round of distill happened before is_done check on round 2


def test_loop_terminates_on_max_rounds(tmp_path, mocker):
    """Loop exits cleanly when max_rounds is hit, even if LLM says not done."""
    cfg = _config(tmp_path, max_rounds=2)
    cfg.vault_path.mkdir()

    reflection_seq = [
        {"is_done": False, "confidence": 4, "what_we_know": "a",
         "what_is_missing": "...", "next_query": "q1",
         "next_query_rationale": "...", "suggest_stop": False},
        {"is_done": False, "confidence": 5, "what_we_know": "b",
         "what_is_missing": "...", "next_query": "q2",
         "next_query_rationale": "...", "suggest_stop": False},
        # Third reflection would normally come, but loop stops first
    ]
    _common_mocks(mocker, reflection_seq)

    summary = run(cfg)
    assert summary["stop_reason"] == "max_rounds"
    assert summary["rounds_completed"] == 2


def test_loop_terminates_on_no_candidates(tmp_path, mocker):
    """If all candidates were already seen (full dedup), stop with no_candidates."""
    cfg = _config(tmp_path)
    cfg.vault_path.mkdir()

    reflection_seq = [
        {"is_done": False, "confidence": 4, "what_we_know": "a",
         "what_is_missing": "...", "next_query": "q1",
         "next_query_rationale": "...", "suggest_stop": False},
    ]
    _common_mocks(mocker, reflection_seq)
    # All candidates have arxiv_ids in seen_ids before the loop
    # We simulate by returning the same papers twice (the second time they're all dedup'd)
    # First round distills 2; second round reflection asks for more but everything is dedup
    second_reflection = {
        "is_done": False, "confidence": 5, "what_we_know": "b",
        "what_is_missing": "...", "next_query": "q2",
        "next_query_rationale": "...", "suggest_stop": False,
    }
    mocker.patch("paper_distiller.qa.loop.reflect",
                 side_effect=[reflection_seq[0], second_reflection])
    # gather_candidates always returns the same 2 papers
    mocker.patch("paper_distiller.qa.loop.gather_candidates",
                 return_value=[_paper(1), _paper(2)])

    summary = run(cfg)
    assert summary["stop_reason"] == "no_candidates"


def test_loop_terminates_on_max_articles(tmp_path, mocker):
    """Loop exits when total distilled articles reach max_articles."""
    cfg = _config(tmp_path, max_articles=2, per_round=2)
    cfg.vault_path.mkdir()

    reflection_seq = [
        {"is_done": False, "confidence": 4, "what_we_know": "a",
         "what_is_missing": "...", "next_query": "q1",
         "next_query_rationale": "...", "suggest_stop": False},
    ]
    _common_mocks(mocker, reflection_seq)

    summary = run(cfg)
    assert summary["stop_reason"] == "max_articles"
    assert summary["articles_distilled_count"] == 2


def test_loop_persists_state_each_round(tmp_path, mocker):
    """After each round, state.json is written under .paper_distiller/qa-sessions/<sid>/."""
    cfg = _config(tmp_path, max_rounds=1)
    cfg.vault_path.mkdir()

    reflection_seq = [
        {"is_done": False, "confidence": 4, "what_we_know": "a",
         "what_is_missing": "...", "next_query": "q1",
         "next_query_rationale": "...", "suggest_stop": False},
    ]
    _common_mocks(mocker, reflection_seq)

    summary = run(cfg)

    session_id = summary["session_id"]
    state_path = (cfg.vault_path / ".paper_distiller" / "qa-sessions"
                  / session_id / "state.json")
    assert state_path.exists()
    data = json.loads(state_path.read_text(encoding="utf-8"))
    assert data["question"] == cfg.qa_question
    assert data["rounds_completed"] >= 1
    assert data["stop_reason"] == "max_rounds"
```

- [ ] **Step 2: Run, confirm fail**

```bash
.venv\Scripts\python.exe -m pytest tests/test_qa_loop.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Create `qa/loop.py`**

Create `src/paper_distiller/qa/loop.py`:

```python
"""Question-driven multi-round research loop.

Composes existing L2 primitives (gather_candidates / rank /
fetch_with_fallback / distill_article) into a state-machine loop. The
loop terminates when any of seven conditions fire (see stop_reason in
SessionState).
"""

from __future__ import annotations

import secrets
import tempfile
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from ..config import Config
from ..distill.article import distill as distill_article
from ..distill.filter import rank
from ..llm.openai_compatible import LLMClient, LLMError
from ..pipeline import gather_candidates, fetch_with_fallback
from ..vault.crosslink import load_index
from ..vault.store import VaultStore, slugify
from .answer import synthesize, AnswerError
from .reflection import reflect, ReflectionError
from .state import SessionState, RoundRecord, write_state, read_state


# qwen-plus pricing in CNY per 1M tokens (rough; only used for the cost budget
# circuit breaker, not for billing)
_PRICE_IN_CNY_PER_M = 2.1
_PRICE_OUT_CNY_PER_M = 12.7


def _new_session_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M") + "-" + secrets.token_hex(3)[:5]


def _article_summary_line(article) -> str:
    """One-line summary of an article for the reflection prompt."""
    title = article.title.replace("\n", " ").strip()[:120]
    return f"[[{article.slug}]] {title}"


def _record_round(
    state: SessionState,
    *,
    round_num: int,
    reflection: dict,
    candidates_count: int,
    distilled: list,
) -> None:
    state.history.append(RoundRecord(
        round=round_num,
        query=reflection.get("next_query", ""),
        rationale=reflection.get("next_query_rationale", ""),
        candidates_found=candidates_count,
        new_articles=len(distilled),
        article_slugs=[a.slug for a in distilled],
        what_we_know=reflection.get("what_we_know", ""),
        what_is_missing=reflection.get("what_is_missing", ""),
        confidence=int(reflection.get("confidence", 0)),
        timestamp=datetime.now().isoformat(timespec="seconds"),
    ))


def _update_cost(state: SessionState, llm: LLMClient) -> None:
    state.tokens_in_total = llm.total_tokens_in
    state.tokens_out_total = llm.total_tokens_out
    state.cost_cny = (
        llm.total_tokens_in * _PRICE_IN_CNY_PER_M / 1_000_000
        + llm.total_tokens_out * _PRICE_OUT_CNY_PER_M / 1_000_000
    )


def _audit_trail_markdown(history: list, stop_reason: str, state: SessionState) -> str:
    """Render the audit table for the survey footer."""
    rows = ["| 轮 | Query | 新增 | LLM 判断 | Confidence |",
            "|---|---|---|---|---|"]
    for r in history:
        what_missing = (r.what_is_missing or r.what_we_know or "").replace("\n", " ")[:50]
        rows.append(
            f"| {r.round} | {r.query[:40]} | {r.new_articles} | "
            f"{what_missing} | {r.confidence} |"
        )
    table = "\n".join(rows)
    footer = (
        f"\n\n**Stop reason**: {stop_reason}\n"
        f"**Rounds**: {state.rounds_completed}\n"
        f"**Articles distilled**: {len(state.articles_distilled)}\n"
        f"**Total cost**: ¥{state.cost_cny:.2f} ({state.tokens_in_total} in / "
        f"{state.tokens_out_total} out tokens)\n"
        f"**Session ID**: {state.session_id}\n"
    )
    return table + footer


def _build_survey_body(answer: dict, state: SessionState) -> str:
    """Assemble the final survey body: LLM answer + cited articles table + audit trail."""
    parts = [answer["body"]]
    cited = answer.get("cited_slugs") or []
    if cited:
        cited_rows = ["", "## 引用的 articles", "", "| Slug | 标题 |", "|---|---|"]
        slug_to_article = {a.slug: a for a in state.articles_distilled}
        for slug in cited:
            article = slug_to_article.get(slug)
            if article is not None:
                title = (article.title or "").replace("\n", " ")[:80]
                cited_rows.append(f"| [[{slug}]] | {title} |")
        parts.append("\n".join(cited_rows))
    parts.append("\n## 研究过程（audit trail）\n")
    parts.append(_audit_trail_markdown(state.history, state.stop_reason, state))
    return "\n".join(parts)


def _interactive_continue(reflection: dict, round_num: int, max_rounds: int) -> bool | str:
    """Print reflection JSON and prompt Y/n/q. Returns True (continue), False (stop)."""
    print(f"\n--- Round {round_num} / {max_rounds} reflection ---")
    print(f"  confidence: {reflection.get('confidence')}")
    print(f"  what_we_know: {reflection.get('what_we_know')}")
    print(f"  what_is_missing: {reflection.get('what_is_missing')}")
    print(f"  next_query: {reflection.get('next_query')}")
    print(f"  rationale: {reflection.get('next_query_rationale')}")
    reply = input(f"Continue to round {round_num + 1}? [Y/n/q] ").strip().lower()
    return reply in ("", "y", "yes")


def run(cfg: Config) -> dict:
    """Execute the QA loop. Returns a summary dict."""
    if cfg.qa_resume_session_id:
        existing = read_state(cfg.vault_path, cfg.qa_resume_session_id)
        if existing is None:
            raise ValueError(f"resume session not found: {cfg.qa_resume_session_id}")
        if existing.is_done:
            raise ValueError(
                f"session {cfg.qa_resume_session_id} already done "
                f"(stop_reason={existing.stop_reason!r}); cannot resume"
            )
        state = existing
    else:
        state = SessionState(
            session_id=_new_session_id(),
            question=cfg.qa_question,
            config_snapshot={
                "max_rounds": cfg.qa_max_rounds,
                "max_articles": cfg.qa_max_articles,
                "max_cost_cny": cfg.qa_max_cost_cny,
                "confidence_threshold": cfg.qa_confidence_threshold,
                "per_round": cfg.qa_per_round,
                "source": cfg.source,
            },
            started_at=datetime.now().isoformat(timespec="seconds"),
        )

    if cfg.dry_run:
        print(f"[DRY-RUN] Would run QA loop for question: {cfg.qa_question!r}")
        return {
            "session_id": state.session_id,
            "stop_reason": "dry_run",
            "rounds_completed": 0,
            "articles_distilled_count": 0,
        }

    store = VaultStore(cfg.vault_path)
    wiki_index = load_index(store)
    llm = LLMClient(cfg.api_key, cfg.base_url, cfg.model)

    prior_queries: list = [r.query for r in state.history if r.query]

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        try:
            while True:
                # 1. reflection
                articles_summary = [
                    _article_summary_line(a) for a in state.articles_distilled
                ]
                round_num = state.rounds_completed + 1
                try:
                    reflection = reflect(
                        question=state.question,
                        articles_summary=articles_summary,
                        prior_queries=prior_queries,
                        round_num=round_num,
                        max_rounds=cfg.qa_max_rounds,
                        llm=llm,
                    )
                except ReflectionError as e:
                    state.stop_reason = f"error: reflection failed: {e}"
                    break
                state.last_reflection = reflection
                _update_cost(state, llm)

                # 2. termination checks
                if state.rounds_completed >= cfg.qa_max_rounds:
                    state.stop_reason = "max_rounds"
                    break
                if reflection.get("is_done") and \
                        int(reflection.get("confidence", 0)) >= cfg.qa_confidence_threshold:
                    state.stop_reason = "llm_done"
                    break
                if reflection.get("suggest_stop"):
                    state.stop_reason = "llm_brake"
                    break

                # 3. interactive checkpoint
                if cfg.qa_interactive:
                    if not _interactive_continue(reflection, round_num, cfg.qa_max_rounds):
                        state.stop_reason = "user_quit"
                        break

                # 4. search
                next_query = reflection.get("next_query") or ""
                if not next_query:
                    state.stop_reason = "no_candidates"  # empty query is a degenerate signal
                    break
                try:
                    cfg.topic = next_query  # the cfg.topic field drives gather_candidates
                    candidates = gather_candidates(cfg)
                except Exception as e:
                    state.stop_reason = f"error: search failed: {e}"
                    break

                # 5. dedup against seen
                new_candidates = []
                for p in candidates:
                    pid = p.arxiv_id or p.doi
                    if pid and pid in state.articles_seen_ids:
                        continue
                    new_candidates.append(p)

                if not new_candidates:
                    state.stop_reason = "no_candidates"
                    break

                # 6. rank top per_round
                try:
                    top = rank(new_candidates, state.question,
                                cfg.qa_per_round, llm)
                except Exception as e:
                    state.stop_reason = f"error: ranker failed: {e}"
                    break

                # 7. distill loop
                distilled_this_round = []
                for paper in top:
                    full_text = fetch_with_fallback(paper, cfg, tmpdir_path)
                    try:
                        article = distill_article(paper, full_text, wiki_index, llm)
                    except LLMError as e:
                        if cfg.verbose:
                            print(f"  distill failed for {paper.arxiv_id}: {e}")
                        continue
                    store.save_entry(category="articles", **article.to_save_kwargs())
                    state.articles_distilled.append(article)
                    pid = paper.arxiv_id or paper.doi or paper.paper_id
                    if pid:
                        state.articles_seen_ids.add(pid)
                    distilled_this_round.append(article)

                # 8. record round + persist
                _record_round(
                    state, round_num=round_num, reflection=reflection,
                    candidates_count=len(candidates), distilled=distilled_this_round,
                )
                if next_query:
                    prior_queries.append(next_query)
                state.rounds_completed += 1
                _update_cost(state, llm)
                write_state(cfg.vault_path, state)

                # 9. article + cost budgets
                if len(state.articles_distilled) >= cfg.qa_max_articles:
                    state.stop_reason = "max_articles"
                    break
                if state.cost_cny >= cfg.qa_max_cost_cny:
                    state.stop_reason = "max_cost"
                    break
        except KeyboardInterrupt:
            state.stop_reason = "user_quit"
            write_state(cfg.vault_path, state)
            print(f"\nSession paused. Resume with: --resume {state.session_id}")

    # Final synthesis (skip if no articles)
    if state.articles_distilled:
        try:
            answer = synthesize(state.question, state.articles_distilled, llm)
        except AnswerError as e:
            if cfg.verbose:
                print(f"answer synthesis failed: {e}; writing skeleton survey only")
            answer = {
                "title": f"QA: {state.question[:60]}",
                "body": f"> 答案合成失败 ({e}). 已蒸馏 {len(state.articles_distilled)} 篇相关文章。",
                "tags": ["qa", "synthesis-failed"],
                "cited_slugs": [a.slug for a in state.articles_distilled],
            }
        body = _build_survey_body(answer, state)
        slug_base = slugify(state.question)[:30] or "untitled"
        slug = f"qa-{slug_base}-{datetime.now().strftime('%Y%m%d')}"
        try:
            saved = store.save_entry(
                category="surveys",
                title=answer["title"],
                body=body,
                tags=answer.get("tags") or ["qa"],
                refs=[f"qa-session:{state.session_id}"],
                slug=slug,
            )
        except ValueError:
            # slug conflict; append random hex
            slug = f"{slug}-{secrets.token_hex(2)}"
            saved = store.save_entry(
                category="surveys",
                title=answer["title"],
                body=body,
                tags=answer.get("tags") or ["qa"],
                refs=[f"qa-session:{state.session_id}"],
                slug=slug,
            )
        survey_slug = saved["slug"]
    else:
        survey_slug = None

    state.is_done = True
    _update_cost(state, llm)
    write_state(cfg.vault_path, state)

    summary = {
        "session_id": state.session_id,
        "stop_reason": state.stop_reason,
        "rounds_completed": state.rounds_completed,
        "articles_distilled_count": len(state.articles_distilled),
        "survey_slug": survey_slug,
        "cost_cny": round(state.cost_cny, 2),
        "tokens_in_total": state.tokens_in_total,
        "tokens_out_total": state.tokens_out_total,
    }
    return summary
```

- [ ] **Step 4: Run loop tests, confirm pass**

```bash
.venv\Scripts\python.exe -m pytest tests/test_qa_loop.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Run full suite**

```bash
.venv\Scripts\python.exe -m pytest -q --tb=no
```

Expected: 75 passed (70 + 5).

- [ ] **Step 6: Commit**

```bash
git add src/paper_distiller/qa/loop.py tests/test_qa_loop.py
git commit -m "feat(qa): main loop orchestrator (loop.py)

State-machine loop composing v0.3 L2 primitives. Termination:
- max_rounds (hard upper bound)
- llm_done (reflection.is_done + confidence >= threshold)
- llm_brake (reflection.suggest_stop)
- no_candidates (all search hits dedup against seen)
- max_articles (total articles distilled budget)
- max_cost (CNY circuit breaker)
- user_quit (Ctrl+C or --interactive 'n'/'q')

Each round persists SessionState to disk. KeyboardInterrupt writes
state and prints --resume hint. Final answer synthesis composes a
surveys/qa-...md doc with cited-articles table + audit trail.

5 integration tests cover the four primary stop reasons and the
state-persistence invariant. Subsystems are mocked at the import-name
level (paper_distiller.qa.loop.reflect, gather_candidates, etc.)."
```

---

## Task 6: `qa/cli.py` + pyproject.toml entry

**Files:**
- Create: `src/paper_distiller/qa/cli.py`
- Modify: `pyproject.toml`
- Create: `tests/test_qa_cli.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_qa_cli.py`:

```python
"""Tests for paper_distiller.qa.cli — argparse + dispatch."""
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from paper_distiller.qa.cli import build_parser, main


def test_parser_required_vault_and_question(tmp_path):
    parser = build_parser()
    args = parser.parse_args([
        "--vault", str(tmp_path), "--question", "why diffusion?",
    ])
    assert args.vault == str(tmp_path)
    assert args.question == "why diffusion?"
    assert args.max_rounds == 5  # default
    assert args.max_articles == 15  # default
    assert args.per_round == 2  # default
    assert args.confidence_threshold == 8
    assert args.interactive is False
    assert args.resume is None


def test_main_dispatches_to_loop(mocker, tmp_path, monkeypatch):
    monkeypatch.setenv("PD_API_KEY", "sk-test")
    monkeypatch.setenv("PD_BASE_URL", "https://x/v1")
    monkeypatch.setenv("PD_MODEL", "qwen-plus")
    mock_run = mocker.patch("paper_distiller.qa.cli.loop_run")
    mock_run.return_value = {
        "session_id": "sid-1", "stop_reason": "llm_done",
        "rounds_completed": 2, "articles_distilled_count": 4,
        "survey_slug": "qa-x-20260518", "cost_cny": 0.5,
        "tokens_in_total": 1000, "tokens_out_total": 500,
    }

    rc = main([
        "--vault", str(tmp_path), "--question", "why?", "--max-rounds", "3",
    ])
    assert rc == 0
    mock_run.assert_called_once()
    cfg = mock_run.call_args[0][0]
    assert cfg.qa_question == "why?"
    assert cfg.qa_max_rounds == 3
```

- [ ] **Step 2: Run, confirm fail**

```bash
.venv\Scripts\python.exe -m pytest tests/test_qa_cli.py -v
```

Expected: `ModuleNotFoundError: No module named 'paper_distiller.qa.cli'`.

- [ ] **Step 3: Create `qa/cli.py`**

Create `src/paper_distiller/qa/cli.py`:

```python
"""paper-distiller-qa command-line entry point."""

from __future__ import annotations

import argparse
import sys

from ..config import load_config_qa
from .loop import run as loop_run


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="paper-distiller-qa",
        description="Multi-round question-driven research loop over arxiv + "
                    "Semantic Scholar, writing a synthesized answer survey doc.",
    )
    p.add_argument("--vault", required=True, help="Path to your Obsidian vault.")
    p.add_argument("--question", required=True, help="Research question to answer.")
    p.add_argument("--max-rounds", type=int, default=5,
                    help="Hard upper bound on loop rounds (default 5).")
    p.add_argument("--max-articles", type=int, default=15,
                    help="Hard upper bound on total articles distilled (default 15).")
    p.add_argument("--max-cost-cny", type=float, default=20.0,
                    help="Cost circuit breaker in CNY (default 20.0).")
    p.add_argument("--confidence-threshold", type=int, default=8,
                    help="LLM is_done confidence required to stop (0-10, default 8).")
    p.add_argument("--per-round", type=int, default=2,
                    help="Articles to distill each round (default 2).")
    p.add_argument("--source", choices=["arxiv", "ss", "both"], default="both",
                    help="Paper source(s) to search (default both).")
    p.add_argument("--interactive", action="store_true",
                    help="Pause after each round and prompt to continue (Y/n/q).")
    p.add_argument("--resume", help="Resume a paused session by its session_id.")
    p.add_argument("--verbose", "-v", action="store_true", help="Detailed logging.")
    p.add_argument("--dry-run", action="store_true",
                    help="Plan only; no LLM, no vault writes.")
    p.add_argument("--model", help="Override PD_MODEL env var.")
    p.add_argument("--provider", help="Override PD_PROVIDER_NAME label.")
    return p


def main(argv: list | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg = load_config_qa(
            vault_path=args.vault,
            question=args.question,
            max_rounds=args.max_rounds,
            max_articles=args.max_articles,
            max_cost_cny=args.max_cost_cny,
            confidence_threshold=args.confidence_threshold,
            per_round=args.per_round,
            source=args.source,
            interactive=args.interactive,
            resume_session_id=args.resume,
            verbose=args.verbose,
            dry_run=args.dry_run,
            model_override=args.model,
            provider_override=args.provider,
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

    try:
        summary = loop_run(cfg)
    except Exception as e:
        print(f"\nError during QA loop: {type(e).__name__}: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc(file=sys.stderr)
        else:
            print("(run with --verbose for full traceback)", file=sys.stderr)
        return 3

    print()
    print(f"  Session:        {summary['session_id']}")
    print(f"  Stop reason:    {summary['stop_reason']}")
    print(f"  Rounds:         {summary['rounds_completed']}")
    print(f"  Articles:       {summary['articles_distilled_count']}")
    print(f"  Survey slug:    {summary.get('survey_slug') or '(none — no articles)'}")
    print(f"  Cost:           ¥{summary['cost_cny']:.2f}")
    print(f"  Tokens in/out:  {summary['tokens_in_total']} / {summary['tokens_out_total']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Update `pyproject.toml` to add the new entry point**

Find this block in `pyproject.toml`:

```toml
[project.scripts]
paper-distiller = "paper_distiller.cli:main"
```

Replace with:

```toml
[project.scripts]
paper-distiller = "paper_distiller.cli:main"
paper-distiller-qa = "paper_distiller.qa.cli:main"
```

- [ ] **Step 5: Reinstall in editable mode so the new console script is registered**

```bash
.venv\Scripts\python.exe -m pip install -e . --no-deps --quiet
```

Expected: no errors. The `paper-distiller-qa` shell command should now exist on PATH.

- [ ] **Step 6: Smoke-check the help output**

```bash
.venv\Scripts\paper-distiller-qa.exe --help
```

Expected: argparse help with `--vault`, `--question`, and the budget/mode flags.

- [ ] **Step 7: Run cli tests, confirm pass**

```bash
.venv\Scripts\python.exe -m pytest tests/test_qa_cli.py -v
```

Expected: 2 passed.

- [ ] **Step 8: Run full suite**

```bash
.venv\Scripts\python.exe -m pytest -q --tb=no
```

Expected: 77 passed (75 + 2).

- [ ] **Step 9: Commit**

```bash
git add src/paper_distiller/qa/cli.py pyproject.toml tests/test_qa_cli.py
git commit -m "feat(qa): paper-distiller-qa CLI entry point

argparse-based CLI with --vault, --question, and 9 optional flags.
Dispatches to qa.loop.run after building Config via load_config_qa.
Friendly error formatting (no stack trace by default; --verbose shows).

pyproject.toml registers paper-distiller-qa as a console script
alongside the existing paper-distiller (v0.3 L2).

2 unit tests cover argparse parsing + dispatch."
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
__version__ = "0.3.0"
```

to:

```python
__version__ = "0.5.0"
```

- [ ] **Step 2: Bump `pyproject.toml` version**

```toml
version = "0.5.0"
```

- [ ] **Step 3: Update `tests/test_smoke.py`**

Change:

```python
    assert paper_distiller.__version__ == "0.3.0"
```

to:

```python
    assert paper_distiller.__version__ == "0.5.0"
```

- [ ] **Step 4: Update `CHANGELOG.md`**

Prepend a `[0.5.0]` section ABOVE the existing `[0.3.0]` entry:

```markdown
## [0.5.0] — 2026-05-18

### Added
- **Question-driven multi-round research loop (`paper-distiller-qa`).** Given a research question, the agent autonomously plans search queries, distills relevant papers across multiple rounds, and synthesizes a cited answer document written to `<vault>/surveys/qa-<slug>-<YYYYMMDD>.md`. Bounded by hard budget (rounds/articles/cost) + LLM "is_done" judgment + diminishing-returns detection.
- **Seven stop reasons** surfaced in the final survey footer + terminal summary: `max_rounds`, `llm_done`, `llm_brake`, `no_candidates`, `max_articles`, `max_cost`, `user_quit`.
- **`--interactive` mode** pauses after each round and prompts to continue (Y/n/q) — useful for prompt debugging and untrusted-question runs.
- **`--resume <session-id>` mode** picks up a paused or crashed session from disk-persisted state (`<vault>/.paper_distiller/qa-sessions/<sid>/state.json`).
- **Audit trail** rendered as a markdown table in every qa-survey doc: per-round query, LLM rationale, new articles, confidence.
- **`qa/state.py`** — `SessionState` + `RoundRecord` dataclasses with disk persistence.
- **Two new prompt templates** in `src/paper_distiller/qa/prompts/`: `reflect.md` (LLM judges loop progress) and `answer.md` (LLM synthesizes final cited answer).

### Changed
- **Pipeline helper promotion.** `pipeline._gather_candidates` and `pipeline._fetch_with_fallback` aliased to public names (`gather_candidates`, `fetch_with_fallback`) for qa-loop reuse. Old underscore names retained for v0.3 back-compat.
- **`Config` extended** with `qa_max_rounds`, `qa_max_articles`, `qa_max_cost_cny`, `qa_confidence_threshold`, `qa_per_round`, `qa_interactive`, `qa_resume_session_id`, `qa_question` (all defaults; v0.3 callers unaffected). New `load_config_qa()` validates qa-specific kwargs.

### Internal
- 16 new unit/integration tests (3 state + 3 reflection + 3 answer + 5 loop + 2 cli); total now **77** (was 61 in v0.3).
- No new runtime dependencies.

### Note on v0.4 gap
v0.4 was explored as a self-shipped LEANN-backed MCP server, then reverted in favor of recommending [vault-mcp](https://github.com/robbiemu/vault-mcp). See [docs/vault-mcp-recommendation.md](docs/vault-mcp-recommendation.md) and the README "Optional companion" section. No v0.4 tag exists; v0.5 is the next semantic-version bump.

```

- [ ] **Step 5: Verify version consistency**

```bash
findstr /R "^version" pyproject.toml
findstr "__version__" src\paper_distiller\__init__.py
findstr "0.5.0" tests\test_smoke.py
```

All should show `0.5.0`.

- [ ] **Step 6: Run full suite one more time**

```bash
.venv\Scripts\python.exe -m pytest -q --tb=no
```

Expected: 77 passed.

- [ ] **Step 7: Commit and tag**

```bash
git add src/paper_distiller/__init__.py pyproject.toml tests/test_smoke.py CHANGELOG.md
git commit -m "chore: bump version to 0.5.0 + changelog (note v0.4 gap)"
git tag -a v0.5.0 -m "v0.5.0 — question-driven multi-round research loop (paper-distiller-qa)"
```

- [ ] **Step 8: Verify tag and history**

```bash
git tag --list -n
git log --oneline | head -12
```

Expected:
- `v0.5.0` in tag list with the annotated message
- Recent log shows 7 v0.5 commits + the v0.4 rollback (b6cc3c3 docs commit) + v0.3 history below

---

## Task 8 (optional): Commit spec + plan files

If the v0.5 spec and plan files (`docs/superpowers/specs/2026-05-18-paper-distiller-v0.5.0-design.md` and `docs/superpowers/plans/2026-05-18-paper-distiller-v0.5.0.md`) are still untracked, commit them:

- [ ] **Step 1: Check git status**

```bash
git status --short docs/superpowers/
```

If they appear as `??`, proceed to Step 2.

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/specs/2026-05-18-paper-distiller-v0.5.0-design.md docs/superpowers/plans/2026-05-18-paper-distiller-v0.5.0.md
git commit -m "docs: add v0.5.0 spec + plan"
```

---

## Manual smoke test (after all tasks complete)

Not a commit step — manual verification per spec §11.

- [ ] Cheap smoke: `paper-distiller-qa --vault "G:/Math research Agent/wiki" --question "扩散模型在金融时序生成的核心难点" --max-rounds 2 --per-round 1 -v`
- [ ] Verify a new survey exists under `wiki/surveys/qa-...-YYYYMMDD.md`
- [ ] Open in Obsidian — frontmatter renders in Properties panel; body has the answer + cited-articles table + audit trail
- [ ] Verify session state at `wiki/.paper_distiller/qa-sessions/<sid>/state.json`
- [ ] Optionally: Ctrl+C the run mid-loop and re-run with `--resume <sid>` — confirms resume works
- [ ] Optionally: run with `--interactive` to see per-round checkpoint prompts

---

## Acceptance criteria (rolled up from spec §11)

After all 7 tasks committed (+ optional 8):

- [ ] `pytest -q --tb=no` from `G:/paper-distiller/`: **77 tests pass**
- [ ] `paper-distiller-qa --help` shows full CLI (--vault, --question, 9 optional flags)
- [ ] End-to-end manual smoke per the section above passes
- [ ] All seven stop reasons appear in the survey footer / terminal summary correctly (verified in tests + manual)
- [ ] State persisted to `<vault>/.paper_distiller/qa-sessions/<sid>/state.json` after each round
- [ ] `--interactive` mode pauses + prompts correctly (manual)
- [ ] `--resume <sid>` continues a paused session (manual)
- [ ] `__version__` and `pyproject.toml` show `0.5.0`
- [ ] `CHANGELOG.md` has a `[0.5.0]` section with v0.4-gap note
- [ ] Annotated tag `v0.5.0` exists
- [ ] No regressions in v0.3 functionality (existing 61 tests still pass)

---

## Self-review notes

**Spec coverage:**
- Spec §1 Goal → all 7 implementation tasks combined deliver the QA loop.
- Spec §3 Out of scope → no tasks attempt deferred features.
- Spec §4 Architecture → matches Task 5 loop.py module-level structure.
- Spec §5 CLI → Task 6 implements all 11 flags (--vault, --question, --max-rounds, --max-articles, --max-cost-cny, --confidence-threshold, --per-round, --source, --interactive, --resume, --verbose/--dry-run).
- Spec §6 Module structure → matches Tasks 2-6 file layout.
- Spec §7 Loop flow in detail → Task 5 loop.py implements all 11 ordered steps + final synthesis.
- Spec §8 Persistence layout → Task 2 state.py write_state / read_state + Task 5 calls write_state each round.
- Spec §9 Prompt templates → Task 3 reflect.md + Task 4 answer.md exist with the specified JSON output keys.
- Spec §10 Cost estimate → no code work needed; the loop tracks tokens via the existing LLMClient and converts via constants at the top of loop.py.
- Spec §11 Acceptance criteria → "Acceptance criteria" section above replicates them.
- Spec §12 Implementation roadmap → this plan IS that decomposition.
- Spec §13 Compatibility → enforced by the v0.3-tests-still-pass invariant in Steps 5/6/9 of each task.

**No placeholders detected** in any task. All code blocks contain runnable code with concrete paths and assertions. All `git commit` messages have substantive bodies, not "TBD".

**Type/name consistency:**
- `SessionState`, `RoundRecord` (Task 2) referenced consistently in Tasks 5 (loop) and 6 (cli return type).
- `reflect()` signature (Task 3) matches the call in Task 5's loop.
- `synthesize()` signature (Task 4) matches the call in Task 5's loop.
- `ReflectionError` / `AnswerError` (Tasks 3/4) caught explicitly in Task 5's loop with matching stop_reason strings.
- `gather_candidates`, `fetch_with_fallback` (Task 1 promotion) imported in Task 5's loop matching the renamed public symbols.
- `load_config_qa()` (Task 1) called by `paper_distiller.qa.cli.main` (Task 6).
- `Config.qa_*` fields (Task 1) read in Task 5's loop with matching names (`cfg.qa_max_rounds`, etc.) and `cfg.qa_question` (which Task 6's CLI populates).

**Estimated total effort:** 9 hours across 7 tasks (≈1–1.5 working days). Task 5 (loop) is the largest single block at ~3 hours.
