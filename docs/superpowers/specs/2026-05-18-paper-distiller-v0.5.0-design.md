# paper-distiller v0.5.0 — Design

**Date**: 2026-05-18
**Author**: brainstorm session (post-v0.3.0 ship; v0.4 deferred to vault-mcp recommendation)
**Status**: design (pending implementation plan)
**Target**: `G:\paper-distiller\` (existing repo, baseline tag `v0.3.0` @ `682611f`)

---

## 1. Goal

Add an L3 **question-driven multi-round research loop** to paper-distiller. Users give a research question; the agent autonomously plans search queries, distills relevant papers across multiple rounds, and synthesizes a cited answer document — all bounded by hard budget + termination heuristics that defend against the "agent won't stop" failure mode the user's prior `autonomous_research` mode (in the deleted Math Research Agent project) suffered from.

This is the "L3" mode envisioned in the original v0.1 design doc. v0.3 implemented L2 (single-pass search-and-distill). v0.5 builds L3 on top by composing L2 primitives into a multi-round loop with a reflection step.

The C path (question-driven QA) is chosen over the B path (citation traversal) and A path (breadth survey) because: (a) it has the most explicit termination signal — "the question is answered" — making it least prone to runaway; (b) its scope stays anchored to one question, avoiding drift; (c) it directly fits the user's day-to-day research workflow ("I'm stuck on X, what's relevant?"). B and A are tentatively scheduled for v0.6 and v0.7.

---

## 2. Context

### What's already in place (v0.3.0)

- L2 pipeline: arxiv + Semantic Scholar candidate gather, LLM ranker, PDF download with SS fallback, LLM distillation, optional session survey, run-log persisted to `.paper_distiller/runs.jsonl`.
- `Config` supports per-source budgets (pool, top_n, force) and embedding key.
- `vault/store.VaultStore` provides Obsidian-compatible markdown CRUD + arxiv-id / DOI dedup via `find_by_arxiv_id` / `find_by_doi`.
- The pipeline's internal helpers (`_gather_candidates`, `merge_candidates`, `_fetch_with_fallback`) implement the L2 round building blocks that L3 will reuse.

### Prior-art lessons (user's deleted autonomous-research mode)

The user previously shipped a multi-round research agent in a different codebase, deleted because it had two recurring failure modes:
1. **Going off-topic** — agent decided unrelated subqueries were interesting and chased them, ending far from the original goal.
2. **Won't stop** — no hard budget, no clean termination signal, runs ate API credit until manually killed.

v0.5 must defend against both **as primary design constraints**, not afterthoughts.

### Why now

- v0.3 has stable L2 primitives — composing into L3 is mostly orchestration code, not new infrastructure.
- v0.4 (own MCP server) was correctly deferred to vault-mcp; doesn't block v0.5.
- vault-mcp can later be added inside the loop as a "what do we already know" signal, but is optional for the MVP.

### v0.4 number gap

v0.4 was explored (LEANN-backed MCP server) and reverted in favor of recommending vault-mcp. README and CHANGELOG document this. v0.5 inherits the next-tag slot directly; CHANGELOG entry will note the gap.

---

## 3. Out of scope for v0.5.0

| Not doing in v0.5 | Reason / where |
|---|---|
| Multi-question session (one session answers several questions in parallel) | YAGNI; single-question MVP first |
| Resume mid-round (after crash within one round) | Only support resume at round boundaries; mid-round adds significant state complexity |
| Branching exploration (try multiple queries in parallel each round, compare) | Doubles per-round budget; complicates state; single linear flow first |
| Web / TUI live progress UI | CLI verbose output is sufficient |
| LLM model switching mid-session | Single model per session simplifies reasoning |
| Cost dry-run / pre-flight estimation | Hard to predict; just enforce the budget at runtime |
| Citation graph traversal from distilled articles | v0.6 (B path) |
| Breadth literature survey | v0.7 (A path) |
| In-loop calls to vault-mcp for prior-knowledge awareness | v0.5.1 candidate; let the MVP ship first |
| `find_by_ss_paper_id` (vault dedup for SS-only papers without arxiv/DOI) | Edge case; current `find_by_arxiv_id` + `find_by_doi` covers vast majority |
| Auto-publishing the survey doc to GitHub Pages / blog | Out of scope for this project entirely |

---

## 4. Architecture summary

```
┌── paper-distiller-qa CLI (new entry point) ────────────────┐
│                                                            │
│   parse args → load_config_qa() → qa.loop.run(cfg)         │
│                                                            │
└────────────────────────────┬───────────────────────────────┘
                             │
                             ▼
┌── qa.loop ───────────────────────────────────────────────────┐
│                                                              │
│   state = SessionState(...)                                  │
│   while not terminated:                                      │
│       reflection = qa.reflection.reflect(question, state)    │
│       check_stop_conditions(state, reflection, cfg)          │
│       if state.is_done: break                                │
│       if cfg.interactive: prompt user "continue?"            │
│       candidates = pipeline.gather_candidates(...)           │
│       new = dedupe_against_seen(candidates, state)           │
│       top = pipeline.rank(new, ...)                          │
│       for paper in top: distill + save_entry                 │
│       update state + persist round-N.json                    │
│                                                              │
│   final_answer = qa.answer.synthesize(question, state)       │
│   save_entry(category="surveys", slug="qa-...", ...)         │
│                                                              │
└──────────────────────────────────────────────────────────────┘
       │                              │                  │
       ▼                              ▼                  ▼
  qa.reflection                  qa.answer        pipeline helpers
  (LLM call, JSON output)        (LLM call,        (gather_candidates,
                                  JSON output)     rank, distill_article,
                                                   fetch_with_fallback,
                                                   etc.)
```

The loop composes existing L2 primitives. The only genuinely new code is:
- `qa/loop.py` — the state machine + termination logic
- `qa/state.py` — SessionState dataclass + on-disk persistence
- `qa/reflection.py` — LLM reflection call wrapper
- `qa/answer.py` — LLM answer synthesis call wrapper
- `qa/prompts/{reflect,answer}.md` — two new prompts
- `qa/cli.py` — argparse + dispatch

Existing pipeline helpers (currently `_gather_candidates`, `_fetch_with_fallback`) are promoted to public names (drop leading underscore) for reuse without coupling.

---

## 5. CLI surface

New entry point declared in `pyproject.toml`:

```toml
[project.scripts]
paper-distiller = "paper_distiller.cli:main"           # existing v0.3 L2
paper-distiller-qa = "paper_distiller.qa.cli:main"     # NEW
```

Full `paper-distiller-qa` reference:

```
paper-distiller-qa --vault <path> --question "<text>"
  [--max-rounds 5]                hard upper bound on rounds (default 5)
  [--max-articles 15]             hard upper bound on total articles distilled
  [--max-cost-cny 20]             cost circuit breaker (CNY)
  [--confidence-threshold 8]      LLM is_done confidence required to stop (0-10)
  [--per-round 2]                 articles to distill each round
  [--source arxiv|ss|both]        inherits v0.3 source choice (default both)
  [--interactive]                 pause after each round, prompt Y/n/q
  [--resume <session-id>]         resume a previously persisted session
  [--verbose / -v]
  [--dry-run]                     plan only; no LLM, no vault writes
```

Why a separate command (not `paper-distiller qa ...` subcommand): keeps backward compatibility with v0.3 — existing scripts that call `paper-distiller --vault X --topic Y` keep working. Subcommands would force a breaking rewrite. Two top-level entry points is fine for an early-stage tool.

---

## 6. Module structure

```
src/paper_distiller/
├── qa/                                ← NEW package
│   ├── __init__.py
│   ├── cli.py                          paper-distiller-qa entry point
│   ├── loop.py                         main orchestrator: state machine + termination
│   ├── state.py                        SessionState dataclass + persistence
│   ├── reflection.py                   LLM reflection call (JSON-out)
│   ├── answer.py                       LLM answer synthesis call (JSON-out)
│   └── prompts/
│       ├── reflect.md
│       └── answer.md
├── pipeline.py                        ← MINOR change: rename internal helpers public
└── config.py                          ← MINOR change: add qa_* fields + load_config_qa()
```

Compatibility note: pipeline's `_gather_candidates` and `_fetch_with_fallback` are referenced by qa code. Two options: (a) qa imports the underscore-prefixed names (Python doesn't enforce privacy); (b) rename in pipeline.py and update the qa references. Option (b) is cleaner — the helpers were always reused-by-design, the underscore prefix was a label slip. Go with (b).

---

## 7. Loop flow in detail

### State machine

```
SessionState (in-memory + persisted to disk after each round)
  session_id: str        # e.g. "20260518-2143-abc12"
  question: str
  config_snapshot: dict
  started_at: str
  rounds_completed: int
  articles_distilled: list[ArticleResult]
  articles_seen_ids: set[str]   # arxiv_ids ∪ dois
  history: list[RoundRecord]
  last_reflection: dict | None
  cost_cny: float
  tokens_in_total: int
  tokens_out_total: int
  is_done: bool
  stop_reason: str

RoundRecord
  round: int
  query: str
  rationale: str
  candidates_found: int
  new_articles: int
  article_slugs: list[str]
  what_we_know: str
  what_is_missing: str
  confidence: int
  timestamp: str
```

### One round (logical sequence)

1. **Reflection**: call `qa.reflection.reflect(state, llm)` — LLM returns JSON `{is_done, confidence, what_we_know, what_is_missing, next_query, next_query_rationale, suggest_stop}`.
2. **Check stop conditions** (in order, first hit wins):
   - `rounds_completed >= max_rounds` → stop_reason = `max_rounds`
   - `reflection.is_done and reflection.confidence >= confidence_threshold` → `llm_done`
   - `reflection.suggest_stop` → `llm_brake`
   - (else continue to step 3)
3. **Interactive checkpoint** (if `--interactive`): print reflection JSON, prompt `Continue to round N+1? [Y/n/q]`. `q` → `user_quit`, `n` → `user_quit`, `Y/Enter` → continue.
4. **Search**: `candidates = pipeline.gather_candidates(query=reflection.next_query, ...)`. Dedupe against `state.articles_seen_ids` → `new_candidates`.
5. **Diminishing-returns check**: `len(new_candidates) == 0` → `no_candidates`.
6. **Rank**: `top = pipeline.rank(new_candidates, question, top_n=cfg.per_round, llm)`.
7. **Distill loop** for `paper in top`: `_fetch_with_fallback` → `distill_article` → `store.save_entry(category="articles", ...)`. Append to `state.articles_distilled`. Add `arxiv_id`/`doi` to `state.articles_seen_ids`.
8. **Article budget**: `len(state.articles_distilled) >= max_articles` → `max_articles`.
9. **Cost budget**: `state.cost_cny >= max_cost_cny` → `max_cost`.
10. **Persist state**: write `state.json` (overwrite) + `rounds/round-N.json` (new).
11. **Record round in history**.

### Final synthesis (outside loop)

1. `qa.answer.synthesize(question, articles_distilled, llm)` → `{title, body, tags, cited_slugs}`.
2. Strip `[[wikilinks]]` from `body` whose slug isn't in `state.articles_distilled` slug set (hallucination scrub mirror of `distill.article._scrub_invented_links`).
3. Append to body: "## 引用的 articles" table from `cited_slugs`, "## 研究过程（audit trail）" table from `state.history`, "**Stop reason** / **Cost** / **Session ID** / **Duration**" footer.
4. `slug = f"qa-{slugify(question)[:30]}-{YYYYMMDD}"` — if conflicts and not `--force`, append random 5-char hex.
5. `store.save_entry(category="surveys", title=..., body=..., tags=..., refs=[f"qa-session:{session_id}"], slug=...)`.
6. Print final terminal summary.

---

## 8. Persistence layout

```
<vault>/.paper_distiller/qa-sessions/
└── <session-id>/                   # e.g. "20260518-2143-abc12"
    ├── meta.json                   # question / config_snapshot / started_at / stop_reason
    ├── state.json                  # latest SessionState snapshot (overwritten each round)
    └── rounds/
        ├── round-1.json            # RoundRecord + the round's distilled article slugs
        ├── round-2.json
        └── ...
```

`session_id` format: `YYYYMMDD-HHMM-<5-char random hex>`. Generated at session start.

`--resume <session-id>` reads `state.json`, validates `rounds_completed > 0`, and starts loop at round `rounds_completed + 1` with full state restored (`articles_distilled`, `articles_seen_ids`, `history`, `cost_cny`, etc.).

If a `--resume` session is already `is_done=True`, the resume is rejected with a friendly error pointing to the saved survey.

### Sessions are gitignored

`<vault>/.gitignore` already excludes `.paper_distiller/` (introduced in v0.1 — verify and patch if necessary). No further changes needed.

---

## 9. Prompt templates

### `qa/prompts/reflect.md`

(Full text in §3 of the brainstorm transcript; below is the key contract for spec purposes.)

**Input placeholders**: `{round_num}`, `{max_rounds}`, `{question}`, `{n_articles}`, `{articles_summary}` (one-line per article), `{prior_queries}` (newline-joined list).

**Required output JSON shape**:
```json
{
  "is_done": false,
  "confidence": 4,
  "what_we_know": "...",
  "what_is_missing": "...",
  "next_query": "...",
  "next_query_rationale": "...",
  "suggest_stop": false
}
```

**Hard constraints embedded in prompt**:
- `next_query` must be related to original question
- `next_query` must differ from all `prior_queries`
- `confidence` must be conservative; `>=8` only when truly answerable

### `qa/prompts/answer.md`

**Input placeholders**: `{question}`, `{n_articles}`, `{articles_full}` (each article: slug + title + body capped at ~12K chars; total cap to ~80K to stay under model context).

**Required output JSON shape**:
```json
{
  "title": "QA: ...",
  "body": "...markdown...",
  "tags": ["..."],
  "cited_slugs": ["slug1", "slug2", ...]
}
```

**Constraints**:
- All `[[wikilinks]]` in body must come from the provided slug set
- Chinese primary
- Body length target 800–2500 chars

---

## 10. Cost estimate

Using qwen-plus at ~$0.30/M in + $1.80/M out (≈¥2.1/M in + ¥12.7/M out):

| Per-round call | tokens in / out | cost CNY |
|---|---|---|
| Reflection | 5K / 1K | ¥0.024 |
| Rank | 8K / 0.5K | ¥0.023 |
| Distill × 2 (full-pdf) | 30K × 2 / 3K × 2 | ¥0.20 |
| **One round** | | **~¥0.25** |

Five rounds: ~¥1.25 + final answer synthesis (~50K in / 3K out → ~¥0.14) = **~¥1.4 / session**.

The `--max-cost-cny 20` default is **~14×** the typical run cost — there's substantial headroom. Power users running deep sessions (more rounds, longer reflection) won't hit the limit casually; abusive runs will.

---

## 11. Acceptance criteria

After all v0.5 implementation tasks complete and `v0.5.0` is tagged:

- [ ] `paper-distiller-qa --help` shows the full CLI (`--vault`, `--question`, all 7 budget/mode flags)
- [ ] `paper-distiller-qa --vault <path> --question "..."` runs end-to-end and writes `<vault>/surveys/qa-...-YYYYMMDD.md`
- [ ] The survey doc renders cleanly in Obsidian (frontmatter parses; body shows audit table)
- [ ] `--max-rounds 1` forces a single round + immediate synthesis
- [ ] `--interactive` pauses each round; `Y` continues, `n`/`q` stops cleanly
- [ ] `--resume <session-id>` continues a paused session at the next round
- [ ] All seven stop reasons appear correctly in the survey footer and terminal summary: `max_rounds` / `llm_done` / `llm_brake` / `no_candidates` / `max_articles` / `max_cost` / `user_quit`
- [ ] State persisted to `<vault>/.paper_distiller/qa-sessions/<sid>/` (meta.json + state.json + rounds/*)
- [ ] 16 new tests pass (`tests/test_qa_*.py`): 3 state + 3 reflection + 3 answer + 5 loop + 2 cli. Full suite goes from 61 → **77 passed**.
- [ ] `CHANGELOG.md` `[0.5.0]` section explains v0.4 gap
- [ ] Annotated tag `v0.5.0` exists

---

## 12. Implementation roadmap (for writing-plans)

Approximate task decomposition (~7 tasks):

1. **Config + pipeline helper promotion**: add `Config.qa_*` fields + `load_config_qa()` (or extend `load_config` with qa-specific kwargs). Rename `pipeline._gather_candidates` → `gather_candidates`, `_fetch_with_fallback` → `fetch_with_fallback`. Update v0.3 callers + tests. ~30 min.
2. **`qa/state.py` + 3 tests** (SessionState + persistence). ~1.5 h.
3. **`qa/prompts/reflect.md` + `qa/reflection.py` + 3 tests**. ~1.5 h.
4. **`qa/prompts/answer.md` + `qa/answer.py` + 3 tests**. ~1.5 h.
5. **`qa/loop.py` + 5 integration tests** (main orchestrator: state machine, termination logic, persistence between rounds, KeyboardInterrupt handling). ~3 h.
6. **`qa/cli.py` + `pyproject.toml` entry + 2 tests**. ~1 h.
7. **CHANGELOG + version bump 0.3.0 → 0.5.0 + tag**. ~30 min.

Total: ~9 hours (≈1–1.5 working days). About 50% larger than v0.3's effort.

---

## 13. Compatibility note

v0.5.0 is **additive** — no breaking changes to v0.3 surface:
- `paper-distiller --vault X --topic Y ...` keeps working unchanged
- v0.3 `Config` fields unchanged (qa fields are additions)
- vault layout unchanged (qa session writes go into `.paper_distiller/qa-sessions/` and `surveys/qa-*.md`)
- `distill/article.py` and `distill/survey.py` reused as-is; no fork
