# Architecture

How paper-distiller is laid out and how data flows through it. Read this if you want to extend it, customize the prompts, or understand a failure.

## Two CLIs

paper-distiller ships two console entry points that share the same underlying primitives:

| CLI | Mode | When to use |
|---|---|---|
| `paper-distiller` | **L2 single-pass** — one search → distill N papers → optional survey | You already know the topic and just want N papers added to the vault |
| `paper-distiller-qa` | **L3 multi-round loop** — reflect → search → distill → repeat until done | You have a research *question* and want the tool to autonomously plan multiple search rounds and synthesize a cited answer |

Both write into the same Obsidian-compatible vault layout (`articles/`, `surveys/`).

## Package layout

```
src/paper_distiller/
├── __init__.py                 __version__ = "0.5.0"
├── config.py                   Config dataclass + load_config()/load_config_qa()
├── cli.py                      `paper-distiller` argparse entry
├── pipeline.py                 The L2 primitives: gather_candidates, rank, fetch_with_fallback
├── llm/
│   └── openai_compatible.py    LLMClient — OpenAI-compatible HTTP, JSON response_format, token accounting
├── sources/
│   ├── arxiv.py                arxiv API search + PDF download (uses the `arxiv` PyPI package)
│   └── semantic_scholar.py     Semantic Scholar API search + openAccessPdf lookup
├── extract/
│   └── pymupdf_extractor.py    PDF → plain text via PyMuPDF
├── distill/
│   ├── filter.py               LLM ranker — given M candidates + topic, return top N
│   ├── article.py              LLM paper distillation — produces an ArticleResult
│   └── survey.py               LLM cluster survey — composes multi-article survey doc
├── prompts/                    L2 prompts (filter.md, article.md, survey.md)
├── vault/
│   ├── schema.py               Category list + Paper / ArticleResult dataclasses
│   ├── store.py                VaultStore — saves entries, dedups by arxiv-id / DOI / slug
│   └── crosslink.py            Loads existing vault index for [[wikilink]] injection during distill
└── qa/                         v0.5 additions — multi-round loop
    ├── cli.py                  `paper-distiller-qa` argparse entry
    ├── loop.py                 The orchestrator state machine
    ├── reflection.py           One LLM reflection call (judge progress + propose next query)
    ├── answer.py               One LLM answer-synthesis call (compose final cited answer)
    ├── state.py                SessionState + RoundRecord + disk persistence
    └── prompts/                QA prompts (reflect.md, answer.md)
```

The `Paper` dataclass in `vault/schema.py` is the cross-source unification point — every source returns the same shape (`arxiv_id`, `doi`, `ss_paper_id`, `title`, `abstract`, etc.), so downstream code never needs to special-case arxiv vs Semantic Scholar.

## L2 flow — `paper-distiller`

```
                            (1)            (2)            (3)              (4)              (5)
  user gives --topic  →  arxiv + SS  →  LLM filter   →  PDF fetch     →  LLM distill   →  vault.save_entry
                          (~30 hits)     (→ top N)       (with fallback)   (per paper)        (articles/<slug>.md)
                                                                                              │
                                                                                              ↓
                                                                                          (6) LLM survey
                                                                                              if N >= PD_MIN_SURVEY
                                                                                              (surveys/<slug>.md)
```

Step-by-step:

1. **`gather_candidates(cfg)`** — calls `sources/arxiv.py` and/or `sources/semantic_scholar.py` per `cfg.source` (`arxiv` / `ss` / `both`). For `both`, merges results and dedups by `arxiv_id`, then `doi` (arxiv source wins on tie).
2. **`rank(candidates, topic, top_n, llm)`** — one LLM call with the `filter.md` prompt; returns indices of the top-N most relevant. Cheap (a few hundred tokens).
3. **`fetch_with_fallback(paper, cfg, tmpdir)`** — tries the paper's primary PDF URL. If it 4xx/timeouts AND the paper has an `arxiv_id` or `doi`, queries Semantic Scholar for `openAccessPdf` and retries. If still no PDF and abstract length ≥ 500 chars, falls back to abstract-only mode. If even the abstract is missing, raises.
4. **`pymupdf_extractor.extract(pdf_path)`** — plain text. PyMuPDF is fast (50ms for typical paper).
5. **`distill_article(paper, full_text, wiki_index, llm)`** — one LLM call with the `article.md` prompt. The `wiki_index` (from `crosslink.load_index`) is a flat list of existing slugs + titles, injected into the prompt so the LLM can cross-reference them via `[[wikilink]]`. Output is an `ArticleResult` dataclass.
6. **`store.save_entry(category="articles", ...)`** — writes `<vault>/articles/<slug>.md` with YAML frontmatter + body. Slug collisions resolved by appending random hex.
7. **Survey** (optional) — if `len(articles) >= cfg.min_papers_for_survey`, one final LLM call with `survey.md` produces a session survey under `<vault>/surveys/`.

**Dedup logic** (L2 + L3 share this):

- Before saving an article, `VaultStore` runs `find_by_arxiv_id`, then `find_by_doi`. If either hits, the existing file is reused (no new write).
- The L3 loop additionally tracks `articles_seen_ids` in memory to avoid re-distilling within the same session.

## L3 flow — `paper-distiller-qa`

A state machine. Each round runs steps 1-8; the loop terminates via one of 7 stop reasons.

```
                           ┌─────────────────────────────────────────┐
                           │  qa.loop.run(cfg)                       │
                           │                                         │
   user gives --question → │  while True:                            │
                           │    1. reflect (LLM)         ─────┐      │
                           │    2. termination checks  →      │      │
                           │    3. interactive checkpoint     │      │  on stop:
                           │    4. search (gather_candidates) │ ←────┤  → synthesize answer (LLM)
                           │    5. dedup vs seen              │      │  → save to surveys/qa-…md
                           │    6. rank (LLM)                 │      │  → write state.json (final)
                           │    7. distill loop (N × LLM)     │      │
                           │    8. write state.json           │      │
                           │    9. check budgets    ──────────┘      │
                           └─────────────────────────────────────────┘
```

### The 7 stop reasons

| Stop reason | Trigger | `is_done`? |
|---|---|---|
| `max_rounds` | `state.rounds_completed >= cfg.qa_max_rounds` | yes |
| `llm_done` | reflection returns `is_done=True` AND `confidence >= cfg.qa_confidence_threshold` | yes |
| `llm_brake` | reflection returns `suggest_stop=True` (diminishing-returns judgement) | yes |
| `no_candidates` | search returned 0 candidates OR all candidates dedup against already-seen | yes |
| `max_articles` | total articles distilled across all rounds reaches `cfg.qa_max_articles` | yes |
| `max_cost` | accumulated `state.cost_cny >= cfg.qa_max_cost_cny` | yes |
| `user_quit` | Ctrl+C OR `--interactive` user answers `n`/`q` | **no** (resumable) |
| `error: <details>` | any uncaught exception in search/rank/distill | **no** (resumable) |

`is_done=False` stops are resumable via `--resume <session-id>`. Done stops cannot be resumed (raises `ValueError`).

### Reflection prompt (`qa/prompts/reflect.md`)

Inputs to the LLM each round:
- The original `question`
- Summary of `articles_summary` distilled so far (one line each: `[[slug]] title`)
- `prior_queries` already searched (to avoid duplicate searches)
- `round_num` and `max_rounds` (so the LLM can pace itself)

LLM returns JSON with:
```json
{
  "is_done": false,
  "confidence": 4,
  "what_we_know": "...",
  "what_is_missing": "...",
  "next_query": "diffusion models long-horizon time series 2024",
  "next_query_rationale": "...",
  "suggest_stop": false
}
```

Temperature 0.3 (deterministic-ish). Retries once on malformed JSON, then raises `ReflectionError`.

### Answer synthesis (`qa/prompts/answer.md`)

After the loop terminates, if `articles_distilled` is non-empty:
1. Concatenate up to 12K chars of each article's body as context
2. Single LLM call (temperature 0.5) producing JSON `{title, body, tags, cited_slugs}`
3. Body is **scrubbed** of `[[wikilinks]]` whose slug isn't in `valid_slugs` (the LLM sometimes invents slugs; we preserve display text but unlink)
4. Wrap with cited-articles table + audit trail (per-round query + LLM confidence) → write to `<vault>/surveys/qa-<slug>-<YYYYMMDD>.md`

### State persistence

After each round, `qa/state.py` writes:

```
<vault>/.paper_distiller/qa-sessions/<session_id>/state.json
```

Contains the full SessionState — `question`, `articles_distilled` (full bodies inline so resume doesn't need to re-distill), `articles_seen_ids` (as sorted JSON list, restored as Python set), `history` (per-round records), `last_reflection`, `tokens_in/out_total`, `cost_cny`, `stop_reason`, `is_done`.

A `--resume <session-id>` invocation re-reads this file, sets `state = existing`, and re-enters the while loop. Already-distilled articles are not re-fetched (their slugs are already in `articles_seen_ids`).

## Cost accounting

Aliyun Bailian `qwen-plus` / `qwen3.5-plus` pricing (rough, CNY per 1M tokens):
- Input: ¥2.1/M
- Output: ¥12.7/M

The LLMClient maintains `total_tokens_in` and `total_tokens_out` accumulators. After every round, `qa.loop._update_cost(state, llm)` rolls them into `state.cost_cny`. This is purely for the cost circuit breaker (`--max-cost-cny`); it is NOT billing-accurate and does not account for provider-specific overhead.

## Prompts as plain markdown

All 5 prompts live as plain markdown files:

- `src/paper_distiller/prompts/filter.md` (rank candidates)
- `src/paper_distiller/prompts/article.md` (distill one paper)
- `src/paper_distiller/prompts/survey.md` (compose multi-article survey)
- `src/paper_distiller/qa/prompts/reflect.md` (judge loop progress)
- `src/paper_distiller/qa/prompts/answer.md` (synthesize final answer)

Edit them directly to change tone, structure, or output language. No Python changes needed — they use Python `str.format()` interpolation with named parameters like `{question}`, `{articles_full}`, etc.

## LLM client contract

The `LLMClient` in `llm/openai_compatible.py` is a minimal HTTP wrapper:

- One `complete(messages, temperature, response_format=None)` method
- `response_format="json"` enables strict-JSON mode (provider-side `response_format: {type: json_object}`)
- Accumulates `total_tokens_in/out` across calls
- Raises `LLMError` on HTTP/timeout/auth failures (caught upstream — distillation errors don't crash the loop)

Any OpenAI-compatible endpoint works: Aliyun Bailian (recommended), DeepSeek, OpenRouter, local Ollama. See README for example configurations.

## Vault format

paper-distiller writes pure Obsidian-flavored markdown — no custom format:

- YAML frontmatter at the top (`title`, `tags`, `slug`, `arxiv_id`, `doi`, `published`, `depth`)
- Body in markdown
- Cross-references via `[[wikilink]]` or `[[wikilink|Display]]`
- Categories are subdirectory names (`articles/`, `surveys/`, `techniques/`, `directions/`, `open-problems/`, `authors/`)

The QA session state lives in `<vault>/.paper_distiller/qa-sessions/<sid>/state.json` — a hidden subdirectory that Obsidian ignores by default.

## Testing

78 tests across `tests/` (run `pytest -q`). All LLM calls are mocked (`unittest.mock` or `pytest-mock`); no real API calls in the test suite. Real-API smoke tests are documented in CHANGELOG entries; not part of CI.
