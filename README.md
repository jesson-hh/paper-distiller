# paper-distiller

> Distill arxiv research papers into an Obsidian-compatible markdown wiki.

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python: 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)

## What it does

Give `paper-distiller` a research topic or an author name, and it will:

1. Search arxiv for relevant papers
2. Use an LLM to rank the top N most relevant
3. Download each PDF and extract the text
4. Distill each paper into a structured Chinese markdown note (一句话 / 问题动因 / 方法 / 关键结果 / 我的 take)
5. Cross-link each note to existing entries in your Obsidian vault via `[[wikilinks]]`
6. Compose a session survey tying the new notes together

The output drops directly into a vault directory that opens in Obsidian — graph view, Dataview, tag pane, full-text search all work out of the box.

## Install (v0.1.0 — from source)

    git clone https://github.com/jesson-hh/paper-distiller
    cd paper-distiller
    python -m venv .venv
    .venv\Scripts\activate          # Windows
    # source .venv/bin/activate     # Linux/macOS
    pip install -e .

## Quick start

1. Copy the config template and fill in your LLM API key:

       cp examples/example.env .env
       # then edit .env, replace PD_API_KEY=sk-your-key-here with your real key

2. Point it at your Obsidian vault:

       paper-distiller --vault /path/to/your/vault --topic "diffusion models for finance" --n 5

3. Open your vault in Obsidian. New articles appear under `articles/`, a session survey under `surveys/`.

## How it works

```
search arxiv  →  LLM filter  →  download PDFs  →  distill each  →  save articles  →  compose survey
   (~30 hits)     (→ top N)        (PyMuPDF)       (LLM call)       (markdown +       (LLM call)
                                                                     frontmatter)
```

Per-paper cost on `qwen-plus` / `qwen3.5-plus` (Aliyun Bailian): roughly $0.02 per paper. A 5-paper run is around $0.10 USD (~¥0.70).

## Vault layout

paper-distiller writes into a vault with these category subdirectories (created on first run):

| Category | What goes there |
|---|---|
| `articles/` | Paper notes — one entry per paper |
| `surveys/` | Cluster mini-surveys composed by paper-distiller, linking multiple articles |
| `techniques/`, `directions/`, `open-problems/`, `authors/` | Reserved for human-curated content. paper-distiller doesn't write here in v0.1. |

Frontmatter and `[[wikilinks]]` follow Obsidian conventions — no custom format.

## Configuration

| Env var | Purpose | Default |
|---|---|---|
| `PD_API_KEY` | LLM API key (Aliyun Bailian, DeepSeek, OpenRouter — any OpenAI-compatible) | required |
| `PD_BASE_URL` | Endpoint base URL | required |
| `PD_MODEL` | Model identifier | required |
| `PD_PROVIDER_NAME` | Logging tag only | `unspecified` |
| `PD_PDF_TIMEOUT` | PDF download timeout (seconds) | `60` |
| `PD_MIN_SURVEY` | Min articles before composing a survey | `2` |

CLI flags override env vars where applicable (`--model`, `--provider`).

## CLI reference

    paper-distiller --vault <path> {--topic <str> | --author <str>}
                    [--n 5] [--pool 30] [--force] [--dry-run]
                    [--verbose] [--model <name>] [--provider <name>]

`--dry-run` skips all LLM calls and vault writes — useful for verifying config before spending API budget.

## Customizing prompts

The 3 distillation prompts live as plain markdown at `src/paper_distiller/prompts/{filter,article,survey}.md`. Edit them directly to change the tone, structure, or language of generated notes — no Python changes needed.

## LLM provider examples

| Provider | `PD_BASE_URL` | `PD_MODEL` |
|---|---|---|
| **Aliyun Bailian (recommended, cheapest)** | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `qwen-plus` |
| Aliyun Bailian (coding plan) | `https://coding.dashscope.aliyuncs.com/v1` | `qwen3.5-plus` |
| DeepSeek | `https://api.deepseek.com/v1` | `deepseek-chat` |
| OpenRouter | `https://openrouter.ai/api/v1` | `qwen/qwen3.5-plus` |
| Local Ollama | `http://localhost:11434/v1` | `qwen2.5` |

## Why "math research" specifically?

The default category schema (articles / techniques / directions / open-problems / authors / surveys) was designed for mathematical/scientific paper research. The tool works fine for other domains today; configurable schema is on the v0.2 roadmap.

## Status

**v0.1.0 — alpha.** Single-pass L2 search-and-distill works end-to-end. Not yet on PyPI; install from GitHub.

### Planned for v0.2

- Multi-round autonomous research loop (the "L3" mode from the design doc)
- Per-vault `paper-distiller.toml` for custom category schemas
- Semantic Scholar as a second source for open-access PDFs
- LEANN-backed semantic crosslink retrieval (replaces the pre-load index when vaults grow past ~500 entries)
- arxiv-id-based dedup (so re-distilling the same paper updates the existing entry instead of creating a sibling)
- Per-paper full-text length threshold (current truthy check accepts very short text as "full-pdf")

### Known limitations in v0.1.0

- arxiv.org occasionally returns 503 / 429; paper-distiller retries 3× then exits with a friendly error (use `--verbose` for the traceback).
- Dedup uses the paper title's slug; if an LLM rewrites the title differently between runs, a duplicate article may be created.
- The 4-byte 500-char "full-pdf vs abstract-only" threshold is currently a truthy check (any non-empty `full_text` counts as full-pdf). Real PDFs are tens of KB so this is rarely a problem in practice.

## Contributing

Issues and PRs welcome. Run tests before submitting:

    pip install -e ".[dev]"
    pytest -v

## License

MIT
