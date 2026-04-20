# Author Entry Template

> Copy this file, rename to `<slug>.md`, place under `wiki/authors/`, then
> fill in. `_templates/` is outside `CATEGORIES`, so it is NOT indexed or
> searched — safe to keep alongside the real entries.

Slug rule: `<lastname>-<firstname>` (lowercase, ASCII). For CJK names,
ASCII-ify the romanization, e.g. `takahashi-tomonori`.

---

## Frontmatter skeleton

```yaml
---
title: "<First Last> — <one-line research focus>"
category: "authors"
slug: "<lastname-firstname>"
tags: ["author", "<domain-1>", "<domain-2>"]
affiliation: "<institution / lab>"
homepage: "<url, optional>"
scholar: "<google scholar id, optional>"
aliases: ["<name variants appearing in arxiv>"]
papers: ["arxiv:XXXX.XXXXX", "arxiv:YYYY.YYYYY"]
refs: ["arxiv:XXXX.XXXXX"]
links: ["<article-slug-if-we-have-a-deep-note>"]
first_seen: "YYYY-MM-DD"
last_updated: "YYYY-MM-DD"
---
```

## Body skeleton

```markdown
# <First Last>

## Profile
- **Affiliation**: …
- **Focus**: one or two sentences on what they work on
- **Coauthors I keep seeing**: [[coauthor-slug]], …
- **Links**: homepage / scholar / github (only if useful)

## 论文蒸馏 (最新在上)

### YYYY — <Paper short title> (arxiv:XXXX.XXXXX)
- **Problem 解决了什么**:
  1–3 bullets on the concrete problem / gap.
- **Path 技术路径**:
  compact pipeline — "X → Y → Z" or numbered steps.
  Keep it to ~5 items; a reader should be able to reconstruct the method
  without reading the paper.
- **Key insight**:
  one sentence — what the paper's *original* idea is.
- **Detailed note**: [[<article-slug>]] (or "TODO — not yet read in depth")
- **我的关注**:
  why this matters for my programme, or "no direct relevance, tracking only".

### YYYY — <Paper> (…)
…

## Connections

Relation-described wikilinks — human-readable complement to the `links:`
frontmatter (which is the machine-readable index). Every bullet should
state *how* two entries relate, not just *that* they do.

- **extends** [[other-article-or-author-slug]] by …
- **builds on** [[technique-slug]] — reuses …, modifies …
- **cited by** [[later-paper-slug]] — that paper's Sec. X picks up …
- **parallel to** [[sibling-line-slug]] — same goal, different method
- **applies to** [[direction-slug]] — this author's work is a plug-in for
  that programme

## Contradictions

Where this author's position conflicts with another entry (self or other).
Keep it specific — quote the disagreement, don't just gesture at it.

- On **<topic>**: [[paper-A]] argues X; [[paper-B]] argues ¬X. Evidence on
  each side: …. My current read: …
- On **<topic>**: …

Leave this section empty if no real disagreements surface — do **not**
fabricate tension. An empty `Contradictions:` is a signal that more
cross-reading is needed, not a problem.

## 追踪

- [ ] New arxiv releases since <YYYY-MM-DD>
- [ ] Deep-read next: <paper> (reason: …)
- [ ] Contact / engagement idea (if any)

## Raw metadata pointers
- arxiv author query cache: `wiki/raw/author/<slug>.json`
- individual paper cache:   `wiki/raw/arxiv/<arxiv-id>.json`
```

---

## Workflow

1. **Seed** — pick an author already present in your `articles/` notes.
2. **Gather** — `arxiv_tool` author-search → save results to
   `wiki/raw/author/<slug>.json`.
3. **Distil per-paper** — for each listed paper:
   - if an `articles/` note exists, lift `Problem / Path / Insight` from it
   - else leave the bullet as `TODO — not yet read` and flag it
4. **Connect** — add the author-slug to every relevant article's `authors:`
   frontmatter (you will need to extend the articles later; safe since it
   is an additive field).
5. **Cross-link** — put `[[author-slug]]` in relevant `directions/`,
   `techniques/`, `open-problems/` entries where attribution matters.
