You are a research literature filter. Given a research topic and a list of arxiv paper candidates, select the top {top_n} most relevant papers for distillation into a research wiki.

# Research topic
{topic}

# Candidates (JSON list)
{candidates_json}

# Selection criteria
- Direct relevance to the research topic (most important)
- Methodological depth — prefer papers with novel methods, frameworks, or proofs over pure applications
- Recency — prefer recent work unless an older paper is foundational
- Diversity — if many candidates are very similar, prefer broader coverage over duplicates

# Output format
Return strictly JSON of this shape (no commentary, no markdown fence):

{{"selected": [{{"arxiv_id": "...", "relevance_score": 9.5, "reason": "..."}}, ...]}}

Exactly {top_n} entries. Order most-relevant first. Only use arxiv_id values from the candidates list above.
