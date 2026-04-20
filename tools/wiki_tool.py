"""Wiki tools exposed to the LLM agent.

These let the agent build up the personal math knowledge base by saving
summaries, techniques, research directions, and open problems, and by
retrieving them when answering new questions.
"""

from __future__ import annotations

import wiki_store


def save_wiki_entry(
    title: str,
    category: str,
    content: str,
    tags: list | None = None,
    refs: list | None = None,
    slug: str | None = None,
) -> dict:
    """Create or update a wiki entry.

    `category` must be one of: articles, techniques, directions, open-problems, authors.
    `content` is the Markdown body (LaTeX via $…$/$$…$$, wiki-links via [[slug]]).
    Re-saving with the same title (or slug) updates the existing entry in place.
    """
    try:
        meta = wiki_store.save_entry(
            title=title,
            category=category,
            body=content,
            tags=tags or [],
            refs=refs or [],
            slug=slug,
        )
        return {"success": True, **meta}
    except ValueError as e:
        return {"success": False, "error": str(e)}


def search_wiki(query: str, category: str = "", limit: int = 10) -> dict:
    """Full-text search across the wiki. Returns ranked hits with snippets."""
    cat = category.strip() or None
    try:
        hits = wiki_store.search_entries(query, category=cat, limit=limit)
    except ValueError as e:
        return {"success": False, "error": str(e)}
    return {"success": True, "query": query, "total": len(hits), "hits": hits}


def read_wiki_entry(category: str, slug: str) -> dict:
    """Read a single wiki entry (metadata + body)."""
    entry = wiki_store.read_entry(category, slug)
    if entry is None:
        return {"success": False, "error": f"entry not found: {category}/{slug}"}
    return {"success": True, **entry.to_dict()}


def list_wiki_entries(category: str = "") -> dict:
    """List wiki entries, optionally filtered by category."""
    cat = category.strip() or None
    try:
        items = wiki_store.list_entries(category=cat)
    except ValueError as e:
        return {"success": False, "error": str(e)}
    return {"success": True, "total": len(items), "entries": items}


def save_raw_metadata(source: str, id: str, data: dict) -> dict:
    """Cache raw metadata from an external source (e.g. source='arxiv', id='2301.12345')."""
    if not source or not id:
        return {"success": False, "error": "source and id are required"}
    if not isinstance(data, dict):
        return {"success": False, "error": "data must be a JSON object"}
    meta = wiki_store.save_raw(source, id, data)
    return {"success": True, **meta}
