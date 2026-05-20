"""Convert OAI-PMH 'arXiv' metadata records to PaperRow.

arxiv's OAI-PMH `metadataPrefix=arXiv` returns records with this nested
structure (sickle.Record exposes `.metadata` as a dict-of-lists). We
flatten to a flat PaperRow matching the bootstrap shape.
"""

from __future__ import annotations

from .store import PaperRow


def _first(d: dict, key: str) -> str | None:
    """OAI metadata values are wrapped in lists; pull first if present."""
    v = d.get(key)
    if isinstance(v, list) and v:
        return str(v[0])
    if isinstance(v, str):
        return v
    return None


def _flatten_authors(metadata: dict) -> list:
    """Flatten Sickle's nested author structure to ['First Last', ...]."""
    out = []
    raw = metadata.get("authors") or []
    if not raw:
        return out
    container = raw[0] if isinstance(raw, list) else raw
    name_records = container.get("author", []) if isinstance(container, dict) else []
    for nr in name_records:
        if not isinstance(nr, dict):
            continue
        last = " ".join(nr.get("keyname", []) or []).strip()
        first = " ".join(nr.get("forenames", []) or []).strip()
        full = f"{first} {last}".strip()
        if full:
            out.append(full)
    return out


def record_to_paper(record) -> PaperRow | None:
    """Convert a sickle Record to a PaperRow. Returns None for malformed records."""
    if record.deleted:
        return None
    md = record.metadata or {}
    arxiv_id = _first(md, "id")
    title = (_first(md, "title") or "").strip()
    if not arxiv_id or not title:
        return None
    abstract = (_first(md, "abstract") or "").strip()
    categories_raw = _first(md, "categories") or ""
    categories = [c for c in categories_raw.split() if c]

    return PaperRow(
        arxiv_id=arxiv_id,
        title=title,
        authors=_flatten_authors(md),
        abstract=abstract,
        categories=categories,
        primary_category=categories[0] if categories else None,
        published=_first(md, "created") or "",
        updated=_first(md, "updated"),
        doi=_first(md, "doi"),
        comment=_first(md, "comments"),
        journal_ref=_first(md, "journal-ref"),
        source="oai-pmh",
    )
