"""File-based Markdown wiki store for the personal math research knowledge base.

Directory layout:

    wiki/
      articles/         literature notes / paper summaries
      techniques/       methods, proof tricks
      directions/       research directions / programmes
      open-problems/    open problems / conjectures
      authors/          author-level distillation summaries (papers × problems × paths)
      raw/              raw metadata cache: raw/<source>/<id>.json

Each entry is a Markdown file with a YAML-like frontmatter block. Only a minimal
subset of YAML is supported (strings, inline lists) so we don't need PyYAML.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import threading
import unicodedata
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

WIKI_ROOT = Path("wiki")
CATEGORIES = ("articles", "techniques", "directions", "open-problems", "authors")
RAW_ROOT = WIKI_ROOT / "raw"

# Auto-commit wiki changes to git. Enabled when WIKI_AUTO_COMMIT=1 in the env.
WIKI_AUTO_COMMIT = os.environ.get("WIKI_AUTO_COMMIT", "0") == "1"
_GIT_LOCK = threading.Lock()


def _ensure_dirs() -> None:
    for c in CATEGORIES:
        (WIKI_ROOT / c).mkdir(parents=True, exist_ok=True)
    RAW_ROOT.mkdir(parents=True, exist_ok=True)


_ensure_dirs()


def _git_commit(paths: list, message: str) -> None:
    """Stage the given paths and commit. Silent on any failure (git missing,
    not a repo, nothing to commit, etc.). Runs in the caller thread — callers
    that want async should dispatch to a thread themselves."""
    if not WIKI_AUTO_COMMIT or not paths:
        return
    with _GIT_LOCK:
        try:
            r = subprocess.run(
                ["git", "rev-parse", "--is-inside-work-tree"],
                capture_output=True, text=True, timeout=5,
            )
            if r.returncode != 0 or "true" not in r.stdout:
                return
            subprocess.run(
                ["git", "add", "--"] + [str(p) for p in paths],
                capture_output=True, timeout=10,
            )
            r2 = subprocess.run(
                ["git", "diff", "--cached", "--quiet", "--"] + [str(p) for p in paths],
                capture_output=True, timeout=5,
            )
            if r2.returncode == 0:
                return  # nothing staged
            subprocess.run(
                ["git", "commit", "-m", message, "--no-verify",
                 "--no-gpg-sign", "--"] + [str(p) for p in paths],
                capture_output=True, timeout=15,
            )
        except Exception:
            return


def _git_commit_async(paths: list, message: str) -> None:
    if not WIKI_AUTO_COMMIT:
        return
    threading.Thread(
        target=_git_commit, args=(paths, message), daemon=True,
    ).start()


# ────────────────────────────────────────────────────────────────
# Slug + path helpers
# ────────────────────────────────────────────────────────────────

_SLUG_CLEAN = re.compile(r"[^a-z0-9\-]+")
_SLUG_DASH = re.compile(r"-{2,}")


def slugify(title: str) -> str:
    """Produce a filesystem-safe slug. Falls back to hash for CJK titles."""
    norm = unicodedata.normalize("NFKD", title)
    ascii_part = norm.encode("ascii", "ignore").decode("ascii").lower().strip()
    ascii_part = ascii_part.replace(" ", "-")
    ascii_part = _SLUG_CLEAN.sub("", ascii_part)
    ascii_part = _SLUG_DASH.sub("-", ascii_part).strip("-")
    if len(ascii_part) >= 3:
        return ascii_part[:80]
    # Fallback: short hash of the original title
    h = hashlib.sha1(title.encode("utf-8")).hexdigest()[:10]
    return f"entry-{h}"


def _validate_category(category: str) -> None:
    if category not in CATEGORIES:
        raise ValueError(f"Invalid category: {category!r}. Must be one of {CATEGORIES}.")


def _entry_path(category: str, slug: str) -> Path:
    _validate_category(category)
    return WIKI_ROOT / category / f"{slug}.md"


# ────────────────────────────────────────────────────────────────
# Minimal frontmatter serializer / parser
# ────────────────────────────────────────────────────────────────

def _fm_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def _fm_unescape(s: str) -> str:
    return s.replace('\\"', '"').replace("\\\\", "\\")


def _format_scalar(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, list):
        parts = [_format_scalar(x) for x in v]
        return "[" + ", ".join(parts) + "]"
    s = str(v)
    return '"' + _fm_escape(s) + '"'


def _parse_scalar(raw: str):
    raw = raw.strip()
    if not raw:
        return ""
    if raw.startswith("[") and raw.endswith("]"):
        inner = raw[1:-1].strip()
        if not inner:
            return []
        # Simple split that respects quoted strings
        items = []
        buf = ""
        in_quote = False
        esc = False
        for ch in inner:
            if esc:
                buf += ch
                esc = False
                continue
            if ch == "\\":
                esc = True
                continue
            if ch == '"':
                in_quote = not in_quote
                continue
            if ch == "," and not in_quote:
                items.append(_parse_scalar(buf))
                buf = ""
                continue
            buf += ch
        if buf.strip():
            items.append(_parse_scalar(buf))
        return items
    if raw.startswith('"') and raw.endswith('"'):
        return _fm_unescape(raw[1:-1])
    if raw in ("true", "false"):
        return raw == "true"
    try:
        if "." in raw:
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def dump_frontmatter(meta: dict, body: str) -> str:
    lines = ["---"]
    for k, v in meta.items():
        lines.append(f"{k}: {_format_scalar(v)}")
    lines.append("---")
    lines.append("")
    lines.append(body.rstrip() + "\n")
    return "\n".join(lines)


def parse_frontmatter(text: str) -> tuple[dict, str]:
    if not text.startswith("---"):
        return {}, text
    lines = text.split("\n")
    if lines[0].strip() != "---":
        return {}, text
    meta: dict = {}
    i = 1
    while i < len(lines) and lines[i].strip() != "---":
        line = lines[i]
        if ":" in line:
            k, _, v = line.partition(":")
            meta[k.strip()] = _parse_scalar(v)
        i += 1
    if i >= len(lines):
        return {}, text
    body = "\n".join(lines[i + 1:])
    # Strip one leading blank line if present
    if body.startswith("\n"):
        body = body[1:]
    return meta, body


# ────────────────────────────────────────────────────────────────
# Entry CRUD
# ────────────────────────────────────────────────────────────────

@dataclass
class Entry:
    slug: str
    category: str
    title: str
    tags: list
    refs: list
    links: list
    created: str
    updated: str
    body: str

    def to_dict(self) -> dict:
        return {
            "slug": self.slug,
            "category": self.category,
            "title": self.title,
            "tags": self.tags,
            "refs": self.refs,
            "links": self.links,
            "created": self.created,
            "updated": self.updated,
            "body": self.body,
        }


_WIKI_LINK_RE = re.compile(r"\[\[([^\]\|]+)(?:\|[^\]]+)?\]\]")


def _extract_links(body: str) -> list:
    return sorted(set(m.group(1).strip() for m in _WIKI_LINK_RE.finditer(body)))


def save_entry(
    title: str,
    category: str,
    body: str,
    tags: list | None = None,
    refs: list | None = None,
    slug: str | None = None,
) -> dict:
    """Create or update a wiki entry. If an entry with the slug exists, updates it
    (preserving created-at). Returns the entry metadata."""
    _validate_category(category)
    if not title.strip():
        raise ValueError("title is required")
    if not body.strip():
        raise ValueError("body is required")
    slug = slug or slugify(title)
    path = _entry_path(category, slug)

    now = datetime.now().isoformat(timespec="seconds")
    created = now
    if path.exists():
        existing_meta, _ = parse_frontmatter(path.read_text(encoding="utf-8"))
        created = existing_meta.get("created", now)

    links = _extract_links(body)
    meta = {
        "title": title,
        "category": category,
        "slug": slug,
        "tags": tags or [],
        "refs": refs or [],
        "links": links,
        "created": created,
        "updated": now,
    }
    path.write_text(dump_frontmatter(meta, body), encoding="utf-8")
    action = "update" if created != now else "add"
    _git_commit_async([path], f"wiki: {action} {category}/{slug}")
    return {**meta, "path": str(path).replace("\\", "/")}


def read_entry(category: str, slug: str) -> Entry | None:
    path = _entry_path(category, slug)
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")
    meta, body = parse_frontmatter(text)
    return Entry(
        slug=meta.get("slug", slug),
        category=meta.get("category", category),
        title=meta.get("title", slug),
        tags=meta.get("tags") or [],
        refs=meta.get("refs") or [],
        links=meta.get("links") or [],
        created=meta.get("created", ""),
        updated=meta.get("updated", ""),
        body=body,
    )


def delete_entry(category: str, slug: str) -> bool:
    path = _entry_path(category, slug)
    if path.exists():
        path.unlink()
        _git_commit_async([path], f"wiki: remove {category}/{slug}")
        return True
    return False


def list_entries(category: str | None = None) -> list:
    """List entries as metadata dicts (no body). Sorted by updated desc."""
    cats = [category] if category else list(CATEGORIES)
    for c in cats:
        _validate_category(c)
    items = []
    for c in cats:
        folder = WIKI_ROOT / c
        if not folder.exists():
            continue
        for f in folder.glob("*.md"):
            try:
                text = f.read_text(encoding="utf-8")
                meta, _ = parse_frontmatter(text)
                items.append({
                    "slug": meta.get("slug", f.stem),
                    "category": meta.get("category", c),
                    "title": meta.get("title", f.stem),
                    "tags": meta.get("tags") or [],
                    "refs": meta.get("refs") or [],
                    "created": meta.get("created", ""),
                    "updated": meta.get("updated", ""),
                })
            except Exception:
                continue
    items.sort(key=lambda x: x.get("updated", ""), reverse=True)
    return items


# ────────────────────────────────────────────────────────────────
# Search (TF-IDF over tokens + CJK bigrams, with field weights)
# ────────────────────────────────────────────────────────────────

# A "word" is a run of letters/digits/underscore; pulled out separately from
# CJK which we bigram-tokenize so Chinese still gets recall without a real
# segmenter.
_WORD_RE = re.compile(r"[A-Za-z0-9_]+")
_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")

_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "of", "in", "on", "to", "for", "is",
    "are", "be", "this", "that", "it", "as", "by", "with", "from", "we",
    "can", "has", "have", "at", "which", "its", "not", "if", "but",
})


def _tokenize(text: str) -> list:
    text = (text or "").lower()
    tokens: list = []
    for m in _WORD_RE.finditer(text):
        w = m.group(0)
        if len(w) >= 2 and w not in _STOPWORDS:
            tokens.append(w)
    # CJK bigrams
    cjk = "".join(_CJK_RE.findall(text))
    for i in range(len(cjk) - 1):
        tokens.append(cjk[i:i + 2])
    return tokens


def _load_corpus(cats: list) -> list:
    corpus = []
    for c in cats:
        folder = WIKI_ROOT / c
        if not folder.exists():
            continue
        for f in folder.glob("*.md"):
            try:
                text = f.read_text(encoding="utf-8")
                meta, body = parse_frontmatter(text)
            except Exception:
                continue
            title = str(meta.get("title", ""))
            tags = " ".join(meta.get("tags") or [])
            doc_tokens = (
                _tokenize(title) * 4
                + _tokenize(tags) * 3
                + _tokenize(body)
            )
            corpus.append({
                "meta": meta,
                "body": body,
                "category": c,
                "fstem": f.stem,
                "tokens": doc_tokens,
                "tf": Counter(doc_tokens),
            })
    return corpus


def _snippet(body: str, query_tokens: list, width: int = 180) -> str:
    """Find the first body region that contains the most distinct query tokens."""
    if not body:
        return ""
    body_lower = body.lower()
    best_idx = -1
    best_hits = 0
    for tok in query_tokens:
        if not tok:
            continue
        idx = body_lower.find(tok)
        if idx < 0:
            continue
        # count how many distinct query tokens appear in a window around idx
        window_start = max(0, idx - width // 2)
        window_end = min(len(body_lower), idx + width // 2)
        window = body_lower[window_start:window_end]
        hits = sum(1 for t in set(query_tokens) if t and t in window)
        if hits > best_hits:
            best_hits = hits
            best_idx = idx
    if best_idx < 0:
        return body[:width].strip().replace("\n", " ")
    start = max(0, best_idx - width // 2)
    end = min(len(body), best_idx + width // 2)
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(body) else ""
    return (prefix + body[start:end].strip() + suffix).replace("\n", " ")


def search_entries(query: str, category: str | None = None, limit: int = 20) -> list:
    """TF-IDF-ranked search across title (weighted 4x), tags (3x), body.

    Falls back to substring matching if the query has no tokens (e.g. one CJK
    character)."""
    q = (query or "").strip()
    if not q:
        return []
    cats = [category] if category else list(CATEGORIES)
    for c in cats:
        _validate_category(c)

    corpus = _load_corpus(cats)
    if not corpus:
        return []

    query_tokens = _tokenize(q)
    # Fallback for queries that tokenize to nothing (e.g. single CJK char)
    if not query_tokens:
        substring = q.lower()
        hits = []
        for doc in corpus:
            title = str(doc["meta"].get("title", "")).lower()
            body = doc["body"].lower()
            tags = " ".join(doc["meta"].get("tags") or []).lower()
            score = 0
            if substring in title:
                score += 5
            if substring in tags:
                score += 3
            if substring in body:
                score += 1
            if score:
                hits.append((score, doc))
        hits.sort(key=lambda x: x[0], reverse=True)
        return [_hit_record(doc, score, [substring]) for score, doc in hits[:limit]]

    # Document frequency
    N = len(corpus)
    df: Counter = Counter()
    for doc in corpus:
        for term in set(doc["tokens"]):
            df[term] += 1

    results = []
    for doc in corpus:
        tf = doc["tf"]
        score = 0.0
        matched = 0
        for term in query_tokens:
            if term in tf:
                idf = math.log((N + 1) / (df[term] + 1)) + 1.0
                score += (1 + math.log(tf[term])) * idf
                matched += 1
        if matched == 0:
            continue
        score *= (1 + 0.2 * matched)  # reward multi-term matches
        results.append((score, doc))

    results.sort(key=lambda x: x[0], reverse=True)
    return [_hit_record(doc, score, query_tokens) for score, doc in results[:limit]]


def _hit_record(doc: dict, score: float, query_tokens: list) -> dict:
    meta = doc["meta"]
    return {
        "slug": meta.get("slug", doc["fstem"]),
        "category": meta.get("category", doc["category"]),
        "title": meta.get("title", doc["fstem"]),
        "tags": meta.get("tags") or [],
        "snippet": _snippet(doc["body"], query_tokens),
        "score": round(float(score), 4),
    }


# ────────────────────────────────────────────────────────────────
# Tag index
# ────────────────────────────────────────────────────────────────

def tag_index(category: str | None = None) -> list:
    """Return [{tag, count}] sorted by count desc."""
    cats = [category] if category else list(CATEGORIES)
    for c in cats:
        _validate_category(c)
    counter: Counter = Counter()
    for c in cats:
        folder = WIKI_ROOT / c
        if not folder.exists():
            continue
        for f in folder.glob("*.md"):
            try:
                meta, _ = parse_frontmatter(f.read_text(encoding="utf-8"))
            except Exception:
                continue
            for t in (meta.get("tags") or []):
                if isinstance(t, str) and t.strip():
                    counter[t.strip()] += 1
    return [{"tag": t, "count": n} for t, n in counter.most_common()]


def entries_by_tag(tag: str, category: str | None = None) -> list:
    tag = (tag or "").strip()
    if not tag:
        return []
    cats = [category] if category else list(CATEGORIES)
    for c in cats:
        _validate_category(c)
    items = []
    for c in cats:
        folder = WIKI_ROOT / c
        if not folder.exists():
            continue
        for f in folder.glob("*.md"):
            try:
                meta, _ = parse_frontmatter(f.read_text(encoding="utf-8"))
            except Exception:
                continue
            if tag in (meta.get("tags") or []):
                items.append({
                    "slug": meta.get("slug", f.stem),
                    "category": meta.get("category", c),
                    "title": meta.get("title", f.stem),
                    "tags": meta.get("tags") or [],
                    "refs": meta.get("refs") or [],
                    "created": meta.get("created", ""),
                    "updated": meta.get("updated", ""),
                })
    items.sort(key=lambda x: x.get("updated", ""), reverse=True)
    return items


# ────────────────────────────────────────────────────────────────
# Backlinks
# ────────────────────────────────────────────────────────────────

def backlinks(slug: str) -> list:
    """Return entries whose body contains [[slug]] reference."""
    results = []
    for c in CATEGORIES:
        folder = WIKI_ROOT / c
        if not folder.exists():
            continue
        for f in folder.glob("*.md"):
            try:
                text = f.read_text(encoding="utf-8")
                meta, body = parse_frontmatter(text)
            except Exception:
                continue
            if slug in _extract_links(body):
                if meta.get("slug") == slug:
                    continue
                results.append({
                    "slug": meta.get("slug", f.stem),
                    "category": meta.get("category", c),
                    "title": meta.get("title", f.stem),
                })
    return results


# ────────────────────────────────────────────────────────────────
# Raw metadata cache
# ────────────────────────────────────────────────────────────────

_SAFE_ID = re.compile(r"[^A-Za-z0-9._\-]")


def _raw_path(source: str, ident: str) -> Path:
    source_safe = _SAFE_ID.sub("_", source.strip().lower()) or "unknown"
    ident_safe = _SAFE_ID.sub("_", ident.strip()) or "unnamed"
    folder = RAW_ROOT / source_safe
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{ident_safe}.json"


def save_raw(source: str, ident: str, data: dict) -> dict:
    """Save raw metadata (e.g. arxiv record) to the raw cache."""
    path = _raw_path(source, ident)
    payload = {
        "source": source,
        "id": ident,
        "fetched": datetime.now().isoformat(timespec="seconds"),
        "data": data,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"source": source, "id": ident, "path": str(path).replace("\\", "/")}


def read_raw(source: str, ident: str) -> dict | None:
    path = _raw_path(source, ident)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def list_raw(source: str | None = None) -> list:
    items = []
    sources = [source] if source else [p.name for p in RAW_ROOT.iterdir() if p.is_dir()]
    for s in sources:
        folder = RAW_ROOT / s
        if not folder.exists():
            continue
        for f in folder.glob("*.json"):
            try:
                obj = json.loads(f.read_text(encoding="utf-8"))
                items.append({
                    "source": obj.get("source", s),
                    "id": obj.get("id", f.stem),
                    "fetched": obj.get("fetched", ""),
                })
            except Exception:
                continue
    items.sort(key=lambda x: x.get("fetched", ""), reverse=True)
    return items
