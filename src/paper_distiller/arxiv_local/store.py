"""SQLite + FTS5 storage for arxiv metadata."""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


def _default_dir() -> Path:
    """Resolve the local mirror directory. Honors PD_ARXIV_LOCAL_DIR env so
    tests (and air-gapped users) can point the store somewhere isolated."""
    override = os.getenv("PD_ARXIV_LOCAL_DIR")
    if override:
        return Path(override)
    return Path(os.path.expanduser("~")) / ".paper-distiller" / "arxiv"


# Resolved at import; tests that need a fresh path should set the env
# BEFORE importing this module (autouse conftest fixture does this).
DEFAULT_DIR = _default_dir()
SCHEMA_VERSION = 1


@dataclass
class PaperRow:
    """One row in the papers table. Mirrors arxiv OAI-PMH metadata shape."""
    arxiv_id: str
    title: str
    authors: list
    abstract: str
    categories: list
    primary_category: str | None
    published: str
    updated: str | None
    doi: str | None
    comment: str | None
    journal_ref: str | None
    source: str  # 'bootstrap' | 'oai-pmh' | 'live-api'


_SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;

CREATE TABLE IF NOT EXISTS papers (
  arxiv_id    TEXT PRIMARY KEY,
  title       TEXT NOT NULL,
  authors     TEXT NOT NULL,
  abstract    TEXT NOT NULL,
  categories  TEXT NOT NULL,
  primary_category TEXT,
  published   TEXT NOT NULL,
  updated     TEXT,
  doi         TEXT,
  comment     TEXT,
  journal_ref TEXT,
  fetched_at  TEXT NOT NULL,
  source      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_papers_primary_category ON papers(primary_category);
CREATE INDEX IF NOT EXISTS idx_papers_published ON papers(published);

CREATE VIRTUAL TABLE IF NOT EXISTS papers_fts USING fts5(
  title, abstract,
  content='papers',
  content_rowid='rowid',
  tokenize='porter unicode61 remove_diacritics 2'
);

CREATE TRIGGER IF NOT EXISTS papers_ai AFTER INSERT ON papers BEGIN
  INSERT INTO papers_fts(rowid, title, abstract)
  VALUES (new.rowid, new.title, new.abstract);
END;

CREATE TRIGGER IF NOT EXISTS papers_ad AFTER DELETE ON papers BEGIN
  INSERT INTO papers_fts(papers_fts, rowid, title, abstract)
  VALUES('delete', old.rowid, old.title, old.abstract);
END;

CREATE TRIGGER IF NOT EXISTS papers_au AFTER UPDATE ON papers BEGIN
  INSERT INTO papers_fts(papers_fts, rowid, title, abstract)
  VALUES('delete', old.rowid, old.title, old.abstract);
  INSERT INTO papers_fts(rowid, title, abstract)
  VALUES (new.rowid, new.title, new.abstract);
END;

CREATE TABLE IF NOT EXISTS meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
"""


class Store:
    """Thin wrapper around SQLite. WAL mode for safe concurrent reads."""

    def __init__(self, db_path: Path | str):
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False — the agent's ArxivSearcher uses
        # asyncio.to_thread to run search in a worker thread, but creates
        # the Store on the main thread. SQLite's default cross-thread guard
        # would raise. With WAL mode (set in _SCHEMA) concurrent reads + a
        # single serialized writer are safe.
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        self._conn.execute(
            "INSERT OR IGNORE INTO meta(key, value) VALUES('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def paper_count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]

    def get_by_id(self, arxiv_id: str) -> PaperRow | None:
        row = self._conn.execute(
            "SELECT * FROM papers WHERE arxiv_id = ?", (arxiv_id,)
        ).fetchone()
        if row is None:
            return None
        return PaperRow(
            arxiv_id=row["arxiv_id"],
            title=row["title"],
            authors=json.loads(row["authors"]),
            abstract=row["abstract"],
            categories=json.loads(row["categories"]),
            primary_category=row["primary_category"],
            published=row["published"],
            updated=row["updated"],
            doi=row["doi"],
            comment=row["comment"],
            journal_ref=row["journal_ref"],
            source=row["source"],
        )

    def upsert_many(self, papers: Iterable[PaperRow]) -> int:
        """Insert or replace papers in a single transaction. Returns count."""
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        rows = [
            (
                p.arxiv_id, p.title, json.dumps(p.authors, ensure_ascii=False),
                p.abstract, json.dumps(p.categories), p.primary_category,
                p.published, p.updated, p.doi, p.comment, p.journal_ref,
                now, p.source,
            )
            for p in papers
        ]
        self._conn.executemany(
            """INSERT OR REPLACE INTO papers
               (arxiv_id, title, authors, abstract, categories, primary_category,
                published, updated, doi, comment, journal_ref, fetched_at, source)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
        self._conn.commit()
        return len(rows)

    def load_state(self) -> dict:
        out: dict = {
            "schema_version": SCHEMA_VERSION,
            "last_sync": None,
            "bootstrap_source": None,
            "bootstrap_completed_at": None,
        }
        for row in self._conn.execute("SELECT key, value FROM meta"):
            k = row["key"]
            v = row["value"]
            if k == "schema_version":
                out["schema_version"] = int(v)
            elif k in ("last_sync", "bootstrap_source", "bootstrap_completed_at"):
                out[k] = v
        return out

    def save_state(self, state: dict) -> None:
        for k, v in state.items():
            if v is None:
                continue
            self._conn.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
                (k, str(v)),
            )
        self._conn.commit()
