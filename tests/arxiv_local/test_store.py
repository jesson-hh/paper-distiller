"""Tests for arxiv_local.store — SQLite + FTS5."""

from __future__ import annotations


def test_store_creates_schema_on_open(tmp_path):
    from paper_distiller.arxiv_local.store import Store

    db_path = tmp_path / "arxiv.db"
    store = Store(db_path)
    assert db_path.exists()
    tables = {row[0] for row in store._conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    )}
    assert "papers" in tables
    assert "papers_fts" in tables
    store.close()


def test_paper_count_zero_on_fresh_store(tmp_path):
    from paper_distiller.arxiv_local.store import Store

    store = Store(tmp_path / "arxiv.db")
    assert store.paper_count() == 0
    store.close()


def test_upsert_single_paper(tmp_path):
    from paper_distiller.arxiv_local.store import Store, PaperRow

    store = Store(tmp_path / "arxiv.db")
    p = PaperRow(
        arxiv_id="2401.12345",
        title="A Diffusion Model for Test",
        authors=["Alice", "Bob"],
        abstract="We propose a fake method to test the store.",
        categories=["cs.LG", "stat.ML"],
        primary_category="cs.LG",
        published="2024-01-15",
        updated=None,
        doi=None,
        comment=None,
        journal_ref=None,
        source="bootstrap",
    )
    store.upsert_many([p])
    assert store.paper_count() == 1
    fetched = store.get_by_id("2401.12345")
    assert fetched.title == "A Diffusion Model for Test"
    assert fetched.authors == ["Alice", "Bob"]
    assert fetched.categories == ["cs.LG", "stat.ML"]
    store.close()


def test_upsert_idempotent_on_same_id(tmp_path):
    from paper_distiller.arxiv_local.store import Store, PaperRow

    store = Store(tmp_path / "arxiv.db")
    p = PaperRow(
        arxiv_id="2401.12345", title="Old Title", authors=["A"],
        abstract="old", categories=["cs.LG"], primary_category="cs.LG",
        published="2024-01-15", updated=None, doi=None, comment=None,
        journal_ref=None, source="bootstrap",
    )
    store.upsert_many([p])
    p2 = PaperRow(
        arxiv_id="2401.12345", title="New Title", authors=["A"],
        abstract="updated", categories=["cs.LG"], primary_category="cs.LG",
        published="2024-01-15", updated="2024-02-01", doi=None,
        comment=None, journal_ref=None, source="oai-pmh",
    )
    store.upsert_many([p2])
    assert store.paper_count() == 1
    assert store.get_by_id("2401.12345").title == "New Title"
    store.close()


def test_fts_index_triggered_on_insert(tmp_path):
    from paper_distiller.arxiv_local.store import Store, PaperRow

    store = Store(tmp_path / "arxiv.db")
    store.upsert_many([PaperRow(
        arxiv_id="2401.0", title="Latent Diffusion Models", authors=["X"],
        abstract="We propose latent diffusion for image synthesis.",
        categories=["cs.CV"], primary_category="cs.CV",
        published="2024-01-01", updated=None, doi=None, comment=None,
        journal_ref=None, source="bootstrap",
    )])
    n_fts = store._conn.execute(
        "SELECT COUNT(*) FROM papers_fts"
    ).fetchone()[0]
    assert n_fts == 1
    store.close()


def test_state_load_default(tmp_path):
    from paper_distiller.arxiv_local.store import Store

    store = Store(tmp_path / "arxiv.db")
    state = store.load_state()
    assert state["schema_version"] >= 1
    assert state["last_sync"] is None
    store.close()


def test_state_save_and_reload(tmp_path):
    from paper_distiller.arxiv_local.store import Store

    db_path = tmp_path / "arxiv.db"
    store = Store(db_path)
    store.save_state({"last_sync": "2026-05-20T00:00:00Z", "schema_version": 1})
    store.close()

    store2 = Store(db_path)
    state = store2.load_state()
    assert state["last_sync"] == "2026-05-20T00:00:00Z"
    store2.close()
