"""Tests for paper-distiller-arxiv CLI dispatch."""

from __future__ import annotations


def test_cli_stats_shows_zero_on_fresh_db(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        "paper_distiller.arxiv_local.cli.DEFAULT_DIR", tmp_path
    )
    from paper_distiller.arxiv_local.cli import main

    rc = main(["stats"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "papers:" in captured.out
    assert "0" in captured.out


def test_cli_search_returns_local_results(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        "paper_distiller.arxiv_local.cli.DEFAULT_DIR", tmp_path
    )
    from paper_distiller.arxiv_local.cli import main
    from paper_distiller.arxiv_local.store import Store, PaperRow

    store = Store(tmp_path / "arxiv.db")
    store.upsert_many([PaperRow(
        arxiv_id="2401.99", title="Local Diffusion Test", authors=["X"],
        abstract="abstract", categories=["cs.LG"], primary_category="cs.LG",
        published="2024-01-01", updated=None, doi=None, comment=None,
        journal_ref=None, source="bootstrap",
    )])
    store.close()

    rc = main(["search", "diffusion", "--n", "5"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "2401.99" in out


def test_cli_search_no_matches(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        "paper_distiller.arxiv_local.cli.DEFAULT_DIR", tmp_path
    )
    from paper_distiller.arxiv_local.cli import main

    rc = main(["search", "asdfqwerty"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "(no matches)" in out


def test_cli_bootstrap_refuses_existing_db_without_force(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "paper_distiller.arxiv_local.cli.DEFAULT_DIR", tmp_path
    )
    from paper_distiller.arxiv_local.cli import main
    from paper_distiller.arxiv_local.store import Store, PaperRow

    store = Store(tmp_path / "arxiv.db")
    store.upsert_many([PaperRow(
        arxiv_id="2401.0", title="X", authors=["A"], abstract="x",
        categories=["cs.LG"], primary_category="cs.LG",
        published="2024-01-01", updated=None, doi=None, comment=None,
        journal_ref=None, source="bootstrap",
    )])
    store.close()

    rc = main(["bootstrap"])
    assert rc == 2


def test_cli_doctor_runs_without_crash(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        "paper_distiller.arxiv_local.cli.DEFAULT_DIR", tmp_path
    )
    from paper_distiller.arxiv_local.cli import main

    rc = main(["doctor"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "doctor" in out
    assert "paper count:" in out
