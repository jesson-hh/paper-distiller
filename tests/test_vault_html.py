"""Tests for HTML rendering on VaultStore.save_entry."""
from pathlib import Path

import pytest

from paper_distiller.vault.html_render import render_html
from paper_distiller.vault.store import VaultStore


def test_render_html_produces_full_doc():
    html = render_html("Test Title", "# Hello\n\nWorld")
    assert "<!DOCTYPE html>" in html
    assert "<title>Test Title</title>" in html
    assert "<h1>Hello</h1>" in html
    assert "<p>World</p>" in html


def test_render_html_expands_wikilinks():
    html = render_html(
        "T",
        "See [[cnf-convergence]] and [[ssm|state-space model]].",
    )
    assert '<a href="cnf-convergence.html">cnf-convergence</a>' in html
    assert '<a href="ssm.html">state-space model</a>' in html


def test_render_html_includes_mathjax():
    html = render_html("T", "Formula: $f(x) = x^2$")
    assert "mathjax" in html.lower()
    # Inline math stays as-is (MathJax processes it at render time)
    assert "$f(x) = x^2$" in html


def test_save_entry_writes_both_md_and_html(tmp_path):
    vault = VaultStore(tmp_path)
    vault.save_entry(
        title="Sample",
        category="articles",
        body="# Sample\n\nBody paragraph with [[other]] link.",
        tags=["t"],
        refs=["arxiv:x"],
        slug="sample-1",
    )
    md_path = tmp_path / "articles" / "sample-1.md"
    html_path = tmp_path / "articles" / "sample-1.html"
    assert md_path.exists()
    assert html_path.exists()
    html = html_path.read_text(encoding="utf-8")
    assert "Sample" in html
    assert '<a href="other.html">other</a>' in html


def test_save_entry_html_failure_does_not_break_md(tmp_path, mocker):
    """If render_html raises, .md must still be saved."""
    mocker.patch(
        "paper_distiller.vault.html_render.render_html",
        side_effect=RuntimeError("boom"),
    )
    vault = VaultStore(tmp_path)
    vault.save_entry(
        title="Sample",
        category="articles",
        body="body",
        slug="sample-2",
    )
    assert (tmp_path / "articles" / "sample-2.md").exists()
    # html may or may not exist — but md MUST exist
