"""Shared pytest fixtures."""
import pytest
from pathlib import Path


@pytest.fixture
def tmp_vault(tmp_path: Path) -> Path:
    """A clean temporary vault directory for tests that need one."""
    vault = tmp_path / "vault"
    vault.mkdir()
    return vault


@pytest.fixture(autouse=True)
def _isolate_arxiv_local(tmp_path_factory, monkeypatch):
    """Point PD_ARXIV_LOCAL_DIR at an empty tmp dir for every test.

    The local arxiv mirror lives at ~/.paper-distiller/arxiv/ by default —
    a real user DB with thousands of papers can leak into integration tests
    and change their behavior (LocalFirstFetcher takes the local path with
    not enough results, then tries to top up via live arxiv API, hits 429).
    Per-test isolation eliminates this cross-environment dependency.
    """
    isolated = tmp_path_factory.mktemp("arxiv_local_isolated")
    monkeypatch.setenv("PD_ARXIV_LOCAL_DIR", str(isolated))
    yield


@pytest.fixture(autouse=True)
def _reset_rate_limiters():
    """Reset the module-level SourceLimiters between every test.

    The singletons hold last-call timestamps + cooldown state that would
    otherwise bleed across tests — e.g. a searcher test that triggers a
    cooldown would block the next integration test's searcher, surfacing
    as confusing "skipping" warnings and empty candidate lists.
    """
    from paper_distiller.agents.rate_limit import (
        ARXIV_LIMITER, SS_LIMITER, OPENALEX_LIMITER,
    )
    ARXIV_LIMITER.reset()
    SS_LIMITER.reset()
    OPENALEX_LIMITER.reset()
    yield
    ARXIV_LIMITER.reset()
    SS_LIMITER.reset()
    OPENALEX_LIMITER.reset()
