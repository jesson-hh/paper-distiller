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
