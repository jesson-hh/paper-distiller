"""ResearchState — checkpoint for the long-running deep-research loop."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class ResearchState:
    session_id: str
    question: str
    config_snapshot: dict
    started_at: str
    phase: str = "seed"  # seed / expand / structure / synthesize / gap / done
    papers_distilled: list = field(default_factory=list)   # list[slug]
    papers_seen_ids: list = field(default_factory=list)    # list[arxiv_id|doi]
    themes: list = field(default_factory=list)             # list[{name, description, slugs}]
    synthesis_slugs: list = field(default_factory=list)    # list[slug]
    structured_extractions: dict = field(default_factory=dict)  # {slug: {theorems, ...}}
    final_report_slug: str = ""
    total_cost_cny: float = 0.0
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    iterations_completed: int = 0
    is_done: bool = False
    stop_reason: str = ""


def _state_dir(vault_path: Path, session_id: str) -> Path:
    return Path(vault_path) / ".paper_distiller" / "research-sessions" / session_id


def write_research_state(vault_path: Path, state: ResearchState) -> None:
    """Persist the latest ResearchState snapshot to
    <vault>/.paper_distiller/research-sessions/<sid>/state.json.
    """
    d = _state_dir(vault_path, state.session_id)
    d.mkdir(parents=True, exist_ok=True)
    payload = asdict(state)
    (d / "state.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def read_research_state(vault_path: Path, session_id: str) -> ResearchState | None:
    """Read a previously persisted ResearchState. Returns None if not found."""
    d = _state_dir(vault_path, session_id)
    path = d / "state.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return ResearchState(**data)
