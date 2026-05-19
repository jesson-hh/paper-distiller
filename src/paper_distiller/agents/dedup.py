"""CandidateDedup — filters shared['candidates'] against qa_state.articles_seen_ids.

QA-mode only. In single-pass (no qa_state), passes candidates through unchanged.
A paper is filtered if either its arxiv_id or doi appears in articles_seen_ids
(which tracks arxiv_id ∪ doi across rounds in the same session).
"""

from __future__ import annotations

from .base import Context


class CandidateDedup:
    name = "candidate-dedup"
    deps = ["candidate-merger"]

    async def run(self, ctx: Context) -> dict:
        candidates = ctx.shared.get("candidates", [])
        state = ctx.shared.get("qa_state")
        if state is None:
            return {"candidates": candidates}
        seen = state.articles_seen_ids
        if not seen:
            return {"candidates": candidates}
        filtered = []
        for p in candidates:
            ids = {pid for pid in (p.arxiv_id, p.doi) if pid}
            if ids & seen:
                continue
            filtered.append(p)
        return {"candidates": filtered}
