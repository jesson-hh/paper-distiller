"""GapDetector — LLM judges whether the research loop should continue or stop."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path


_PROMPT_FILE = Path(__file__).parent / "prompts" / "gap.md"

_STOP_FALLBACK = {
    "should_continue": False,
    "missing_aspects": [],
    "next_query": "",
    "rationale": "Gap detector failed — conservatively stopping.",
}


def _themes_summary(themes: list) -> str:
    if not themes:
        return "(no themes yet)"
    parts = []
    for t in themes:
        name = t.get("name", "?")
        n_slugs = len(t.get("slugs", []))
        parts.append(f"{name} ({n_slugs} articles)")
    return ", ".join(parts)


class GapDetector:
    name = "gap-detector"
    deps: list[str] = []

    async def run(self, ctx) -> dict:
        state = ctx.shared.get("research_state")
        if state is None:
            return {"gap_analysis": dict(_STOP_FALLBACK)}

        prompt = _PROMPT_FILE.read_text(encoding="utf-8").format(
            question=state.question,
            n_papers=len(state.papers_distilled),
            themes_summary=_themes_summary(state.themes),
            n_syntheses=len(state.synthesis_slugs),
            iterations=state.iterations_completed,
            slugs_list="\n".join(f"- {s}" for s in state.papers_distilled[:30])
                       or "(none)",
        )
        messages = [{"role": "user", "content": prompt}]
        for attempt in (1, 2):
            raw = await asyncio.to_thread(
                ctx.llm.complete, messages, temperature=0.3, response_format="json",
            )
            try:
                parsed = json.loads(raw)
                # Validate shape
                if not isinstance(parsed.get("should_continue"), bool):
                    raise ValueError("missing should_continue bool")
                return {"gap_analysis": {
                    "should_continue": parsed["should_continue"],
                    "missing_aspects": parsed.get("missing_aspects", []) or [],
                    "next_query": parsed.get("next_query", "") or "",
                    "rationale": parsed.get("rationale", "") or "",
                }}
            except (json.JSONDecodeError, ValueError):
                if attempt == 2:
                    return {"gap_analysis": dict(_STOP_FALLBACK)}
                continue
        return {"gap_analysis": dict(_STOP_FALLBACK)}
