"""LLM reflection call for the QA loop.

Wraps a single LLM invocation that produces structured JSON describing the
loop's progress: whether the question is answered, what's missing, and the
next query to try if not.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..llm.openai_compatible import LLMClient


class ReflectionError(RuntimeError):
    pass


_PROMPT_FILE = Path(__file__).parent / "prompts" / "reflect.md"

_REQUIRED_KEYS = {
    "is_done", "confidence", "what_we_know", "what_is_missing",
    "next_query", "next_query_rationale", "suggest_stop",
}


def _render_prompt(
    question: str,
    articles_summary: list,
    prior_queries: list,
    round_num: int,
    max_rounds: int,
) -> str:
    if articles_summary:
        articles_block = "\n".join(f"- {s}" for s in articles_summary)
    else:
        articles_block = "(尚无已蒸馏的论文)"
    if prior_queries:
        queries_block = "\n".join(f"- {q}" for q in prior_queries)
    else:
        queries_block = "(本轮是第一次搜索)"
    return _PROMPT_FILE.read_text(encoding="utf-8").format(
        round_num=round_num,
        max_rounds=max_rounds,
        question=question,
        n_articles=len(articles_summary),
        articles_summary=articles_block,
        prior_queries=queries_block,
    )


def _parse_response(raw: str) -> dict:
    parsed = json.loads(raw)
    missing = _REQUIRED_KEYS - set(parsed.keys())
    if missing:
        raise ValueError(f"reflection JSON missing keys: {missing}")
    return parsed


def reflect(
    question: str,
    articles_summary: list,
    prior_queries: list,
    round_num: int,
    max_rounds: int,
    llm: LLMClient,
) -> dict:
    """One reflection call. Retries once on malformed JSON; raises on second failure."""
    prompt = _render_prompt(question, articles_summary, prior_queries,
                            round_num, max_rounds)
    messages = [{"role": "user", "content": prompt}]
    for attempt in (1, 2):
        raw = llm.complete(messages, temperature=0.3, response_format="json")
        try:
            return _parse_response(raw)
        except (json.JSONDecodeError, ValueError):
            if attempt == 2:
                raise ReflectionError(
                    f"reflection returned malformed JSON twice: {raw[:200]}"
                )
            continue
    raise ReflectionError("unreachable")  # pragma: no cover
