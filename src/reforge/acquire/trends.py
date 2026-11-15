"""Trend context via Tavily web search (the agent's external-search tool).

When deconstructing a reference reel, search the public web for what this trend/style
is and fold a compact digest into the ReelBrief, so the judge (shortlist + rerank)
understands WHY the reel works - not just what it looks like. Enrichment only: any
failure (no key, network, non-200) yields "" and the pipeline proceeds unchanged.

`_trend_query` and `_digest` are pure (unit-tested).
"""
from __future__ import annotations

import requests

from ..config import settings
from ..llm.base import ReelBrief

TAVILY_URL = "https://api.tavily.com/search"
_MAX_QUERY_WORDS = 15


# --- pure helpers -----------------------------------------------------------

def _trend_query(brief: ReelBrief, overlay_text: str, ref_name: str) -> str:
    """Compose a short web-search query (<= ~15 words) from theme/vibe + overlay + ref name.

    e.g. 'instagram reel trend "DISCIPLINE" late-night grind discipline why it works'.
    """
    words: list[str] = ["instagram", "reel", "trend"]
    seen = {w.lower() for w in words}

    def add(text: str, limit: int) -> list[str]:
        fresh = [w for w in text.split() if w.strip('"').lower() not in seen][:limit]
        seen.update(w.strip('"').lower() for w in fresh)
        return fresh

    overlay = " ".join(overlay_text.split()[:4])
    if overlay:
        words.append(f'"{overlay}"')
        seen.update(w.lower() for w in overlay.split())
    words += add(brief.theme, 3)
    words += add(brief.vibe, 3)
    stem = ref_name.rsplit("/", 1)[-1].rsplit(".", 1)[0].replace("_", " ").replace("-", " ")
    words += add(stem, 2)
    words += ["why", "it", "works"]
    return " ".join(words[:_MAX_QUERY_WORDS])


def _digest(results: list[dict], max_chars: int = 600) -> str:
    """Join result titles + content snippets into one compact trend-context paragraph.

    Truncates cleanly at a word boundary (ellipsis) when over ``max_chars``.
    """
    parts = []
    for r in results:
        title = (r.get("title") or "").strip()
        content = (r.get("content") or "").strip()
        piece = ": ".join(p for p in (title, content) if p)
        if piece:
            parts.append(piece)
    text = " | ".join(parts)
    if len(text) <= max_chars:
        return text
    cut = text[: max_chars - 1]
    if " " in cut:
        cut = cut[: cut.rfind(" ")].rstrip()
    return cut + "…"


# --- I/O --------------------------------------------------------------------

def trend_context(brief: ReelBrief, overlay_text: str = "", ref_name: str = "") -> str:
    """Search Tavily for what this trend is; return a compact digest ("" on any failure)."""
    if not settings.tavily_api_key:
        return ""
    try:
        resp = requests.post(TAVILY_URL, json={
            "api_key": settings.tavily_api_key,
            "query": _trend_query(brief, overlay_text, ref_name),
            "max_results": 4,
            "search_depth": "basic",
        }, timeout=20)
        if resp.status_code != 200:
            return ""
        return _digest(resp.json().get("results") or [])
    except Exception:
        return ""
