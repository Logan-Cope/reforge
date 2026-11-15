"""Synthetic relevance labels for the retrieval eval (standard pooling methodology).

The hand-labeled taste log covers too few beats for a meaningful clip-vs-hybrid A/B,
so we grow the labeled set synthetically: for every shot of every reference reel,
pool the CANDIDATE UNION of both retrievers (so neither arm is judged only on the
other's blind spots), then have a cheap TEXT-ONLY Haiku judge mark each pooled
candidate relevant/not-relevant to the shot's ask ("could this clip plausibly serve
this beat?" - a generous RELEVANCE bar, not final taste; taste lives in the rerank).

Labels are stored as explicit ``moment_id`` sets in ``eval_results/labeled-beats.json``
(NOT taste-log path fragments - ``dataset.resolve_fragment`` substring-matching is the
wrong tool for machine-generated ids) and scored by
``harness.run_retrieval_eval_from_labels``.

Pure helpers (``dedup_by_moment_id``, ``label_prompt``, ``parse_labels``,
``labeled_beat_record``) are unit-tested; ``pool_candidates`` / ``label_pool`` /
``make_client`` are thin I/O seams (DB retrieve, anthropic SDK).
"""
from __future__ import annotations

from ..llm.base import ClipMoment, ReelBrief, ShotSpec

# The judged pool: union of both retrievers' top-POOL_K, deduped by moment id.
POOL_K = 30

_RELEVANCE_TOOL = {
    "name": "record_relevance",
    "description": "Record, for every candidate index, whether the clip could plausibly serve this beat.",
    "input_schema": {
        "type": "object",
        "properties": {
            "labels": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "candidate": {"type": "integer", "description": "candidate index"},
                        "relevant": {"type": "boolean",
                                     "description": "true if this clip could plausibly serve this beat"},
                    },
                    "required": ["candidate", "relevant"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["labels"],
        "additionalProperties": False,
    },
}


def dedup_by_moment_id(candidate_lists: list[list[ClipMoment]]) -> list[ClipMoment]:
    """Union of candidate lists, deduped by ``moment_id``, first occurrence wins (pure).

    Order is stable: list order, then rank within each list - so the pool has no
    retriever-specific ordering signal beyond first-seen.
    """
    seen: set = set()
    pooled: list[ClipMoment] = []
    for candidates in candidate_lists:
        for c in candidates:
            if c.moment_id in seen:
                continue
            seen.add(c.moment_id)
            pooled.append(c)
    return pooled


def pool_candidates(shot: ShotSpec, ref_frame: str | None, brief: ReelBrief | None = None,
                    k: int = POOL_K, retrieve_fn=None) -> list[ClipMoment]:
    """Candidate union of BOTH retrievers for one shot (pooling; I/O unless injected)."""
    if retrieve_fn is None:
        from ..match.retrieve import retrieve as retrieve_fn  # pragma: no cover - I/O seam
    return dedup_by_moment_id([
        retrieve_fn(shot, ref_frame, k=k, retriever="clip"),
        retrieve_fn(shot, ref_frame, k=k, retriever="hybrid", brief=brief),
    ])


def label_prompt(shot: ShotSpec, brief: ReelBrief | None, candidate_lines: list[str]) -> str:
    """The Haiku relevance-judge prompt: shot ask + brief + numbered candidate metadata (pure)."""
    from ..match.retrieve import brief_summary, shot_ask

    listing = "\n".join(f"[{i}] {line}" for i, line in enumerate(candidate_lines))
    parts = []
    if brief is not None:
        parts.append("Reel creative brief:\n" + brief_summary(brief))
    parts.append("This beat needs:\n" + shot_ask(shot))
    parts.append("Candidate clips (stored metadata only):\n" + listing)
    parts.append(
        "For EVERY candidate index, record via record_relevance whether the clip could "
        "PLAUSIBLY serve this beat. This is a RELEVANCE judgment, not a final taste pick - "
        "be GENEROUS: mark relevant=true when the action, energy, or setting is in the "
        "right neighborhood for the beat's role and feeling, even if it would not be the "
        "single best pick. Mark relevant=false only when the clip clearly cannot serve "
        "this beat (wrong kind of action AND wrong energy/mood). Label every index."
    )
    return "\n\n".join(parts)


def parse_labels(result: dict, n_candidates: int) -> list[bool]:
    """record_relevance tool output -> one bool per candidate index (pure, tolerant).

    Unmentioned indices default to False; out-of-range / non-int indices and junk rows
    are skipped; ``relevant`` is coerced with ``bool``. Never raises.
    """
    flags = [False] * n_candidates
    rows = result.get("labels") if isinstance(result, dict) else None
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        ci = row.get("candidate")
        if isinstance(ci, bool) or not isinstance(ci, int) or not (0 <= ci < n_candidates):
            continue
        flags[ci] = bool(row.get("relevant", False))
    return flags


def make_client():
    """An anthropic client wired like the rest of reforge (gateway-aware, traced). I/O seam."""
    import anthropic

    from ..config import settings
    from ..llm.claude import _client_config, _maybe_trace

    api_key, base_url = _client_config(
        settings.anthropic_api_key, settings.ai_gateway_api_key, settings.ai_gateway_base_url)
    return _maybe_trace(anthropic.Anthropic(api_key=api_key, base_url=base_url,
                                            timeout=180.0, max_retries=3))


def label_pool(client, shot: ShotSpec, brief: ReelBrief | None,
               candidates: list[ClipMoment], model: str = "claude-haiku-4-5",
               usage: dict | None = None) -> list[tuple[int, bool]]:
    """TEXT-ONLY Haiku relevance labels for a pooled candidate list.

    One forced-tool call per beat over the candidates' stored metadata lines (no
    images). Returns ``(moment_id, relevant)`` per candidate, pool order. ``usage``
    (optional) accumulates ``{"input", "output", "calls"}`` for cost reporting.
    """
    from ..match.retrieve import format_candidate_meta

    lines = [format_candidate_meta(c) for c in candidates]
    response = client.messages.create(
        model=model,
        max_tokens=4096,
        tools=[_RELEVANCE_TOOL],
        tool_choice={"type": "tool", "name": "record_relevance"},
        messages=[{"role": "user", "content": label_prompt(shot, brief, lines)}],
    )
    if usage is not None:
        u = getattr(response, "usage", None)
        usage["input"] = usage.get("input", 0) + (getattr(u, "input_tokens", 0) or 0)
        usage["output"] = usage.get("output", 0) + (getattr(u, "output_tokens", 0) or 0)
        usage["calls"] = usage.get("calls", 0) + 1
    result = next(b.input for b in response.content if b.type == "tool_use")
    flags = parse_labels(result, len(candidates))
    return [(c.moment_id, flag) for c, flag in zip(candidates, flags)]


def labeled_beat_record(run: str, shot: ShotSpec, brief: ReelBrief | None,
                        labels: list[tuple[int, bool]]) -> dict:
    """One ``labeled-beats.json`` record: explicit relevant moment ids + the query context (pure).

    ``query_meta`` (the shot's ask) and ``brief`` are stored so the from-labels eval can
    rebuild the EXACT query both retrievers were labeled against - no LLM re-read drift.
    """
    record = {
        "run": run,
        "beat": shot.idx,
        "query_meta": {
            "description": shot.description,
            "query": shot.query,
            "energy": shot.energy,
            "role": shot.role,
            "intended_feeling": shot.intended_feeling,
            "intended_energy": shot.intended_energy,
        },
        "relevant_moment_ids": [mid for mid, relevant in labels if relevant],
        "pooled": len(labels),
        "relevant": sum(1 for _, relevant in labels if relevant),
    }
    if brief is not None:
        record["brief"] = {
            "vibe": brief.vibe,
            "emotional_register": brief.emotional_register,
            "energy_profile": brief.energy_profile,
            "tone": brief.tone,
            "theme": brief.theme,
        }
    return record
