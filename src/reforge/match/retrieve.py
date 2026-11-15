"""Retrieval: per reference shot, candidate moments from the embedding lanes.

Two retrievers (settings.retriever / the ``retriever`` arg):

- "clip" - the original single-lane path: CLIP image kNN on the reference frame,
  optionally merged with a CLIP-text query (regression path for the 9B A/B).
- "hybrid" (Phase 9B, default) - adds the lanes that see action/energy, not just
  visual look-alikes: caption-text kNN (OpenAI embedding of a HyDE-style intent
  query composed from the shot + brief) and BM25 full-text over captions, fused
  with the CLIP image lane by Reciprocal Rank Fusion (k=60), quality-weighted.

Candidates are filtered to "exclude clips with a non-Logan person" - v1 =
`is_logan=true` (also drops the lone object/no-person clip, which is moot for
fitness; the proper `subject` enum swap is a one-line change in `retrieval_filters`).
Returns top-K `ClipMoment` candidates.

The pure helpers (`retrieval_filters`, `merge_candidates`, `rrf_fuse`, `hyde_text`,
`row_to_moment`) are unit-tested; `retrieve` lazily imports the embedders + DB.
"""
from __future__ import annotations

from typing import Any

from ..llm.base import ClipMoment, ReelBrief, ShotSpec


def retrieval_filters(exclude_non_logan_person: bool = True) -> dict[str, Any]:
    """Filter dict for the candidate query.

    v1: "exclude non-Logan person" == keep ``is_logan=true`` (objects/no-person are
    dropped too - moot for fitness). FUTURE: when a ``subject`` enum lands, return
    ``{"subject_not_in": ("other", "multiple")}`` here to also keep object/no-person -
    callers are unchanged.
    """
    return {"is_logan": True} if exclude_non_logan_person else {}


def quality_adjusted_distance(distance: float, quality_score, weight: float) -> float:
    """Down-weight low-quality moments by inflating their effective kNN distance (pure).

    A low ``quality_score`` (0-1) adds a penalty so junk moments rank LOWER even when their
    embedding is near. ``weight=0`` or unknown quality (legacy moments) = no penalty.
    """
    if not weight or quality_score is None:
        return distance
    return distance + weight * (1.0 - quality_score)


def merge_candidates(result_lists: list[list[dict]], k: int, *, quality_weight: float = 0.0) -> list[dict]:
    """Merge kNN row lists (image + text), dedup by moment id keeping the nearest, top-K.

    Ranks by quality-adjusted distance so low-quality moments are down-weighted (3.7).
    """
    def adj(row):
        return quality_adjusted_distance(row["distance"], row.get("quality_score"), quality_weight)

    best: dict[Any, dict] = {}
    for rows in result_lists:
        for row in rows:
            mid = row["id"]
            if mid not in best or adj(row) < adj(best[mid]):
                best[mid] = row
    return sorted(best.values(), key=adj)[:k]


def hyde_text(shot: ShotSpec, brief: ReelBrief | None = None) -> str:
    """A hypothetical ideal-clip caption for this beat (HyDE-style intent query, pure).

    Written in the caption lane's own register - a one-paragraph library caption of the
    clip we WISH we had ("Logan performs ... explosive ... at night ...") - so its text
    embedding lands near the stored moment captions. Composes the shot's creative-brief
    fields (role / intended feeling / intended energy / description / query) plus the
    reel brief's vibe + emotional register when available.
    """
    action = (shot.description or shot.query or "on camera").strip().rstrip(".")
    sentences = [f"Logan {action}."]
    energy = shot.intended_energy or shot.energy
    if energy:
        sentences.append(f"The energy is {energy}.")
    if shot.intended_feeling:
        sentences.append(f"It feels {shot.intended_feeling}.")
    if shot.role:
        sentences.append(f"This is the {shot.role} beat of the reel.")
    query = (shot.query or "").strip().rstrip(".")
    if query and query != action:
        sentences.append(f"Shows {query}.")
    if brief is not None:
        sentences.append(f"Overall vibe: {brief.vibe}; "
                         f"emotional register: {brief.emotional_register}.")
    return " ".join(sentences)


def rrf_fuse(result_lists: list[list[dict]], k: int, rrf_k: int = 60,
             quality_weight: float = 0.0) -> list[dict]:
    """Reciprocal Rank Fusion across retrieval lanes (pure): top-K row dicts.

    Per moment id, ``score = sum over lanes of 1 / (rrf_k + rank)`` (rank 1-based;
    rrf_k=60 is the standard damping so cross-lane consensus beats one lane's top
    rank). No score normalization needed - only ranks are used. Each moment keeps
    its nearest row (best distance) for the output and for tie-breaking equal scores.

    Quality down-weight: the fused score is multiplied by ``1 - weight * (1 - q)``
    (multiplicative because RRF scores have no distance units to add a penalty to -
    unlike ``quality_adjusted_distance``): q=1 is untouched, low q loses up to
    ``weight`` of its score, and unknown quality (None, legacy moments) stays neutral.
    """
    scores: dict[Any, float] = {}
    best: dict[Any, dict] = {}
    for rows in result_lists:
        for rank, row in enumerate(rows, start=1):
            mid = row["id"]
            scores[mid] = scores.get(mid, 0.0) + 1.0 / (rrf_k + rank)
            if mid not in best or row["distance"] < best[mid]["distance"]:
                best[mid] = row

    def adjusted(mid) -> float:
        quality = best[mid].get("quality_score")
        if not quality_weight or quality is None:
            return scores[mid]
        return scores[mid] * (1.0 - quality_weight * (1.0 - quality))

    ordered = sorted(best, key=lambda mid: (-adjusted(mid), best[mid]["distance"]))
    return [best[mid] for mid in ordered[:k]]


def row_to_moment(row: dict) -> ClipMoment:
    """Map a `db.client.knn` row to a `ClipMoment` candidate."""
    return ClipMoment(
        clip_key=row["storage_key"],
        start_s=row["start_s"],
        end_s=row["end_s"],
        description=row["description"],
        is_logan=row["is_logan"],
        framing_notes=row["framing_notes"],
        energy=row["energy"],
        location=row.get("location"),
        moment_id=row["id"],
        clip_id=row["clip_id"],
        thumbnail_key=row["thumbnail_key"],
        distance=row["distance"],
        quality_score=row.get("quality_score"),
    )


# --- Phase 3.5 cheap shortlist: pure prompt assembly + top-N selection ------

def brief_summary(brief: ReelBrief) -> str:
    """One-paragraph text of the reel's creative read (for the cheap shortlist prompt)."""
    text = (f"VIBE: {brief.vibe}. EMOTIONAL REGISTER: {brief.emotional_register}. "
            f"ENERGY PROFILE: {brief.energy_profile}. TONE: {brief.tone}. THEME: {brief.theme}.")
    if brief.trend_context:
        text += f" TREND CONTEXT: {brief.trend_context}"
    return text


def shot_ask(shot: ShotSpec) -> str:
    """What this beat needs to do, in the reel's arc (for shortlist + rerank)."""
    return (f"ROLE: {shot.role or '?'}. INTENDED FEELING: {shot.intended_feeling or '?'}. "
            f"INTENDED ENERGY: {shot.intended_energy or shot.energy}. SHOWS: {shot.description}.")


def format_candidate_meta(c: ClipMoment) -> str:
    """A candidate's stored metadata as one text line (no image - cheap shortlist input)."""
    q = "?" if c.quality_score is None else f"{c.quality_score:.2f}"
    return (f"energy={c.energy}; location={c.location or '?'}; quality={q}; "
            f"framing={c.framing_notes}; desc={c.description}")


def select_top_n(candidates: list[ClipMoment], scores: list[float], n: int) -> list[ClipMoment]:
    """Top-N candidates by score (desc, stable); missing scores default to 0."""
    ranked = sorted(
        enumerate(candidates),
        key=lambda it: (-(scores[it[0]] if it[0] < len(scores) else 0.0), it[0]),
    )
    return [c for _, c in ranked[:n]]


# --- Phase 3.6 cross-reel freshness (explore/exploit) -----------------------

def freshness_penalty(count: int, recency: float = 0.0) -> float:
    """Penalty to SUBTRACT from a candidate's shortlist score.

    Never-used clips get a small boost (negative penalty); used clips get a positive penalty
    that grows with how OFTEN (``count``) and how RECENTLY (``recency`` in [0,1], 1=latest reel)
    they've been used. Scaled by ``FRESHNESS_WEIGHT`` where applied.
    """
    if count <= 0:
        return -1.0
    return float(count) + recency


def apply_freshness(fit_scores: list[float], candidates: list[ClipMoment],
                    usage: dict | None, weight: float) -> list[float]:
    """Combine taste-fit scores with the cross-reel freshness term (weight 0 = no-op)."""
    if not weight:
        return list(fit_scores)
    adjusted = []
    for i, c in enumerate(candidates):
        fit = fit_scores[i] if i < len(fit_scores) else 0.0
        u = (usage or {}).get(c.clip_id)
        pen = freshness_penalty(u["count"], u.get("recency", 0.0)) if u else freshness_penalty(0)
        adjusted.append(fit - weight * pen)
    return adjusted


def shortlist(vision, brief: ReelBrief, shot: ShotSpec, candidates: list[ClipMoment],
              n: int = 6, *, usage: dict | None = None, freshness_weight: float = 0.0) -> list[ClipMoment]:
    """Cheap text-only (Haiku) narrowing of a wide pool to ~n finalists, by fit to the reel
    brief + shot role/feeling, with a cross-reel freshness term. Keeps the expensive vision
    rerank off the wide pool."""
    if len(candidates) <= n:
        return candidates
    lines = [format_candidate_meta(c) for c in candidates]
    fit = vision.shortlist_score(brief_summary(brief), shot_ask(shot), lines)
    scores = apply_freshness(fit, candidates, usage, freshness_weight)
    return select_top_n(candidates, scores, n)


def retrieve(
    shot: ShotSpec,
    ref_frame: str | None = None,
    *,
    k: int = 20,
    use_text: bool = True,
    exclude_non_logan_person: bool = True,
    retriever: str | None = None,
    brief: ReelBrief | None = None,
    conn=None,
) -> list[ClipMoment]:
    """Top-K candidate moments for a reference shot (wide pool for the 3.5 shortlist).

    ``ref_frame`` is a representative frame path (image signal). ``retriever`` picks the
    lane set ("clip" | "hybrid"; default ``settings.retriever`` - see module docstring);
    ``use_text`` only applies to the "clip" path, and ``brief`` only enriches the
    "hybrid" HyDE query. Energy is NOT hard-filtered - the taste layer decides fit
    downstream.
    """
    from ..config import settings

    mode = retriever or settings.retriever
    filters = retrieval_filters(exclude_non_logan_person)
    # Pull a few extra per lane so the fusion has room to dedup before the top-K cut.
    pool = max(k * 2, k)

    if mode == "clip":
        rows = _clip_lanes(shot, ref_frame, pool, use_text, filters, conn)
        merged = merge_candidates(rows, k, quality_weight=settings.quality_weight)
    elif mode == "hybrid":
        rows = _hybrid_lanes(shot, ref_frame, pool, brief, filters, conn)
        merged = rrf_fuse(rows, k, quality_weight=settings.quality_weight)
    else:
        raise ValueError(f"unknown retriever {mode!r}; expected 'clip' or 'hybrid'")
    return [row_to_moment(r) for r in merged]


def _clip_lanes(shot: ShotSpec, ref_frame: str | None, pool: int, use_text: bool,
                filters: dict[str, Any], conn) -> list[list[dict]]:
    """The original CLIP-only lanes: image kNN + optional CLIP-text query (9B A/B baseline)."""
    from ..db.client import knn
    from ..ingest.embed import clip_embed, clip_text_embed

    if ref_frame is None and not (use_text and shot.query):
        raise ValueError("retrieve needs a reference frame and/or a text query")
    result_lists: list[list[dict]] = []
    if ref_frame is not None:
        result_lists.append(knn(clip_embed(ref_frame), "clip", k=pool, filters=filters, conn=conn))
    if use_text and shot.query:
        result_lists.append(knn(clip_text_embed(shot.query), "clip", k=pool, filters=filters, conn=conn))
    return result_lists


def _hybrid_lanes(shot: ShotSpec, ref_frame: str | None, pool: int,
                  brief: ReelBrief | None, filters: dict[str, Any], conn) -> list[list[dict]]:
    """Phase 9B lanes: CLIP image kNN + caption-text kNN (HyDE intent) + caption BM25."""
    from ..db.client import bm25, caption_knn, knn
    from ..ingest.embed import clip_embed, openai_text_embed

    result_lists: list[list[dict]] = []
    if ref_frame is not None:
        result_lists.append(knn(clip_embed(ref_frame), "clip", k=pool, filters=filters, conn=conn))
    qvec = openai_text_embed([hyde_text(shot, brief)])[0]
    result_lists.append(caption_knn(qvec, k=pool, filters=filters, conn=conn))
    query_text = shot.query or shot.description
    if query_text:
        result_lists.append(bm25(query_text, k=pool, filters=filters, conn=conn))
    return result_lists
