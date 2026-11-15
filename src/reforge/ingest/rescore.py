"""TIER-1 re-score (Phase 9A): recompute quality from the cached Tier-0 raw signals.

`rescore_library` reads `cache_cv_signals` + `cache_vlm_assessments` (never the decoder, the
embedder, or the VLM) and recomputes `score_window` for every active moment - so changing a
scorer parameter re-scores the whole library in MINUTES. Writes versioned `window_scores`;
optionally promotes onto the live moments + REFRESHes the retrieval view.

`promote_scorer` activates an already-computed score version (one UPDATE + REFRESH, seconds) -
used to A/B between scorers or roll back to the baseline without recomputing.
"""
from __future__ import annotations

from . import cache
from .quality import score_window


def rescore_library(scorer_version: str, *, combine_w: float = 0.5,
                    promote: bool = True, conn=None) -> dict:
    """Recompute every active moment's score from cached signals. Tier-1 only (no decode/embed).

    Bulk read (2 queries) -> pure score in memory -> bulk upsert window_scores (one executemany)
    -> promote onto moments (one UPDATE) -> REFRESH the view. That's a handful of round-trips for
    the whole library, vs the 2-3h re-ingest.
    """
    from ..db import client as db

    own = conn is None
    conn = conn or db.get_conn()
    ev, vv = cache.EXTRACTOR_VERSION, cache.vlm_quality_version()
    try:
        cv = db.read_all_cv_signals(ev, conn=conn)
        vlm = db.read_all_vlm(vv, conn=conn)
        moments = db.active_moments_for_rescore(conn=conn)

        rows = []
        missing = 0
        for m in moments:
            key = (m["content_hash"], m["window_idx"])
            sig, v = cv.get(key), vlm.get(key)
            if sig is None or v is None:
                missing += 1
                continue
            r = score_window(
                exposure=sig["exposure"], sharpness=sig["sharpness"], stability=sig["stability"],
                subject_present=sig["subject_present"], vlm_usable=v["usable"],
                vlm_flags=v["flags"], combine_w=combine_w,
            )
            rows.append((m["content_hash"], m["window_idx"], r["cv_quality"], r["vlm_quality"],
                         r["quality_score"], r["quality_flags"]))

        db.bulk_put_window_scores(scorer_version, rows, conn=conn)
        conn.commit()
        promoted = 0
        if promote:
            promoted = db.promote_window_scores(scorer_version, conn=conn)
            conn.commit()
            db.refresh_retrieval_view(conn=conn)
        return {"scorer_version": scorer_version, "scored": len(rows), "missing": missing,
                "promoted": promoted, "moments": len(moments)}
    finally:
        if own and not conn.closed:
            conn.close()


def promote_scorer(scorer_version: str, conn=None) -> dict:
    """Activate a previously-computed score version onto moments + REFRESH (no recompute)."""
    from ..db import client as db

    own = conn is None
    conn = conn or db.get_conn()
    try:
        n = db.promote_window_scores(scorer_version, conn=conn)
        conn.commit()
        db.refresh_retrieval_view(conn=conn)
        return {"scorer_version": scorer_version, "promoted_rows": n}
    finally:
        if own and not conn.closed:
            conn.close()
