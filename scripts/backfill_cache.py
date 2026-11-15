"""One-time backfill of the Phase 9A tiered cache from existing data (no video decode).

Existing moments already carry their final scores + a content-addressed thumbnail on disk
(`data/thumbs/<hash12>/<window>.jpg`). This populates:
  - moments.window_idx   (parsed from the thumbnail key)
  - Tier-0 cache_windows / cache_frames / cache_cv_signals (raw CV recomputed from the THUMBNAIL,
    stability recovered from the stored cv_quality) / cache_vlm_assessments (usable+flags) /
    cache_annotations / cache_embeddings (copied from moment_clip_embeddings)
  - Tier-1 window_scores at the current SCORER_VERSION = EXACTLY the stored moment scores
    (the faithful baseline, so MV retrieval == today's retrieval).

After this, a re-score reads cache_cv_signals + cache_vlm_assessments and recomputes in minutes.
Idempotent (append-only cache; ON CONFLICT DO NOTHING).
"""
from __future__ import annotations

import os

from reforge.db.client import get_conn
from reforge.ingest import cache
from reforge.ingest.quality import frame_signals, subject_present

# cv_quality = 0.35*stability + 0.25*sharpness + 0.20*exposure + 0.20*subject_present (quality.py).
_W_STAB, _W_SHARP, _W_EXP, _W_SUBJ = 0.35, 0.25, 0.20, 0.20


def _recover_stability(cv_quality: float, sharpness: float, exposure: float, subject: float) -> float:
    """Invert the cv_quality formula to get the original stability (exact for unclamped scores)."""
    stab = (cv_quality - _W_SHARP * sharpness - _W_EXP * exposure - _W_SUBJ * subject) / _W_STAB
    return max(0.0, min(1.0, stab))


def main() -> None:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT m.id, c.content_hash, m.thumbnail_key, m.start_s, m.end_s,
               m.cv_quality, m.vlm_quality, m.quality_score, m.quality_flags,
               m.description, m.energy, m.location, m.is_logan, m.framing_notes
        FROM moments m JOIN clips c ON c.id = m.clip_id
        WHERE m.superseded = false AND c.content_hash IS NOT NULL
        ORDER BY m.id
    """)
    rows = cur.fetchall()
    ev, sv = cache.EXTRACTOR_VERSION, cache.SCORER_VERSION
    vv, av = cache.vlm_quality_version(), cache.annotator_version()
    print(f"backfilling {len(rows)} active moments  (extractor={ev} scorer={sv})", flush=True)

    done = skipped_no_thumb = 0
    annotated: set[str] = set()
    for i, r in enumerate(rows):
        (mid, chash, thumb, a, b, cvq, vlmq, qscore, qflags,
         desc, energy, loc, is_logan, framing) = r
        widx = cache.window_idx_from_thumbnail_key(thumb)
        if widx is None:
            continue
        cur.execute("UPDATE moments SET window_idx = %s WHERE id = %s", (widx, mid))

        # Tier-0 geometry + frame (cheap, from the row).
        cur.execute("""INSERT INTO cache_windows (content_hash, extractor_version, window_idx, start_s, end_s)
                       VALUES (%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING""", (chash, ev, widx, a, b))
        if thumb:
            cur.execute("""INSERT INTO cache_frames (content_hash, extractor_version, window_idx, frame_role, storage_key)
                           VALUES (%s,%s,%s,'f0',%s) ON CONFLICT DO NOTHING""", (chash, ev, widx, thumb))

        # Tier-0 raw CV signals - recompute exp/sharp/subject from the saved thumbnail, recover stability.
        tpath = thumb if thumb and os.path.isabs(thumb) else (thumb or "")
        if tpath and os.path.exists(tpath):
            exposure, sharpness = frame_signals(tpath)
            subj = subject_present(tpath)
            stability = _recover_stability(cvq if cvq is not None else 0.0, sharpness, exposure, subj)
            cur.execute("""INSERT INTO cache_cv_signals
                             (content_hash, extractor_version, window_idx, exposure, sharpness, stability, subject_present)
                           VALUES (%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING""",
                        (chash, ev, widx, exposure, sharpness, stability, subj))
        else:
            skipped_no_thumb += 1

        # Tier-0 VLM assessment (usable = stored vlm_quality; flags = stored combined flags).
        cur.execute("""INSERT INTO cache_vlm_assessments (content_hash, vlm_version, window_idx, usable, flags)
                       VALUES (%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING""",
                    (chash, vv, widx, vlmq, _as_json(qflags)))

        # Tier-1 baseline: window_scores @ current scorer = EXACTLY the stored scores (faithful).
        cur.execute("""INSERT INTO window_scores
                         (content_hash, scorer_version, window_idx, cv_quality, vlm_quality, quality_score, quality_flags)
                       VALUES (%s,%s,%s,%s,%s,%s,%s)
                       ON CONFLICT (content_hash, scorer_version, window_idx) DO NOTHING""",
                    (chash, sv, widx, cvq, vlmq, qscore, _as_json(qflags)))

        # Tier-0 annotation: once per clip (inherited by every window).
        if chash not in annotated:
            cur.execute("""INSERT INTO cache_annotations
                             (content_hash, annotator_version, description, energy, location, is_logan, framing_notes)
                           VALUES (%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING""",
                        (chash, av, desc, energy, loc, is_logan, framing))
            annotated.add(chash)

        done += 1
        if (i + 1) % 400 == 0:
            conn.commit()
            print(f"  {i+1}/{len(rows)}", flush=True)

    conn.commit()

    # Tier-0 embeddings: copy current per-moment vectors into the content-addressed cache (bulk).
    embv = cache.embed_version()
    cur.execute("""
        INSERT INTO cache_embeddings (content_hash, embed_version, window_idx, model, embedding)
        SELECT c.content_hash, %s, m.window_idx, e.model, e.embedding
        FROM moments m
        JOIN clips c ON c.id = m.clip_id
        JOIN moment_clip_embeddings e ON e.moment_id = m.id
        WHERE m.superseded = false AND m.window_idx IS NOT NULL AND c.content_hash IS NOT NULL
        ON CONFLICT DO NOTHING
    """, (embv,))
    emb_n = cur.rowcount
    conn.commit()

    print(f"\nDone: {done} moments cached, {skipped_no_thumb} missing-thumb (CV skipped), "
          f"{len(annotated)} clip annotations, {emb_n} embeddings copied (embed_version={embv}).", flush=True)
    conn.close()


def _as_json(value):
    from psycopg.types.json import Json
    return Json(value if value is not None else [])


if __name__ == "__main__":
    main()
