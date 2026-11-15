#!/usr/bin/env python3
"""Re-pick ONE flagged beat, using the reviewer's flag as the rerank critique.

The review app's "Re-pick this beat" button queues a ``repick:<slot_id>`` run request;
the worker runs this script. It rebuilds the beat's ShotSpec + ReelBrief from what the
run persisted (slots.ref_* columns + reels.brief - the full deconstruction is not
stored, so the retrieval query degrades to the shot description), re-extracts reference
frames from the reference video, and re-runs ``matcher.match_slot`` with the flagged
moment/clip hard-excluded and the flag text as critique. The winning plan lands as a
swap (``swapped_moment_id`` + refreshed ``candidates``), the verdict clears so the
reviewer can judge the fresh pick (the note stays - it is why), and the reel goes
cut-dirty (the rendered mp4 no longer matches the picks).

Usage:
    PYTHONPATH=src python scripts/repick_slot.py --slot-id 71
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))  # allow running without PYTHONPATH=src

from reforge.llm.base import ReelBrief, ShotSpec  # noqa: E402

_BRIEF_FIELDS = ("vibe", "emotional_register", "energy_profile", "tone", "theme",
                 "trend_context")


# --- pure helpers (unit-tested; no I/O) ---------------------------------------

def shot_from_slot(idx: int, start_s: float, end_s: float, ref_description: str | None,
                   ref_text_overlay: str | None, ref_energy: str | None) -> ShotSpec:
    """Rebuild the reference ShotSpec from the persisted slot columns.

    The original retrieval query is not stored, so the description doubles as the
    query (that is most of what the query was built from anyway).
    """
    desc = ref_description or ""
    return ShotSpec(idx=idx, start_s=float(start_s), end_s=float(end_s),
                    description=desc, on_screen_text=ref_text_overlay or "",
                    energy=ref_energy or "", query=desc)


def brief_from_json(brief: dict | None) -> ReelBrief:
    """``reels.brief`` jsonb -> ReelBrief; missing rows/keys degrade to empty strings."""
    brief = brief or {}
    return ReelBrief(**{f: brief.get(f, "") for f in _BRIEF_FIELDS})


def critique_from_flag(note: str | None) -> str:
    """The rerank critique carrying the human's flag (reason [+ typed note])."""
    return (f"Human flagged this pick: {note or 'no reason given'}. "
            "Pick something that addresses this.")


def reel_state_from_rows(rows: list[tuple]) -> dict:
    """Variety state from the reel's OTHER slots (clip_id, moment_id, location,
    description per effective moment) - same fields the gate rebuilds, so the re-pick
    cannot duplicate a sibling's clip or blow the location cap."""
    used_clip_ids: set = set()
    used_moment_ids: set = set()
    location_counts: dict = {}
    clip_use_count: dict = {}
    used_looks: list[str] = []
    for clip_id, moment_id, location, description in rows:
        loc = location or "?"
        used_clip_ids.add(clip_id)
        used_moment_ids.add(moment_id)
        location_counts[loc] = location_counts.get(loc, 0) + 1
        clip_use_count[clip_id] = clip_use_count.get(clip_id, 0) + 1
        used_looks.append(f"{loc}: {(description or '').strip()[:45]}")
    return {"used_clip_ids": used_clip_ids, "used_moment_ids": used_moment_ids,
            "location_counts": location_counts, "clip_use_count": clip_use_count,
            "used_looks": used_looks}


def candidates_for_slot(matched) -> list[dict]:
    """The refreshed ``slots.candidates`` jsonb: the NEW pick first, then runner-ups.

    The new pick's entry carries the render's (clamped) in_point - ``effective_in_point``
    resolves a swapped slot from the candidates entry matching ``swapped_moment_id``,
    so the entry must exist and must hold the seconds the re-render will cut.
    """
    from reforge.app.queries import alternates_json

    cands = alternates_json([matched.plan] + list(matched.alternates))
    cands[0]["in_point"] = matched.render.in_point
    return cands


# --- DB + pipeline (I/O) -------------------------------------------------------

_SLOT_SQL = """
    SELECT s.reel_id, s.idx, s.start_s, s.end_s, s.ref_description, s.ref_text_overlay,
           s.ref_energy, s.human_note, m.id, m.clip_id, r.reference_key, r.brief
    FROM slots s
    JOIN moments m ON m.id = COALESCE(s.swapped_moment_id, s.chosen_moment_id)
    JOIN reels r ON r.id = s.reel_id
    WHERE s.id = %s
"""

_SIBLINGS_SQL = """
    SELECT m.clip_id, m.id, m.location, m.description
    FROM slots s
    JOIN moments m ON m.id = COALESCE(s.swapped_moment_id, s.chosen_moment_id)
    WHERE s.reel_id = %s AND s.id <> %s
"""


def main() -> None:
    ap = argparse.ArgumentParser(description="Re-pick one flagged beat (flag = critique).")
    ap.add_argument("--slot-id", type=int, required=True)
    args = ap.parse_args()

    from reforge.config import settings
    from reforge.db import client as db
    from reforge.db.client import get_conn
    from reforge.ingest.frames import extract_frame, sample_timestamps
    from reforge.llm import get_captioner
    from reforge.match.matcher import match_slot
    from reforge.storage.base import get_storage
    from reforge.taste import record_taste_event

    conn = get_conn()
    try:
        row = conn.execute(_SLOT_SQL, (args.slot_id,)).fetchone()
        if row is None:
            raise SystemExit(f"slot {args.slot_id} not found (or has no resolvable moment)")
        (reel_id, idx, start_s, end_s, ref_description, ref_text_overlay, ref_energy,
         human_note, old_moment_id, old_clip_id, reference_key, brief_json) = row
        siblings = conn.execute(_SIBLINGS_SQL, (reel_id, args.slot_id)).fetchall()
        conn.commit()
    finally:
        conn.close()

    ref_path = reference_key if os.path.isabs(reference_key or "") \
        else os.path.join(REPO, reference_key or "")
    if not reference_key or not os.path.exists(ref_path):
        raise SystemExit(
            f"reference video not found: {ref_path} (reels.reference_key={reference_key!r}) - "
            "the re-pick needs its frames for retrieval; restore that file first")

    shot = shot_from_slot(idx, start_s, end_s, ref_description, ref_text_overlay, ref_energy)
    brief = brief_from_json(brief_json)
    critique = critique_from_flag(human_note)
    state = reel_state_from_rows(siblings)
    print(f"repick slot {args.slot_id} (reel {reel_id} beat {idx}): "
          f"excluding moment {old_moment_id} / clip {old_clip_id}")
    print(f"  critique: {critique}")

    try:
        usage = db.clip_usage()
    except Exception:
        usage = {}

    tmp = tempfile.mkdtemp(prefix=f"reforge_repick_{args.slot_id}_")
    try:
        dur = max(float(end_s) - float(start_s), 0.1)
        frames = [
            extract_frame(ref_path, float(start_s) + off, os.path.join(tmp, f"ref_{j}.jpg"))
            for j, off in enumerate(sample_timestamps(dur, n=3))
        ]
        matched = match_slot(
            shot, frames, get_captioner(), brief,
            k=20, shortlist_n=6, cand_frames=5, storage=get_storage(), usage=usage,
            fresh_w=settings.freshness_weight, cap=settings.diversity_location_cap,
            exclude_moment_ids={old_moment_id}, hard_exclude_clip_ids={old_clip_id},
            critique=critique, **state)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    new_moment_id = matched.plan.moment.moment_id
    cands = candidates_for_slot(matched)
    print(f"  new pick: moment {new_moment_id} ({os.path.basename(matched.render.src)}) "
          f"in={matched.render.in_point:.2f} | {matched.plan.reason}")

    from psycopg.types.json import Json

    conn = get_conn()
    try:
        # the new pick lands as a swap; verdict clears so the card is open to judge
        # again; the note stays (it is WHY this repick happened); the cut is now stale.
        conn.execute("""
            UPDATE slots SET swapped_moment_id = %s, candidates = %s, human_verdict = NULL
            WHERE id = %s
        """, (new_moment_id, Json(cands), args.slot_id))
        conn.execute("UPDATE reels SET cut_dirty = true WHERE id = %s", (reel_id,))
        conn.commit()
    finally:
        conn.close()

    record_taste_event({"event": "repick", "run": reel_id, "beat": idx,
                        "critique": critique, "old": old_moment_id, "new": new_moment_id})
    print(f"repick slot {args.slot_id}: done (reel {reel_id} is cut-dirty)")


if __name__ == "__main__":
    main()
