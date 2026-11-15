#!/usr/bin/env python3
"""Re-render a persisted reel with the reviewer's swaps applied.

The review app records a swap as ``slots.swapped_moment_id`` (+ human_verdict='swapped'),
but the original mp4 keeps playing the judge's picks. This script rebuilds the EDL from
the DB with swaps applied (the cut grid is preserved: each slot keeps its start/end
duration), renders a NEW versioned mp4 (``output/reel<id>_v<k>.mp4`` - the first free
suffix, never overwriting anything), and points ``reels.output_key`` at it. The previous
file stays on disk untouched (house rule: renders and rows are append-only).

Usage:
    PYTHONPATH=src python scripts/rerender_reel.py --reel-id 8
    PYTHONPATH=src python scripts/rerender_reel.py --reel-id 8 --dry-run   # print the EDL only
    PYTHONPATH=src python scripts/rerender_reel.py --reel-id 8 --out output/custom.mp4
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from typing import Callable

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))  # allow running without PYTHONPATH=src

# Single source of truth with the review app's beat cards: the card previews and this
# re-render must cut the SAME seconds for a swapped slot.
from reforge.app.queries import effective_in_point  # noqa: E402

FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"


# --- pure helpers (unit-tested; no I/O) ---------------------------------------

def dominant_text(texts: list[str | None]) -> str:
    """Most common non-empty stripped string (``dominant_overlay`` on raw strings).

    ``slots.ref_text_overlay`` is the per-shot reference OCR; its dominant value is the
    best available proxy for the persistent overlay the original render burned.
    """
    vals = [t.strip() for t in texts if t and t.strip()]
    return Counter(vals).most_common(1)[0][0] if vals else ""


def versioned_output(reel_id: int, exists: Callable[[str], bool] = os.path.exists) -> str:
    """First free ``output/reel<id>_v<k>.mp4`` with k >= 2 (never overwrite a render)."""
    k = 2
    while exists(f"output/reel{reel_id}_v{k}.mp4"):
        k += 1
    return f"output/reel{reel_id}_v{k}.mp4"


# --- DB read (single join: slots -> effective moment -> clip) -------------------

_SLOT_SQL = """
    SELECT s.idx, s.start_s, s.end_s, s.params, s.candidates,
           s.chosen_moment_id, s.swapped_moment_id, s.ref_text_overlay,
           m.start_s, m.end_s, cl.storage_key
    FROM slots s
    JOIN moments m ON m.id = COALESCE(s.swapped_moment_id, s.chosen_moment_id)
    JOIN clips cl ON cl.id = m.clip_id
    WHERE s.reel_id = %s
    ORDER BY s.idx
"""


def _load(reel_id: int) -> tuple[str, str | None, list[tuple]]:
    """Read the reel row + its slots (with effective moment/clip joined in)."""
    from reforge.db.client import get_conn

    conn = get_conn()
    try:
        reel = conn.execute(
            "SELECT reference_key, output_key FROM reels WHERE id = %s", (reel_id,)).fetchone()
        if reel is None:
            raise SystemExit(f"reel {reel_id} not found")
        slots = conn.execute(_SLOT_SQL, (reel_id,)).fetchall()
        conn.commit()
    finally:
        conn.close()
    if not slots:
        raise SystemExit(f"reel {reel_id} has no slots with a resolvable moment - nothing to render")
    return reel[0], reel[1], slots


def main() -> None:
    ap = argparse.ArgumentParser(description="Re-render a reel applying the reviewer's swaps.")
    ap.add_argument("--reel-id", type=int, required=True)
    ap.add_argument("--out", default=None, help="output path override (must not exist)")
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve + print the EDL and the would-be output; no render, no DB write")
    args = ap.parse_args()

    from reforge.render.engine import SlotRender
    from reforge.storage.base import get_storage

    reference_key, old_output, slot_rows = _load(args.reel_id)

    out = args.out or versioned_output(
        args.reel_id, exists=lambda p: os.path.exists(os.path.join(REPO, p)))
    out_abs = out if os.path.isabs(out) else os.path.join(REPO, out)
    if os.path.exists(out_abs):
        raise SystemExit(f"refusing to overwrite existing render: {out}")

    # the recreation's audio comes from the reference reel - it must be reachable locally
    ref_path = reference_key if os.path.isabs(reference_key) else os.path.join(REPO, reference_key)
    if not os.path.exists(ref_path):
        raise SystemExit(
            f"reference video not found: {ref_path} (reels.reference_key={reference_key!r}) - "
            "its audio track is muxed into the re-render; restore that file first")

    storage = get_storage()
    edl: list[SlotRender] = []
    overlay_texts: list[str | None] = []
    n_swapped = 0
    print(f"reel {args.reel_id}: {len(slot_rows)} slots  ref={reference_key}")
    print(f"  current output_key: {old_output or '(none)'}")
    for (idx, start_s, end_s, params, candidates, chosen_id, swapped_id,
         ref_text, m_start, m_end, storage_key) in slot_rows:
        params = params or {}
        zoom = float(params.get("zoom", 1.0))
        assert zoom == 1.0, f"slot {idx}: zoom={zoom} violates the locked recipe (must be 1.0)"
        in_point = effective_in_point(params, candidates, swapped_id, m_start, m_end)
        duration = float(end_s) - float(start_s)   # cut grid preserved
        src = storage.ensure_local(storage_key)
        edl.append(SlotRender(
            src=src, in_point=in_point, duration=duration, zoom=1.0,
            xfrac=float(params.get("xfrac", 0.5)), yfrac=float(params.get("yfrac", 0.5)),
            brightness=params.get("brightness") or "none"))
        overlay_texts.append(ref_text)
        swapped = swapped_id is not None
        n_swapped += swapped
        tag = (f"SWAPPED moment {chosen_id} -> {swapped_id}" if swapped
               else f"kept moment {chosen_id}")
        print(f"  slot {idx:02d}: {os.path.basename(src):32} in={in_point:6.2f} "
              f"dur={duration:5.2f}  {tag}")

    overlay = dominant_text(overlay_texts)
    print(f"  swaps applied: {n_swapped}/{len(edl)} | overlay={overlay!r} | out -> {out}")

    if args.dry_run:
        print("DRY RUN: no render, no DB write")
        return

    from reforge.deconstruct.reference import extract_audio
    from reforge.render.engine import render_reel
    from reforge.render.overlay import burn_text

    work = os.path.join(REPO, "data", "rerender_work")
    audio_path = extract_audio(ref_path, os.path.join(work, f"reel{args.reel_id}_audio.m4a"))

    # render_reel's concat list needs an ABSOLUTE tmp_dir; burn the overlay separately
    # (this Homebrew ffmpeg lacks the `drawtext` filter - no libfreetype).
    silent = os.path.splitext(out_abs)[0] + "_notext.mp4"
    render_reel(edl, audio_path, silent, text="",
                tmp_dir=os.path.abspath(os.path.join(work, "segs")), font=FONT)
    burn_text(silent, overlay, out_abs, font=FONT)
    os.remove(silent)  # only the temp; every prior render stays on disk
    print(f"WROTE {out} ({os.path.getsize(out_abs)} bytes)")

    from reforge.db.client import get_conn

    conn = get_conn()
    try:
        conn.execute("UPDATE reels SET output_key = %s WHERE id = %s", (out, args.reel_id))
        conn.commit()
    finally:
        conn.close()
    print(f"reels.id={args.reel_id} output_key: {old_output or '(none)'} -> {out} "
          f"(previous file kept on disk)")


if __name__ == "__main__":
    main()
