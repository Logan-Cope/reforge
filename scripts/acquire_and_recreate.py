#!/usr/bin/env python3
"""Batch front door: Instagram reel URLs -> recreated reels.

For each URL: dedup against Reforge's OWN `source_reels` ledger (not whatever upstream
discovery tool supplied the URLs), download the reel, deconstruct it, and apply the caption-heavy filter -
Reforge can only recreate low/no-caption B-roll (it swaps in Logan's OWN footage), so
caption-heavy references are skipped. Survivors run the SAME pipeline as
`scripts/recreate.py` (build_edl -> gate -> render + burn overlay -> persist).

Each URL is wrapped in try/except so one bad reel never kills the batch.

Usage:
    PYTHONPATH=src python scripts/acquire_and_recreate.py --urls U1 U2 ...
    PYTHONPATH=src python scripts/acquire_and_recreate.py --urls-file reels.txt --limit 5
    PYTHONPATH=src python scripts/acquire_and_recreate.py --urls U1 --deconstruct-only  # cheap dry run
"""
from __future__ import annotations

import argparse
import dataclasses
import os

from reforge.acquire import dedup
from reforge.acquire.download import download_reel, shortcode_from_url
from reforge.deconstruct.reference import deconstruct
from reforge.llm import get_captioner
from reforge.match.matcher import build_edl
from reforge.render.engine import render_reel
from reforge.render.overlay import burn_text

FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"


def _read_urls_file(path: str) -> list[str]:
    """One URL per line; '#' comments and blank lines ignored."""
    urls = []
    with open(path) as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if line:
                urls.append(line)
    return urls


def _persist(reference: str, decon, matched, out_path: str) -> int:
    """Write the reels/slots run record (same logic as recreate.py's `_persist`)."""
    from reforge.app.queries import alternates_json, update_slot_candidates
    from reforge.db import client as db

    total = sum(m.render.duration for m in matched)
    reel_id = db.insert_reel(
        reference_key=reference,
        total_duration=total,
        cut_grid=[[s.start_s, s.end_s] for s in decon.shots],
        brief=dataclasses.asdict(decon.brief),
        output_key=out_path,
    )
    for m in matched:
        slot_id = db.insert_slot(
            reel_id, m.shot.idx,
            start_s=m.shot.start_s, end_s=m.shot.end_s,
            ref_description=m.shot.description, ref_text_overlay=m.shot.on_screen_text,
            ref_energy=m.shot.energy, chosen_moment_id=m.plan.moment.moment_id,
            params={"in_point": m.render.in_point, "zoom": m.render.zoom,
                    "xfrac": m.render.xfrac, "yfrac": m.render.yfrac,
                    "brightness": m.render.brightness},
            reason=m.plan.reason,
        )
        alts = alternates_json(getattr(m, "alternates", []) or [])
        if alts:
            update_slot_candidates(slot_id, alts)
    return reel_id


def _recreate(reference: str, decon, vision, work_dir: str, shortcode: str) -> tuple[str, int]:
    """Run recreate.py's building blocks on a deconstructed reel; return (out_path, reel_id)."""
    from reforge.match.gate import run_gate

    matched = build_edl(decon, vision)
    matched, _events = run_gate(decon, matched, vision)

    stem = os.path.splitext(os.path.basename(reference))[0]
    name = f"{shortcode}_recreation.mp4" if stem == shortcode else f"{shortcode}_{stem}.mp4"
    out_path = os.path.join("output", name)
    os.makedirs("output", exist_ok=True)

    # render_reel's concat list needs an ABSOLUTE tmp_dir; burn the overlay separately
    # (this Homebrew ffmpeg lacks the `drawtext` filter - no libfreetype).
    silent = os.path.splitext(out_path)[0] + "_notext.mp4"
    render_reel([m.render for m in matched], decon.audio_path, silent, text="",
                tmp_dir=os.path.abspath(os.path.join(work_dir, "segs")), font=FONT)
    out = burn_text(silent, decon.overlay_text, out_path, font=FONT)
    os.remove(silent)
    reel_id = _persist(reference, decon, matched, out)
    return out, reel_id


def process_url(url: str, vision, work_root: str, *, deconstruct_only: bool) -> str:
    """Handle one URL end to end; return a one-line status verdict. Never raises for a bad reel."""
    shortcode = shortcode_from_url(url)
    if not shortcode:
        return f"SKIP  {url} (not an Instagram reel/post URL)"

    if dedup.already_seen(shortcode):
        dedup.mark_skipped(shortcode, "already in source_reels", "skipped_dup")
        return f"SKIP  {shortcode} (already seen -> skipped_dup)"

    dedup.record_pulled(shortcode, url)
    try:
        path = download_reel(url)
        work_dir = os.path.join(work_root, shortcode)
        decon = deconstruct(path, vision, work_dir)
        density = dedup.caption_density(decon.shots)
        variety = dedup.caption_variety(decon.shots, decon.overlay_text)
        overlay_wc = len((decon.overlay_text or "").split())
        heavy = dedup.is_caption_heavy(variety, overlay_wc)
        head = (f"{shortcode}: downloaded {os.path.basename(path)} | "
                f"{len(decon.shots)} shots | caption-variety={variety:.2f} density={density:.2f} "
                f"overlay={decon.overlay_text!r} ({overlay_wc}w)")

        if heavy:
            reason = f"caption-heavy (variety={variety:.2f}, overlay_words={overlay_wc})"
            dedup.mark_skipped(shortcode, reason, "skipped_caption")
            return f"SKIP  {head} -> skipped_caption: {reason}"

        if deconstruct_only:
            return f"OK    {head} -> RECREATABLE (deconstruct-only, no recreate)"

        out, reel_id = _recreate(path, decon, vision, work_dir, shortcode)
        dedup.mark_recreated(shortcode, reel_id, density)
        return f"DONE  {head} -> recreated {out} (reels.id={reel_id})"
    except Exception as e:  # one bad reel never kills the batch
        dedup.mark_skipped(shortcode, f"{type(e).__name__}: {e}", "error")
        return f"ERROR {shortcode}: {type(e).__name__}: {e}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch: Instagram reel URLs -> recreated reels.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--urls", nargs="+", help="one or more reel URLs")
    src.add_argument("--urls-file", help="file with one URL per line ('#' comments ok)")
    ap.add_argument("--limit", type=int, default=None, help="only process the first N URLs")
    ap.add_argument("--deconstruct-only", action="store_true",
                    help="cheap caption-filter dry run (download + deconstruct, no recreate)")
    ap.add_argument("--work", default="data/acquire_work", help="scratch dir for frames/audio/segs")
    args = ap.parse_args()

    urls = args.urls if args.urls else _read_urls_file(args.urls_file)
    if args.limit is not None:
        urls = urls[: args.limit]

    vision = get_captioner()
    recreated = passed = skipped = 0
    print(f"[acquire] {len(urls)} URL(s){' (deconstruct-only)' if args.deconstruct_only else ''}")
    for url in urls:
        line = process_url(url, vision, args.work, deconstruct_only=args.deconstruct_only)
        print(f"  {line}")
        if line.startswith("DONE"):
            recreated += 1
        elif line.startswith("OK"):
            passed += 1
        else:  # SKIP / ERROR
            skipped += 1

    cost = vision.cost_summary()
    verdict = (f"{passed} recreatable (deconstruct-only)" if args.deconstruct_only
               else f"{recreated} recreated")
    print(f"[summary] {verdict}, {skipped} skipped/errored")
    print(f"COST: ${cost['total_usd']:.3f} over {cost['total_calls']} calls")


if __name__ == "__main__":
    main()
