#!/usr/bin/env python3
"""Auto-recreate a reference reel: deconstruct -> match (taste) -> render (NO manual EDL).

Phase 3.5: the reference read produces a reel-level creative brief; per shot we retrieve a
wide pool, cheaply shortlist to finalists (fit to the brief), then taste-rerank on Opus to
pick the clip that serves the shot's role within the reel's vibe. Reports per slot the
chosen clip + in_point + the rerank's one-line reason, persists the run (reels/slots), and
prints the real per-run LLM cost.

Usage:
    python scripts/recreate.py                      # full DISCIPLINE recreation
    python scripts/recreate.py --max-shots 2        # quick pipeline check (first N shots)
    python scripts/recreate.py --beat 0.896         # force a regular cut cadence (s)
    python scripts/recreate.py --no-persist         # skip writing the reels/slots run record
"""
import argparse
import dataclasses
import os

from reforge.deconstruct.reference import deconstruct
from reforge.llm import get_captioner
from reforge.match.matcher import build_edl
from reforge.render.engine import render_reel
from reforge.render.overlay import burn_text

FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"


def _persist(args, decon, matched, out_path):
    from reforge.app.queries import alternates_json, update_slot_candidates
    from reforge.db import client as db

    total = sum(m.render.duration for m in matched)
    reel_id = db.insert_reel(
        reference_key=args.reference,
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
        # runner-up plans -> slots.candidates (the review app's one-tap swap options)
        alts = alternates_json(getattr(m, "alternates", []) or [])
        if alts:
            update_slot_candidates(slot_id, alts)
    return reel_id


def main() -> None:
    ap = argparse.ArgumentParser(description="Auto-recreate a reference reel (Phase 3.5 taste layer).")
    ap.add_argument("--reference", default="reference/DISCIPLINE.mp4")
    ap.add_argument("--out", default="output/DISCIPLINE_recreation_phase36.mp4")
    ap.add_argument("--beat", type=float, default=None, help="force a regular cut cadence (s)")
    ap.add_argument("--k", type=int, default=20, help="candidates retrieved per shot (wide pool)")
    ap.add_argument("--shortlist", type=int, default=6, help="finalists after the cheap shortlist")
    ap.add_argument("--max-shots", type=int, default=None, help="only build the first N shots")
    ap.add_argument("--no-persist", action="store_true", help="skip writing the reels/slots record")
    ap.add_argument("--no-gate", action="store_true",
                    help="skip the holistic gate (review vs gold references + re-pick weak beats)")
    args = ap.parse_args()

    work = "data/recreate_work"
    vision = get_captioner()  # ClaudeVision: read_reference + shortlist (Haiku), rerank (Opus)

    print(f"[1/3] deconstruct {args.reference}")
    decon = deconstruct(args.reference, vision, work, beat_s=args.beat)
    b = decon.brief
    print(f"      BRIEF - vibe={b.vibe!r} register={b.emotional_register!r} energy={b.energy_profile!r}")
    print(f"              tone={b.tone!r} theme={b.theme!r}")
    if b.trend_context:
        print(f"      TREND - {b.trend_context[:120]}")
    print(f"      {len(decon.shots)} shots | overlay={decon.overlay_text!r}")
    for s in decon.shots:
        print(f"        shot {s.idx:02d} [{s.start_s:5.2f}-{s.end_s:5.2f}] role={s.role!r} "
              f"feel={s.intended_feeling!r} q={s.query!r}")

    if args.max_shots is not None:
        decon = dataclasses.replace(
            decon,
            shots=decon.shots[: args.max_shots],
            frames_per_shot=decon.frames_per_shot[: args.max_shots],
        )

    print(f"[2/3] match (retrieve k={args.k} -> shortlist {args.shortlist} -> taste rerank; "
          f"+diversity/freshness) over {len(decon.shots)} shots")
    matched = build_edl(decon, vision, k=args.k, shortlist_n=args.shortlist)
    from collections import Counter
    clip_ids, loc_counts = [], Counter()
    for m in matched:
        sr = m.render
        mom = m.plan.moment
        clip_ids.append(mom.clip_id)
        loc_counts[mom.location or "?"] += 1
        print(f"        slot {m.shot.idx:02d}: {os.path.basename(sr.src):30} loc={(mom.location or '?'):8} "
              f"in={sr.in_point:5.2f} dur={sr.duration:.2f} br={sr.brightness}")
        print(f"                 reason: {m.plan.reason}")
    assert all(m.render.zoom == 1.0 for m in matched), "LOCKED recipe: zoom must be 1.0 everywhere"
    dupes = [c for c, n in Counter(clip_ids).items() if n > 1]
    over_cap = {loc: n for loc, n in loc_counts.items() if n > 2}
    print(f"      zoom==1.0 on all {len(matched)} slots ✓ | unique clips: {len(set(clip_ids))}/{len(matched)} "
          f"| repeats: {dupes or 'none'} | locations: {dict(loc_counts)} | over-cap(2): {over_cap or 'none'}")

    if not args.no_gate:
        from reforge.match.gate import run_gate
        from reforge.taste import append_taste_log

        run = os.path.splitext(os.path.basename(args.out))[0]
        print("[gate] holistic review vs gold references (re-pick clearly-weak beats)")
        matched, events = run_gate(decon, matched, vision)
        for e in events:
            print(f"        iter {e['iter']} slot {e['slot']:02d}: {e['old']} -> {e['new']}")
            print(f"                 critique: {e['critique']}")
            append_taste_log({**e, "run": run})
        iters = max((e["iter"] for e in events), default=1)
        append_taste_log({"event": "gate_summary", "run": run, "slots": len(matched),
                          "repicks": len(events), "iters": iters})
        print(f"       gate: {len(events)} re-pick(s) over {iters} round(s); "
              f"{len(matched)} slots hold")
        assert all(m.render.zoom == 1.0 for m in matched), "LOCKED recipe: zoom must be 1.0 everywhere"

    print(f"[3/3] render -> {args.out}")
    # render_reel's concat list needs an ABSOLUTE tmp_dir; burn the overlay separately
    # (this Homebrew ffmpeg lacks the `drawtext` filter - no libfreetype).
    silent = os.path.splitext(args.out)[0] + "_notext.mp4"
    render_reel([m.render for m in matched], decon.audio_path, silent, text="",
                tmp_dir=os.path.abspath(os.path.join(work, "segs")), font=FONT)
    out = burn_text(silent, decon.overlay_text, args.out, font=FONT)
    os.remove(silent)
    print(f"WROTE {out} ({os.path.getsize(out)} bytes)")

    if not args.no_persist:
        reel_id = _persist(args, decon, matched, out)
        print(f"persisted run: reels.id={reel_id} (+{len(matched)} slots)")

    cost = vision.cost_summary()
    by_model = "  ".join(f"{m.split('-',1)[-1]}:{d['calls']}call/${d['usd']:.3f}"
                         for m, d in cost["per_model"].items())
    print(f"COST: ${cost['total_usd']:.3f} over {cost['total_calls']} calls  [{by_model}]")


if __name__ == "__main__":
    main()
