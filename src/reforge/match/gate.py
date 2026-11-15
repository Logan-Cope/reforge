"""Holistic quality gate (taste-architecture step 6): the pass that reaches postable.

After `build_edl` assembles the pick sequence, the gate views the WHOLE sequence next to
the gold-reference reels + Logan's rubric, flags clearly-weak beats, and re-picks just
those beats via `matcher.match_slot` (excluding the rejected moment, with the gate's
critique as rerank feedback). Iterates up to `settings.gate_max_iters` rounds or until
every slot holds. The pure helpers (`slot_line`, `reel_state_excluding`,
`gold_reference_paths`) are unit-tested; `run_gate` lazily imports ffmpeg/db seams.
"""
from __future__ import annotations

import glob
import os
import shutil
import tempfile

from .matcher import MatchedSlot

_GOLD_REELS_CAP = 2       # at most this many gold reels sampled into the prompt
_GOLD_FRAMES_PER_REEL = 6
_SLOT_FRAMES = 2          # frames extracted per assembled slot (in_point + midpoint)


def gold_reference_paths(gold_dir: str | None = None) -> list[str]:
    """Sorted ``*.mp4`` paths in the gold-references taste dir (the quality bar)."""
    if gold_dir is None:
        from ..config import settings
        gold_dir = os.path.join(settings.taste_dir, "gold-references")
    return sorted(glob.glob(os.path.join(gold_dir, "*.mp4")))


def slot_line(m: MatchedSlot) -> str:
    """One text line describing a matched slot for the gate prompt (pure)."""
    return (f"slot {m.shot.idx:02d} | {os.path.basename(m.render.src)} | "
            f"loc={m.plan.moment.location or '?'} | reason: {m.plan.reason}")


def reel_state_excluding(matched: list[MatchedSlot], skip_idx: int) -> dict:
    """Rebuild the reel-variety state from every slot EXCEPT ``skip_idx`` (pure).

    Same fields `build_edl` tracks - so a gate re-pick cannot duplicate another slot's
    clip, blow the location cap, or repeat another slot's look.
    """
    used_clip_ids: set = set()
    used_moment_ids: set = set()
    location_counts: dict = {}
    clip_use_count: dict = {}
    used_looks: list[str] = []
    for i, m in enumerate(matched):
        if i == skip_idx:
            continue
        moment = m.plan.moment
        loc = moment.location or "?"
        used_clip_ids.add(moment.clip_id)
        used_moment_ids.add(moment.moment_id)
        location_counts[loc] = location_counts.get(loc, 0) + 1
        clip_use_count[moment.clip_id] = clip_use_count.get(moment.clip_id, 0) + 1
        used_looks.append(f"{loc}: {(moment.description or '').strip()[:45]}")
    return {"used_clip_ids": used_clip_ids, "used_moment_ids": used_moment_ids,
            "location_counts": location_counts, "clip_use_count": clip_use_count,
            "used_looks": used_looks}


def _gold_frames(tmp_dir: str) -> list[str]:
    """Sample frames across (capped) gold reels into ``tmp_dir``; [] when none exist."""
    from ..ingest.frames import extract_frame, sample_timestamps
    from ..ingest.probe import probe_media

    frames: list[str] = []
    for gi, path in enumerate(gold_reference_paths()[:_GOLD_REELS_CAP]):
        duration_s = probe_media(path)["duration_s"]
        for j, t in enumerate(sample_timestamps(duration_s, n=_GOLD_FRAMES_PER_REEL)):
            frames.append(extract_frame(path, t, os.path.join(tmp_dir, f"g{gi}_{j}.jpg")))
    return frames


def run_gate(decon, matched: list[MatchedSlot], vision, *, max_iters: int | None = None,
             storage=None) -> tuple[list[MatchedSlot], list[dict]]:
    """Review the assembled reel against the gold bar; re-pick weak beats in place.

    Returns ``(matched, events)`` - ``matched`` with weak slots replaced (zoom stays 1.0;
    `match_slot` asserts it), and one ``gate_repick`` event dict per replacement.
    """
    from ..config import settings
    from ..db import client as db
    from ..ingest.frames import extract_frame
    from ..storage.base import get_storage
    from .matcher import match_slot

    max_iters = settings.gate_max_iters if max_iters is None else max_iters
    storage = storage or get_storage()
    try:
        usage = db.clip_usage()
    except Exception:
        usage = {}

    events: list[dict] = []
    # Rejections accumulate per slot across rounds, at CLIP level: a critique is almost
    # always about the clip's whole look, so a re-pick must not return to the same clip
    # via a different moment (seen live: a mirror-gym clip re-picked twice from itself).
    rejected_clips: dict[int, set] = {}
    rejected_moments: dict[int, set] = {}
    gold_tmp = tempfile.mkdtemp(prefix="reforge_gate_gold_")
    try:
        gold_frames = _gold_frames(gold_tmp)
        for it in range(1, max_iters + 1):
            tmp = tempfile.mkdtemp(prefix=f"reforge_gate_{it}_")
            try:
                slot_frames = []
                for i, m in enumerate(matched):
                    r = m.render
                    times = [r.in_point + r.duration * f / _SLOT_FRAMES for f in range(_SLOT_FRAMES)]
                    slot_frames.append([
                        extract_frame(r.src, t, os.path.join(tmp, f"s{i}_{j}.jpg"))
                        for j, t in enumerate(times)
                    ])
                verdicts = vision.gate_review(decon.brief, [slot_line(m) for m in matched],
                                              slot_frames, gold_frames)
            finally:
                shutil.rmtree(tmp, ignore_errors=True)

            weak = [v for v in verdicts if not v.hold]
            if not weak:
                break
            for v in weak:
                old = matched[v.slot]
                rejected_clips.setdefault(v.slot, set()).add(old.plan.moment.clip_id)
                rejected_moments.setdefault(v.slot, set()).add(old.plan.moment.moment_id)
                new = match_slot(
                    old.shot, decon.frames_per_shot[v.slot], vision, decon.brief,
                    k=20, shortlist_n=6, cand_frames=5, storage=storage, usage=usage,
                    fresh_w=settings.freshness_weight, cap=settings.diversity_location_cap,
                    exclude_moment_ids=rejected_moments[v.slot],
                    hard_exclude_clip_ids=rejected_clips[v.slot], critique=v.critique,
                    **reel_state_excluding(matched, v.slot),
                )
                events.append({"event": "gate_repick", "iter": it, "slot": v.slot,
                               "critique": v.critique,
                               "old": os.path.basename(old.render.src),
                               "new": os.path.basename(new.render.src)})
                matched[v.slot] = new
    finally:
        shutil.rmtree(gold_tmp, ignore_errors=True)
    return matched, events
