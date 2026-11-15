"""Per-shot match -> SlotRender EDL.

For each reference shot: retrieve candidates -> rerank (rank + framing + peak in_point +
2-person flag) -> pick the best single-person candidate -> emit a `render.engine.SlotRender`
whose duration is the reference cut-grid slot (this inherits beat-sync). Framing stays on
the LOCKED recipe: **zoom is always 1.0** (asserted), never punch in.

`clamp_in_point`, `framing_from_recipe`, and `pick_best` are pure (unit-tested);
`build_edl` lazily imports retrieve/render/probe.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass, field

from ..llm.base import ClipMoment, ReelBrief, ShotSpec, SlotPlan


@dataclass
class MatchedSlot:
    """One matched slot: the reference shot, the rerank's chosen plan, the render spec."""
    shot: ShotSpec
    plan: SlotPlan          # chosen moment + in_point + framing + reason
    render: object          # render.engine.SlotRender
    alternates: list = field(default_factory=list)  # runner-up SlotPlans (best-first, <=3) for the swap UI


# --- pure helpers -----------------------------------------------------------

def clamp_in_point(suggested: float, slot_dur: float, clip_dur: float) -> float:
    """Keep ``[in_point, in_point+slot_dur]`` inside ``[0, clip_dur]``."""
    latest = max(0.0, clip_dur - slot_dur)
    return max(0.0, min(suggested, latest))


def framing_from_recipe(zoom: float | None = None, xfrac: float | None = None,
                        yfrac: float | None = None, brightness: str | None = None) -> dict:
    """Sanitize rerank framing hints to the LOCKED recipe.

    ``zoom`` is IGNORED and forced to 1.0 - never punch in (the v1-vs-v2 rule). Only
    xfrac/yfrac (off-center subject) and brightness (dark night) may deviate.
    """
    x = 0.5 if xfrac is None else min(1.0, max(0.0, xfrac))
    y = 0.5 if yfrac is None else min(1.0, max(0.0, yfrac))
    b = brightness if brightness in ("none", "mild", "strong") else "none"
    return {"zoom": 1.0, "xfrac": x, "yfrac": y, "brightness": b}


def pick_best(plans: list[SlotPlan]) -> SlotPlan | None:
    """First non-second-person plan (plans are best-first); fall back to best if all flagged."""
    if not plans:
        return None
    return next((p for p in plans if not p.second_person), plans[0])


def diversity_filter(finalists, used_clip_ids: set, location_counts: dict, location_cap: int):
    """Intra-reel variety filter (pure).

    HARD-exclude finalists whose ``clip_id`` is already used this reel (no literal repeats).
    SOFT-exclude finalists whose ``location`` already hit ``location_cap`` - but only if that
    leaves alternatives (relax the soft cap rather than emptying the pool). Returns ``[]`` when
    every finalist's clip is already used (the matcher then does its least-used fallback).
    """
    survivors = [f for f in finalists if f.clip_id not in used_clip_ids]
    under_cap = [f for f in survivors if location_counts.get(f.location or "?", 0) < location_cap]
    return under_cap if under_cap else survivors


def exclude_moments(candidates: list[ClipMoment], exclude_moment_ids) -> list[ClipMoment]:
    """Drop candidates whose ``moment_id`` is excluded (pure; the gate's re-pick filter)."""
    if not exclude_moment_ids:
        return list(candidates)
    return [c for c in candidates if c.moment_id not in exclude_moment_ids]


# --- orchestration ----------------------------------------------------------

def _other_in_point(moment, in_point: float) -> float:
    """A different point in the moment's window (the thin-beat reuse must not be identical)."""
    w = (moment.end_s - moment.start_s) or 1.0
    frac = (in_point - moment.start_s) / w
    return moment.start_s + (0.75 if frac < 0.5 else 0.25) * w


def match_slot(shot: ShotSpec, shot_frames: list[str], vision, brief: ReelBrief, *,
               k: int, shortlist_n: int, cand_frames: int, storage, usage: dict,
               fresh_w: float, cap: int, used_clip_ids: set, used_moment_ids: set,
               location_counts: dict, clip_use_count: dict, used_looks: list[str],
               exclude_moment_ids=frozenset(), hard_exclude_clip_ids=frozenset(),
               critique: str = "") -> MatchedSlot:
    """Match ONE reference shot: retrieve -> shortlist -> diversity -> taste rerank -> pick.

    The reel-state args (``used_*``, ``location_counts``, ``clip_use_count``) are READ, not
    mutated - the caller carries state forward. ``exclude_moment_ids`` drops candidates
    outright (the gate's re-pick must not re-choose the rejected moment) and ``critique`` is
    the gate's note on why the previous pick failed, passed through to the rerank.
    """
    from ..ingest.frames import extract_frame, sample_timestamps
    from ..ingest.probe import probe_media
    from ..render.engine import SlotRender
    from .retrieve import retrieve, shortlist

    slot_dur = shot.end_s - shot.start_s
    candidates = exclude_moments(retrieve(shot, shot_frames[0], k=k, brief=brief), exclude_moment_ids)
    if hard_exclude_clip_ids:  # gate-rejected clips: unreachable even via the thin-beat fallback
        candidates = [c for c in candidates if c.clip_id not in hard_exclude_clip_ids]
    if not candidates:
        raise ValueError(f"no candidates retrieved for shot {shot.idx}")
    finalists = shortlist(vision, brief, shot, candidates, n=shortlist_n,
                          usage=usage, freshness_weight=fresh_w)

    pool = diversity_filter(finalists, used_clip_ids, location_counts, cap)
    force_diff = False
    if not pool:  # thin beat - every finalist's clip already used; reuse the least-used clip
        pool = [min(finalists, key=lambda f: clip_use_count.get(f.clip_id, 0))]
        force_diff = True

    tmp = tempfile.mkdtemp(prefix=f"reforge_rerank_{shot.idx:02d}_")
    try:
        candidate_frames = []
        for ci, cand in enumerate(pool):
            src = storage.ensure_local(cand.clip_key)
            frames = [
                extract_frame(src, cand.start_s + off, os.path.join(tmp, f"c{ci}_{j}.jpg"))
                for j, off in enumerate(sample_timestamps(cand.end_s - cand.start_s, n=cand_frames))
            ]
            candidate_frames.append((cand, frames))

        plans = vision.rerank(shot, shot_frames, candidate_frames, brief,
                              avoid_looks=list(used_looks), critique=critique)
        plans = sorted(plans, key=lambda p: p.score, reverse=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    chosen = pick_best(plans)
    if chosen is None:  # rerank returned nothing - fall back to the nearest pool clip
        top = pool[0]
        chosen = SlotPlan(moment=top, in_point=(top.start_s + top.end_s) / 2)
    alternates = [p for p in plans if p is not chosen][:3]  # runner-ups, best-first (swap UI)

    moment = chosen.moment
    src = storage.ensure_local(moment.clip_key)
    clip_dur = probe_media(src)["duration_s"]
    in_point = chosen.in_point
    if force_diff and moment.moment_id in used_moment_ids:  # never the identical moment
        in_point = _other_in_point(moment, in_point)

    framing = framing_from_recipe(chosen.zoom, chosen.xfrac, chosen.yfrac, chosen.brightness)
    assert framing["zoom"] == 1.0, "LOCKED recipe violated: zoom must be 1.0 (no punch-in)"
    render = SlotRender(
        src=src,
        in_point=clamp_in_point(in_point, slot_dur, clip_dur),
        duration=slot_dur,
        zoom=framing["zoom"], xfrac=framing["xfrac"], yfrac=framing["yfrac"],
        brightness=framing["brightness"],
    )
    return MatchedSlot(shot=shot, plan=chosen, render=render, alternates=alternates)


def build_edl(decon, vision, *, k: int = 20, shortlist_n: int = 6, cand_frames: int = 5,
              storage=None, location_cap=None, freshness_weight=None, usage=None) -> list[MatchedSlot]:
    """`Deconstruction` -> `[MatchedSlot]`, one per reference shot.

    3.5 wide retrieve → cheap shortlist → taste rerank. 3.6 adds VARIETY: cross-reel freshness
    in the shortlist (rotate over time), and intra-reel diversity in order - no clip repeats,
    locations capped, the rerank told which looks to avoid. Framing stays on the LOCKED recipe
    (zoom=1.0, asserted); durations = the reference cut grid (beat-sync). Per-shot work lives
    in `match_slot` (also reused by the holistic gate's re-picks).
    """
    from ..config import settings
    from ..db import client as db
    from ..storage.base import get_storage

    storage = storage or get_storage()
    cap = settings.diversity_location_cap if location_cap is None else location_cap
    fresh_w = settings.freshness_weight if freshness_weight is None else freshness_weight
    if usage is None:
        try:
            usage = db.clip_usage()
        except Exception:
            usage = {}

    brief = decon.brief
    used_clip_ids: set = set()
    used_moment_ids: set = set()
    location_counts: dict = {}
    clip_use_count: dict = {}
    used_looks: list[str] = []
    matched: list[MatchedSlot] = []

    for shot, shot_frames in zip(decon.shots, decon.frames_per_shot):
        m = match_slot(shot, shot_frames, vision, brief,
                       k=k, shortlist_n=shortlist_n, cand_frames=cand_frames,
                       storage=storage, usage=usage, fresh_w=fresh_w, cap=cap,
                       used_clip_ids=used_clip_ids, used_moment_ids=used_moment_ids,
                       location_counts=location_counts, clip_use_count=clip_use_count,
                       used_looks=used_looks)
        matched.append(m)

        # carry reel state forward
        moment = m.plan.moment
        loc = moment.location or "?"
        used_clip_ids.add(moment.clip_id)
        used_moment_ids.add(moment.moment_id)
        location_counts[loc] = location_counts.get(loc, 0) + 1
        clip_use_count[moment.clip_id] = clip_use_count.get(moment.clip_id, 0) + 1
        used_looks.append(f"{loc}: {(moment.description or '').strip()[:45]}")
    return matched
