"""Unit tests for the pure matcher logic in match/matcher.py (no DB / no model)."""
from reforge.llm.base import ClipMoment, SlotPlan
from reforge.match.matcher import (
    clamp_in_point,
    diversity_filter,
    framing_from_recipe,
    pick_best,
)


def _cm(clip_id, location):
    return ClipMoment("k", 0.0, 1.6, "d", True, "fn", "moderate",
                      location=location, clip_id=clip_id, moment_id=clip_id * 10)


def _plan(score, second_person=False):
    moment = ClipMoment("library/x.mp4", 0.0, 1.6, "d", True, "fn", "moderate")
    return SlotPlan(moment=moment, in_point=0.5, score=score, second_person=second_person)


# --- in_point clamping ------------------------------------------------------

def test_clamp_in_point_keeps_slot_inside_clip():
    assert clamp_in_point(2.0, 0.87, 10.0) == 2.0           # fits
    assert clamp_in_point(9.8, 0.87, 10.0) == 10.0 - 0.87   # pulled back so slot fits
    assert clamp_in_point(-1.0, 0.87, 10.0) == 0.0          # never negative


def test_clamp_in_point_clip_shorter_than_slot():
    assert clamp_in_point(3.0, 2.0, 1.0) == 0.0             # can't fit; start at 0


# --- framing stays on the LOCKED recipe (NEVER zoom) ------------------------

def test_framing_forces_zoom_one_even_if_asked_to_punch_in():
    f = framing_from_recipe(zoom=0.7, xfrac=0.35, yfrac=0.5, brightness="strong")
    assert f["zoom"] == 1.0                                  # the v1-vs-v2 guardrail
    assert f["xfrac"] == 0.35
    assert f["brightness"] == "strong"


def test_framing_defaults_and_clamps():
    f = framing_from_recipe()
    assert f == {"zoom": 1.0, "xfrac": 0.5, "yfrac": 0.5, "brightness": "none"}
    assert framing_from_recipe(xfrac=1.5)["xfrac"] == 1.0    # clamp to [0,1]
    assert framing_from_recipe(brightness="bogus")["brightness"] == "none"  # whitelist


# --- candidate selection (2-person backstop) --------------------------------

def test_pick_best_skips_second_person_candidates():
    plans = [_plan(0.9, second_person=True), _plan(0.8), _plan(0.7)]
    assert pick_best(plans).score == 0.8                     # drop the flagged top one


def test_pick_best_falls_back_when_all_flagged():
    plans = [_plan(0.9, second_person=True), _plan(0.8, second_person=True)]
    assert pick_best(plans).score == 0.9                     # better than a gap


def test_pick_best_none_when_empty():
    assert pick_best([]) is None


# --- intra-reel diversity filter --------------------------------------------

def test_diversity_filter_hard_excludes_already_used_clips():
    out = diversity_filter([_cm(1, "gym"), _cm(2, "dock"), _cm(3, "gym")],
                           used_clip_ids={2}, location_counts={}, location_cap=2)
    assert [c.clip_id for c in out] == [1, 3]


def test_diversity_filter_soft_excludes_capped_location_when_alternatives_exist():
    out = diversity_filter([_cm(1, "gym"), _cm(2, "dock")],
                           used_clip_ids=set(), location_counts={"gym": 2}, location_cap=2)
    assert [c.clip_id for c in out] == [2]   # gym at cap, dock kept


def test_diversity_filter_relaxes_soft_cap_rather_than_emptying():
    out = diversity_filter([_cm(1, "gym"), _cm(3, "gym")],
                           used_clip_ids=set(), location_counts={"gym": 2}, location_cap=2)
    assert [c.clip_id for c in out] == [1, 3]  # all gym -> relax soft, keep them


def test_diversity_filter_empty_when_all_clips_used():
    out = diversity_filter([_cm(1, "gym"), _cm(2, "dock")],
                           used_clip_ids={1, 2}, location_counts={}, location_cap=2)
    assert out == []  # signal the matcher's least-used fallback
