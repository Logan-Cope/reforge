"""Pure-helper tests for scripts/repick_slot.py (re-pick one flagged beat).

Only the pure helpers are tested: rebuilding the ShotSpec/ReelBrief from persisted
columns (with graceful degradation for missing fields - reels 8-14 predate some
persistence), the critique composed from the human's flag, the sibling-slot variety
state, and the refreshed candidates jsonb (the NEW pick must be present with the
render's in_point, or ``effective_in_point`` cannot resolve the swap). The DB reads,
frame extraction, and match_slot call are I/O seams.
"""
import importlib.util
import pathlib

from reforge.llm.base import ClipMoment, SlotPlan
from reforge.match.matcher import MatchedSlot


def _load():
    path = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "repick_slot.py"
    spec = importlib.util.spec_from_file_location("reforge_repick", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rp = _load()


# --- shot_from_slot ------------------------------------------------------------

def test_shot_from_slot_rebuilds_the_reference_shot():
    shot = rp.shot_from_slot(3, 2.6, 3.5, "man lifting in dark gym", "NO EXCUSES", "explosive")
    assert shot.idx == 3
    assert shot.start_s == 2.6 and shot.end_s == 3.5
    assert shot.description == "man lifting in dark gym"
    assert shot.on_screen_text == "NO EXCUSES"
    assert shot.energy == "explosive"
    assert shot.query == "man lifting in dark gym"   # query not persisted: description stands in


def test_shot_from_slot_degrades_missing_columns_to_empty():
    shot = rp.shot_from_slot(0, 0, 1.2, None, None, None)
    assert shot.description == "" and shot.on_screen_text == "" and shot.energy == ""
    assert shot.query == ""


# --- brief_from_json -------------------------------------------------------------

def test_brief_from_json_maps_fields():
    b = rp.brief_from_json({"vibe": "gritty grind", "emotional_register": "moody",
                            "energy_profile": "builds", "tone": "raw", "theme": "discipline",
                            "trend_context": "gym motivation trend"})
    assert b.vibe == "gritty grind" and b.theme == "discipline"
    assert b.trend_context == "gym motivation trend"


def test_brief_from_json_degrades_missing_or_none():
    assert rp.brief_from_json(None).vibe == ""
    assert rp.brief_from_json({"vibe": "x", "unknown_key": "y"}).vibe == "x"  # extras ignored


# --- critique_from_flag ------------------------------------------------------------

def test_critique_carries_the_flag_text():
    c = rp.critique_from_flag("caption placement: covers my face")
    assert "Human flagged this pick" in c
    assert "caption placement: covers my face" in c
    assert "addresses this" in c


def test_critique_tolerates_missing_note():
    assert "no reason given" in rp.critique_from_flag(None)


# --- reel_state_from_rows -----------------------------------------------------------

def test_reel_state_from_rows_builds_variety_state():
    rows = [(10, 100, "gym", "dumbbell press"),
            (10, 101, "gym", "cable row"),
            (11, 102, None, "dock sunrise")]
    state = rp.reel_state_from_rows(rows)
    assert state["used_clip_ids"] == {10, 11}
    assert state["used_moment_ids"] == {100, 101, 102}
    assert state["location_counts"] == {"gym": 2, "?": 1}
    assert state["clip_use_count"] == {10: 2, 11: 1}
    assert state["used_looks"][0] == "gym: dumbbell press"


def test_reel_state_from_rows_empty_reel():
    state = rp.reel_state_from_rows([])
    assert state["used_clip_ids"] == set() and state["used_looks"] == []


# --- candidates_for_slot -------------------------------------------------------------

class _Render:
    in_point = 3.7


def _plan(moment_id, score, in_point=1.2):
    moment = ClipMoment(f"library/C{moment_id}.MP4", 0.0, 8.0, "desc", True, "fn", "moderate",
                        moment_id=moment_id, clip_id=moment_id,
                        thumbnail_key=f"thumbs/h{moment_id}/000.jpg")
    return SlotPlan(moment=moment, in_point=in_point, score=score, reason="fits")


def test_candidates_put_the_new_pick_first_with_the_render_in_point():
    """effective_in_point resolves a swapped slot from the candidates entry matching
    swapped_moment_id - the new pick MUST be present, with the clamped render seconds."""
    matched = MatchedSlot(shot=object(), plan=_plan(950, 0.9, in_point=9.9),
                          render=_Render(), alternates=[_plan(951, 0.7), _plan(952, 0.5)])
    cands = rp.candidates_for_slot(matched)
    assert [c["moment_id"] for c in cands] == [950, 951, 952]
    assert cands[0]["in_point"] == 3.7               # render's clamped in, not the raw plan's
    assert cands[1]["in_point"] == 1.2               # runner-ups keep their own previews
