"""Unit tests for the Phase 3.5 taste-layer dataclasses."""
from reforge.llm.base import ClipMoment, ReelBrief, ShotSpec, SlotPlan


def test_reelbrief_carries_the_reel_creative_read():
    brief = ReelBrief(vibe="late-night grind", emotional_register="gritty",
                      energy_profile="builds to explosive", tone="serious", theme="discipline")
    assert brief.emotional_register == "gritty"
    assert brief.theme == "discipline"


def test_shotspec_taste_fields_default_blank():
    # existing positional construction (pre-3.5) must still work
    shot = ShotSpec(0, 0.0, 1.0, "a pull-up", "", "explosive", "pull-up")
    assert shot.role == ""
    assert shot.intended_feeling == ""
    assert shot.intended_energy == ""


def test_shotspec_taste_fields_settable():
    shot = ShotSpec(0, 0.0, 1.0, "a pull-up", "", "explosive", "pull-up",
                    role="opener", intended_feeling="hook the viewer", intended_energy="explosive")
    assert shot.role == "opener"
    assert shot.intended_feeling == "hook the viewer"


def test_slotplan_reason_defaults_blank():
    moment = ClipMoment("k", 0.0, 1.0, "d", True, "fn", "moderate")
    assert SlotPlan(moment=moment, in_point=0.5).reason == ""
