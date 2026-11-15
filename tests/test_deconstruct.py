"""Unit tests for the pure deconstruct logic in deconstruct/reference.py."""
import pytest

from reforge.deconstruct.reference import dominant_overlay, estimate_beat
from reforge.llm.base import ShotSpec


# --- beat estimation (for the regular-cadence fallback) ---------------------

def test_estimate_beat_is_median_gap_between_cuts():
    assert estimate_beat([0.9, 1.8, 2.7, 3.6]) == pytest.approx(0.9)


def test_estimate_beat_unsorted_input():
    assert estimate_beat([2.7, 0.9, 3.6, 1.8]) == pytest.approx(0.9)


def test_estimate_beat_none_when_too_few_cuts():
    assert estimate_beat([5.0]) is None
    assert estimate_beat([]) is None


# --- dominant on-screen overlay (the persistent text to burn) ---------------

def _shot(i, text):
    return ShotSpec(idx=i, start_s=0.0, end_s=1.0, description="d",
                    on_screen_text=text, energy="moderate", query="q")


def test_dominant_overlay_picks_most_common_nonempty_text():
    shots = [_shot(0, "DISCIPLINE"), _shot(1, "DISCIPLINE"), _shot(2, ""), _shot(3, "DISCIPLINE")]
    assert dominant_overlay(shots) == "DISCIPLINE"


def test_dominant_overlay_empty_when_no_text():
    shots = [_shot(0, ""), _shot(1, "   ")]
    assert dominant_overlay(shots) == ""
