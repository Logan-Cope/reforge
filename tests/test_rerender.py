"""Pure-helper tests for scripts/rerender_reel.py (the swap-applying re-render CLI).

Only the pure helpers are tested: the effective in_point per slot (swap-aware - now
lives in ``reforge.app.queries`` so the review app's cards and the re-render share ONE
source of truth; the script imports it from there), the dominant overlay text across
slot OCR strings, and the versioned output filename that never overwrites an existing
render. The DB read + ffmpeg render are I/O seams.
"""
import importlib.util
import pathlib

from reforge.app.queries import effective_in_point


def _load():
    path = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "rerender_reel.py"
    spec = importlib.util.spec_from_file_location("reforge_rerender", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rr = _load()

CANDS = [
    {"moment_id": 901, "clip_key": "library/C901.MP4", "in_point": 1.2},
    {"moment_id": 902, "clip_key": "library/C902.MP4", "in_point": 7.5},
]


# --- effective_in_point (single source of truth in reforge.app.queries) ----------

def test_script_uses_the_relocated_helper():
    """The re-render CLI and the app must compute in_points with the SAME function."""
    assert rr.effective_in_point is effective_in_point


def test_in_point_unswapped_uses_params():
    assert effective_in_point({"in_point": 4.2}, CANDS, None, 0.0, 10.0) == 4.2


def test_in_point_swapped_uses_matching_candidate():
    assert effective_in_point({"in_point": 4.2}, CANDS, 902, 0.0, 10.0) == 7.5


def test_in_point_swapped_no_candidate_match_falls_back_to_moment_midpoint():
    assert effective_in_point({"in_point": 4.2}, CANDS, 999, 2.0, 6.0) == 4.0


def test_in_point_swapped_none_candidates_falls_back_to_midpoint():
    assert effective_in_point({"in_point": 4.2}, None, 901, 10.0, 20.0) == 15.0


def test_in_point_unswapped_missing_params_defaults_to_zero():
    assert effective_in_point(None, [], None, 0.0, 10.0) == 0.0


# --- dominant_text ----------------------------------------------------------------

def test_dominant_text_picks_most_common_nonempty():
    assert rr.dominant_text(["DISCIPLINE", "", "DISCIPLINE", "grind", None]) == "DISCIPLINE"


def test_dominant_text_strips_whitespace():
    assert rr.dominant_text(["  hi  ", "hi", "  "]) == "hi"


def test_dominant_text_all_empty_is_blank():
    assert rr.dominant_text(["", None, "   "]) == ""
    assert rr.dominant_text([]) == ""


# --- versioned_output ---------------------------------------------------------------

def test_versioned_output_starts_at_v2():
    assert rr.versioned_output(8, exists=lambda p: False) == "output/reel8_v2.mp4"


def test_versioned_output_skips_taken_suffixes():
    taken = {"output/reel8_v2.mp4", "output/reel8_v3.mp4"}
    assert rr.versioned_output(8, exists=lambda p: p in taken) == "output/reel8_v4.mp4"


def test_versioned_output_never_returns_existing_path():
    taken = {f"output/reel3_v{k}.mp4" for k in range(2, 9)}
    out = rr.versioned_output(3, exists=lambda p: p in taken)
    assert out == "output/reel3_v9.mp4"
    assert out not in taken
