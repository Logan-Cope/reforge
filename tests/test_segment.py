"""Unit tests for the pure windowing logic in ingest/segment.py.

These touch no media: only the mechanical window math that turns a clip duration
into ~1.5-2s moment windows (the locked Phase 2 design: mechanical windows, not
VLM-timestamped moments).
"""
import pytest

from reforge.ingest.segment import fixed_windows, stable_spans


# --- Phase 3.7 quality-aware segmentation: stable spans ---------------------

def test_stable_spans_trims_high_motion_head_and_tail():
    # motion: high at the ends (set-up / set-down), low (stable) in the middle
    times = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    motion = [9.0, 1.0, 1.0, 1.0, 1.0, 8.0, 9.0]
    spans = stable_spans(motion, times, threshold=5.0, min_span_s=1.0)
    assert spans == [(0.5, 2.0)]              # head idx0 and tail idx5-6 trimmed


def test_stable_spans_drops_runs_shorter_than_min():
    times = [0.0, 0.5, 1.0, 1.5, 2.0]
    motion = [1.0, 9.0, 1.0, 9.0, 1.0]        # no stable run >= min_span
    assert stable_spans(motion, times, threshold=5.0, min_span_s=1.0) == []


def test_stable_spans_two_separate_stable_regions():
    times = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    motion = [1.0, 1.0, 1.0, 9.0, 9.0, 1.0, 1.0, 1.0]   # stable, cut, stable
    spans = stable_spans(motion, times, threshold=5.0, min_span_s=1.0)
    assert spans == [(0.0, 1.0), (2.5, 3.5)]


def test_stable_spans_all_stable_is_one_span():
    times = [0.0, 1.0, 2.0]
    assert stable_spans([1.0, 1.0, 1.0], times, threshold=5.0, min_span_s=1.0) == [(0.0, 2.0)]


def test_short_clip_returns_single_window_covering_whole_clip():
    assert fixed_windows(1.2) == [(0.0, 1.2)]


def test_exact_multiple_of_target_tiles_evenly():
    w = fixed_windows(7.0, target_s=1.75)
    assert w == [(0.0, 1.75), (1.75, 3.5), (3.5, 5.25), (5.25, 7.0)]


def test_windows_tile_without_gaps_or_overlaps():
    w = fixed_windows(10.0)
    for (_s0, e0), (s1, _e1) in zip(w, w[1:]):
        assert e0 == pytest.approx(s1)
    assert w[0][0] == 0.0
    assert w[-1][1] == pytest.approx(10.0)


def test_window_count_picks_nearest_to_target():
    # 10.0 / 1.75 = 5.71 -> rounds to 6 windows
    assert len(fixed_windows(10.0, target_s=1.75)) == 6


def test_windows_stay_near_target_for_typical_durations():
    for dur in [3.0, 5.0, 8.0, 12.0, 20.0]:
        lengths = [e - s for s, e in fixed_windows(dur)]
        for length in lengths:
            assert 1.4 <= length <= 2.2


def test_rejects_nonpositive_duration():
    with pytest.raises(ValueError):
        fixed_windows(0.0)
    with pytest.raises(ValueError):
        fixed_windows(-3.0)
