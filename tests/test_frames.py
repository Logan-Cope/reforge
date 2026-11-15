"""Unit tests for the pure frame-sampling logic in ingest/frames.py.

`sample_timestamps` picks the times to grab representative frames for clip-level
captioning (Claude reads a few frames per clip). No media is touched here.
"""
import pytest

from reforge.ingest.frames import sample_timestamps


def test_sample_timestamps_count():
    assert len(sample_timestamps(10.0, n=4)) == 4


def test_sample_timestamps_are_interior_and_sorted():
    ts = sample_timestamps(10.0, n=4)
    assert ts == sorted(ts)
    assert ts[0] > 0.0          # never the very first frame (often a transition)
    assert ts[-1] < 10.0        # never past the end


def test_sample_timestamps_are_evenly_spaced_segment_centers():
    assert sample_timestamps(8.0, n=4) == pytest.approx([1.0, 3.0, 5.0, 7.0])


def test_sample_timestamps_single_frame_is_midpoint():
    assert sample_timestamps(5.0, n=1) == pytest.approx([2.5])


def test_sample_timestamps_rejects_bad_args():
    with pytest.raises(ValueError):
        sample_timestamps(0.0, n=4)
    with pytest.raises(ValueError):
        sample_timestamps(10.0, n=0)
