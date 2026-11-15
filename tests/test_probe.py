"""Unit tests for the pure ffprobe-JSON parser in ingest/probe.py.

These touch no media: they feed a representative ffprobe ``-of json`` payload to
``parse_probe`` and assert the extracted media fields.
"""
import pytest

from reforge.ingest.probe import parse_probe


def test_parse_probe_extracts_media_fields():
    payload = {
        "streams": [
            {"codec_type": "audio"},
            {"codec_type": "video", "width": 1920, "height": 1080, "r_frame_rate": "30000/1001"},
        ],
        "format": {"duration": "12.5"},
    }
    media = parse_probe(payload)
    assert media["width"] == 1920
    assert media["height"] == 1080
    assert media["duration_s"] == pytest.approx(12.5)
    assert media["fps"] == pytest.approx(29.97, abs=0.01)


def test_parse_probe_handles_integer_frame_rate():
    payload = {
        "streams": [{"codec_type": "video", "width": 1080, "height": 1920, "r_frame_rate": "30/1"}],
        "format": {"duration": "3.0"},
    }
    assert parse_probe(payload)["fps"] == pytest.approx(30.0)


def test_parse_probe_falls_back_to_stream_duration():
    payload = {
        "streams": [
            {"codec_type": "video", "width": 100, "height": 200,
             "r_frame_rate": "24/1", "duration": "4.2"},
        ],
        "format": {},
    }
    assert parse_probe(payload)["duration_s"] == pytest.approx(4.2)


def test_parse_probe_raises_without_video_stream():
    with pytest.raises(ValueError):
        parse_probe({"streams": [{"codec_type": "audio"}], "format": {"duration": "1"}})
