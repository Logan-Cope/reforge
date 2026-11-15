"""Probe a clip's media metadata (duration / width / height / fps) via ffprobe.

`parse_probe` is the pure parser over an ffprobe ``-of json`` payload (unit-tested,
no subprocess). `probe_media` is the thin I/O wrapper that shells out to ffprobe.
"""
from __future__ import annotations

from typing import Any


def _parse_frame_rate(rate: str | None) -> float | None:
    """Parse an ffprobe ``r_frame_rate`` like ``"30000/1001"`` into fps, or None."""
    if not rate:
        return None
    num, _, den = rate.partition("/")
    try:
        numerator, denominator = float(num), float(den or 1)
    except ValueError:
        return None
    return numerator / denominator if denominator else None


def parse_probe(payload: dict[str, Any]) -> dict[str, Any]:
    """Extract media fields from an ffprobe JSON payload.

    Returns ``{duration_s, width, height, fps}``. Duration prefers the container
    (``format.duration``) and falls back to the video stream's duration. Raises
    ValueError if there is no video stream.
    """
    streams = payload.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise ValueError("no video stream in ffprobe output")

    duration = payload.get("format", {}).get("duration") or video.get("duration")
    return {
        "duration_s": float(duration) if duration is not None else None,
        "width": int(video["width"]) if video.get("width") is not None else None,
        "height": int(video["height"]) if video.get("height") is not None else None,
        "fps": _parse_frame_rate(video.get("r_frame_rate")),
    }


def probe_media(local_path: str) -> dict[str, Any]:
    """Run ffprobe on ``local_path`` and return its parsed media fields."""
    import json
    import subprocess

    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", local_path],
        capture_output=True, text=True, check=True,
    ).stdout
    return parse_probe(json.loads(out))
