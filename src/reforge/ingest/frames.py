"""Extract a single representative frame from a clip via ffmpeg.

Each moment window is embedded from (and thumbnailed by) one frame at its midpoint.
The extracted JPEG serves double duty: input to the CLIP image encoder and the stored
thumbnail. Kept small (downscaled) since CLIP resizes to 224px and thumbnails are proxies.
"""
from __future__ import annotations

import subprocess


def sample_timestamps(duration_s: float, n: int = 4) -> list[float]:
    """Pick ``n`` timestamps to grab representative frames for clip-level captioning.

    Points are the centers of ``n`` equal segments - evenly spread and always interior
    (never the very first/last frame, which are often transitions or black).
    """
    if duration_s <= 0:
        raise ValueError(f"duration_s must be positive, got {duration_s}")
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    return [duration_s * (i + 0.5) / n for i in range(n)]


def extract_frame(src: str, t_seconds: float, out_path: str, height: int = 720) -> str:
    """Write one JPEG frame from ``src`` at ``t_seconds`` to ``out_path``; return it.

    ``-ss`` before ``-i`` does a fast keyframe seek (accurate enough for a midpoint
    representative frame). Width is auto (``-2`` keeps it even) to preserve aspect.
    """
    subprocess.run(
        ["ffmpeg", "-nostdin", "-ss", f"{t_seconds:.3f}", "-i", src,
         "-frames:v", "1", "-vf", f"scale=-2:{height}", "-q:v", "3", "-y", out_path],
        capture_output=True, check=True,
    )
    return out_path
