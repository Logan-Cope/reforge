"""Reference reel -> cut grid + shot spec (+ frames, audio, overlay text).

`cut_grid` detects shot boundaries via ffmpeg scene detection, with a **beat-cadence
fallback** for regular-cadence reels (e.g. the DISCIPLINE reel's ~0.87s grid) that scene
detection under-segments because adjacent shots look similar. `deconstruct` packages
everything Phase 3 needs: ShotSpecs, per-shot frames (for image retrieval), the extracted
audio track, and the persistent on-screen overlay text to burn.

`estimate_beat` and `dominant_overlay` are pure (unit-tested).
"""
from __future__ import annotations

import os
import re
import subprocess
from collections import Counter
from dataclasses import dataclass

from ..llm.base import ReelBrief, ShotSpec


@dataclass
class Deconstruction:
    """Everything the matcher + render need from a reference reel."""
    brief: ReelBrief                   # reel-level creative read (Phase 3.5)
    shots: list[ShotSpec]
    frames_per_shot: list[list[str]]   # parallel to shots; [i][0] is the representative frame
    audio_path: str
    overlay_text: str


# --- pure helpers -----------------------------------------------------------

def estimate_beat(cuts: list[float]) -> float | None:
    """Median gap between consecutive cut timestamps (the regular cadence), or None."""
    ordered = sorted(cuts)
    if len(ordered) < 2:
        return None
    gaps = sorted(b - a for a, b in zip(ordered, ordered[1:]))
    mid = len(gaps) // 2
    return gaps[mid] if len(gaps) % 2 else (gaps[mid - 1] + gaps[mid]) / 2


def dominant_overlay(shots: list[ShotSpec]) -> str:
    """The most common non-empty on-screen text across shots (the persistent overlay)."""
    texts = [s.on_screen_text.strip() for s in shots if s.on_screen_text and s.on_screen_text.strip()]
    return Counter(texts).most_common(1)[0][0] if texts else ""


# --- I/O --------------------------------------------------------------------

def _probe_duration(ref_path: str) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", ref_path],
        capture_output=True, text=True).stdout.strip()
    return float(out)


def cut_grid(ref_path: str, threshold: float = 0.2, beat_s: float | None = None,
             min_shots: int = 8) -> list[tuple[float, float]]:
    """Detect cut boundaries -> list of (start_s, end_s) per shot.

    If ``beat_s`` is given, tile the whole reel at that cadence (explicit regular grid).
    Otherwise scene-detect; if that under-segments (< ``min_shots`` shots, common for a
    regular-cadence reel of similar shots), fall back to a beat grid at the cadence
    estimated from the few detected cuts.
    """
    from ..ingest.segment import fixed_windows

    duration = _probe_duration(ref_path)
    if beat_s:
        return fixed_windows(duration, target_s=beat_s)

    out = subprocess.run(
        ["ffmpeg", "-nostdin", "-i", ref_path,
         "-filter:v", f"select='gt(scene,{threshold})',showinfo", "-an", "-f", "null", "-"],
        capture_output=True, text=True).stderr
    cuts = [float(t) for t in re.findall(r"pts_time:([0-9.]+)", out)]
    bounds = [0.0] + cuts + [duration]
    grid = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]

    if len(grid) < min_shots:
        beat = estimate_beat(cuts) or (duration / min_shots)
        return fixed_windows(duration, target_s=beat)
    return grid


def extract_frames(ref_path: str, grid: list[tuple[float, float]], out_dir: str,
                   per_shot: int = 2, height: int = 720) -> list[list[str]]:
    """Sample ``per_shot`` frames inside each shot window; return frames grouped by shot."""
    from ..ingest.frames import extract_frame, sample_timestamps

    os.makedirs(out_dir, exist_ok=True)
    frames_per_shot: list[list[str]] = []
    for i, (start, end) in enumerate(grid):
        paths = []
        for j, offset in enumerate(sample_timestamps(end - start, n=per_shot)):
            out = os.path.join(out_dir, f"shot{i:02d}_{j}.jpg")
            paths.append(extract_frame(ref_path, start + offset, out, height=height))
        frames_per_shot.append(paths)
    return frames_per_shot


def extract_audio(ref_path: str, out_path: str) -> str:
    """Copy the reference reel's audio track to ``out_path`` (for the recreation's mux)."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-nostdin", "-i", ref_path, "-vn", "-acodec", "copy", "-y", out_path],
        capture_output=True, check=True)
    return out_path


def deconstruct(ref_path: str, vision, work_dir: str, *, threshold: float = 0.2,
                beat_s: float | None = None, per_shot: int = 2) -> Deconstruction:
    """Full deconstruct: cut grid + frames -> `vision.read_reference` -> Deconstruction."""
    grid = cut_grid(ref_path, threshold=threshold, beat_s=beat_s)
    frames_per_shot = extract_frames(ref_path, grid, os.path.join(work_dir, "frames"), per_shot=per_shot)
    brief, shots = vision.read_reference(frames_per_shot, grid)
    overlay = dominant_overlay(shots)

    from ..config import settings
    if settings.tavily_api_key:
        from ..acquire.trends import trend_context
        brief.trend_context = trend_context(
            brief, overlay_text=overlay, ref_name=os.path.basename(ref_path))

    audio_path = extract_audio(ref_path, os.path.join(work_dir, "audio.m4a"))
    return Deconstruction(brief, shots, frames_per_shot, audio_path, overlay)
