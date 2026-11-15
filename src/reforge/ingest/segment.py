"""Segment a library clip into candidate moment windows.

Locked Phase 2 design: Logan's clips are mostly one continuous take, so moments are
**mechanical ~1.5-2s windows** that tile the clip - not VLM-timestamped (Gemini
timestamps are imprecise). One clip-level annotation is inherited by every window;
the per-window CLIP frame embedding is the retriever.

`fixed_windows` is the pure window math (unit-tested, no media). `segment_clip` is the
thin I/O wrapper that probes the duration and delegates.
"""
from __future__ import annotations

# Target moment length. Equal tiling keeps each window close to this (1.5-2s band).
TARGET_WINDOW_S = 1.75


def fixed_windows(duration_s: float, target_s: float = TARGET_WINDOW_S) -> list[tuple[float, float]]:
    """Tile ``[0, duration_s]`` into equal, contiguous windows of ~``target_s`` each.

    Window count is chosen as the integer nearest ``duration_s / target_s`` (>=1), so
    every window lands near the target length and the windows exactly cover the clip
    with no gaps or overlaps. Clips shorter than the target yield one whole-clip window.
    """
    if duration_s <= 0:
        raise ValueError(f"duration_s must be positive, got {duration_s}")
    n = max(1, round(duration_s / target_s))
    step = duration_s / n
    return [(i * step, (i + 1) * step if i < n - 1 else duration_s) for i in range(n)]


def stable_spans(motion: list[float], times: list[float], threshold: float,
                 min_span_s: float) -> list[tuple[float, float]]:
    """Contiguous runs where ``motion < threshold``, kept if ≥ ``min_span_s`` (pure).

    ``motion[i]`` is the inter-frame motion at ``times[i]``. Runs are the STABLE (low-motion)
    spans; high-motion frames at a run's head/tail (set-up, set-down, walk-up, a scene cut)
    fall above threshold and are naturally trimmed. Returns ``(start, end)`` per kept span.
    """
    spans: list[tuple[float, float]] = []
    i, n = 0, len(motion)
    while i < n:
        if motion[i] < threshold:
            j = i
            while j + 1 < n and motion[j + 1] < threshold:
                j += 1
            if times[j] - times[i] >= min_span_s:
                spans.append((times[i], times[j]))
            i = j + 1
        else:
            i += 1
    return spans


def windows_from_motion(motion: list[float], times: list[float], duration: float,
                        threshold: float, target_s: float = TARGET_WINDOW_S) -> list[tuple[float, float]]:
    """Stable spans (from a precomputed motion series) tiled into ~target_s moment windows.

    Falls back to whole-clip fixed windows when motion is unavailable or the clip is all-motion,
    so a clip is never left without candidates (quality then down-weights it).
    """
    if not motion:
        return fixed_windows(duration, target_s=target_s)
    spans = stable_spans(motion, times, threshold=threshold, min_span_s=max(1.0, target_s * 0.6))
    if not spans:
        return fixed_windows(duration, target_s=target_s)
    windows: list[tuple[float, float]] = []
    for start, end in spans:
        windows.extend((start + a, start + b) for a, b in fixed_windows(end - start, target_s=target_s))
    return windows


def segment_clip(local_path: str, target_s: float = TARGET_WINDOW_S) -> list[tuple[float, float]]:
    """Quality-aware segmentation: motion-stable spans tiled into ~target_s moments."""
    from .probe import probe_media
    from .quality import motion_series, stability_threshold

    duration = probe_media(local_path)["duration_s"]
    try:
        motion, times = motion_series(local_path)
    except Exception:
        motion, times = [], []
    threshold = stability_threshold(motion) if motion else 0.0
    return windows_from_motion(motion, times, duration, threshold, target_s=target_s)
