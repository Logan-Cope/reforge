"""Per-moment quality signals (Phase 3.7 Increment A).

Detects a PROPERTY (camera in motion, no subject, blur, dark), never a POSITION - there is
no "don't use the end of a clip" rule. Local CV signals (no API) + a one-time VLM pass
combine into a `quality_score` that only DOWN-WEIGHTS: Increment A drops NOTHING
automatically. Both mediapipe (can't see dark / back-view / occluded / tight-crop subjects)
and the VLM (mistakes dark + soft + handheld for camera-handling) over-dropped Logan's
signature training B-roll, so every detector here informs ranking only. Junk is sunk by its
low score in retrieval + the Opus rerank; removal authority is the human Increment B pass.
Every flag (no_subject, camera_motion, camera_handling, transitional, blur, dark) is recorded
for ranking + that later human pass, never acted on to drop.

The pure scoring (`cv_quality_score`, `combine_quality`) is unit-tested; the CV signal
extraction (`moment_cv_signals`, `motion_series`) lazily imports cv2 / mediapipe.
"""
from __future__ import annotations

from ..config import settings


# --- pure scoring -----------------------------------------------------------

def cv_quality_score(stability: float, exposure: float, sharpness: float,
                     subject_present: float) -> tuple[float, list[str]]:
    """Combine normalized [0,1] CV sub-scores into ``(score, flags)``.

    Sub-scores (1 = good): ``stability`` (1=still camera), ``exposure`` (1=well-lit),
    ``sharpness`` (1=sharp), ``subject_present`` (1=person in frame).
    """
    flags: list[str] = []
    if subject_present < 0.5:
        flags.append("no_subject")
    if stability < 0.4:
        flags.append("camera_motion")
    if sharpness < 0.3:
        flags.append("blurry")
    if exposure < 0.25:
        flags.append("dark")
    elif exposure > 0.97:
        flags.append("blown_out")
    score = 0.35 * stability + 0.25 * sharpness + 0.20 * exposure + 0.20 * subject_present
    return max(0.0, min(1.0, score)), flags


def combine_quality(cv_quality: float, vlm_quality: float, weight: float = 0.5) -> float:
    """Final 0-1 quality = ``weight*cv + (1-weight)*vlm`` (weight=0.5 is the canonical scorer)."""
    return max(0.0, min(1.0, weight * cv_quality + (1.0 - weight) * vlm_quality))


def score_window(*, exposure: float, sharpness: float, stability: float,
                 subject_present: float, vlm_usable: float, vlm_flags, combine_w: float = 0.5) -> dict:
    """The TIER-1 scorer (Phase 9A): cached Tier-0 raw signals -> derived quality, pure.

    Reads only cached numbers (no decode / embed / VLM call), so re-scoring the whole library
    is the cheap pure path. Mirrors exactly what the ingest pipeline computes inline, so moving
    to the cache is behavior-preserving. Change a weight/threshold here -> bump SCORER_VERSION
    -> re-score in minutes.
    """
    cv_quality, cv_flags = cv_quality_score(stability, exposure, sharpness, subject_present)
    return {
        "cv_quality": cv_quality,
        "vlm_quality": vlm_usable,
        "quality_score": combine_quality(cv_quality, vlm_usable, combine_w),
        "quality_flags": cv_flags + list(vlm_flags or []),
    }


# --- CV signal extraction (local, no API) -----------------------------------
# Frame-diff motion conflates camera motion with subject motion, so the stability threshold
# is deliberately HIGH (trim only extreme transitions - set-down, cut, walk-up - not vigorous
# exercise). The VLM pass is the authoritative camera-handling / transitional classifier.

_SHARP_SCALE = 200.0  # Laplacian variance of a clearly-sharp frame; blurry frames << this
_pose = None
_pose_unavailable = False


def motion_series(local_path: str, fps: float = 5.0, w: int = 256, h: int = 144):
    """Inter-frame motion across a clip via ffmpeg (fast, autorotated): ``(motion, times)``.

    Samples ``fps`` grayscale frames, returns mean abs pixel diff between consecutive frames
    (orientation-invariant). Empty if the clip can't be read.
    """
    import subprocess

    import numpy as np

    raw = subprocess.run(
        ["ffmpeg", "-nostdin", "-i", local_path, "-vf", f"fps={fps},scale={w}:{h},format=gray",
         "-f", "rawvideo", "-"],
        capture_output=True).stdout
    n = len(raw) // (w * h)
    if n < 2:
        return [], []
    frames = np.frombuffer(raw[: n * w * h], dtype=np.uint8).reshape(n, h, w).astype(np.int16)
    motion = [float(np.mean(np.abs(frames[i] - frames[i - 1]))) for i in range(1, n)]
    times = [i / fps for i in range(1, n)]
    return motion, times


def stability_threshold(motion: list[float]) -> float:
    """A clip-relative 'unstable' cutoff - high, so only extreme transitions trim."""
    import numpy as np

    arr = np.asarray(motion, dtype=float)
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med))) or 1.0
    return max(med + 3.0 * mad, med * 2.0, 4.0)


def window_stability(motion: list[float], times: list[float], start: float, end: float,
                     threshold: float) -> float:
    """Normalized stability (1=still) for a window, from its slice of the motion series."""
    import numpy as np

    vals = [m for m, t in zip(motion, times) if start <= t <= end]
    if not vals:
        return 1.0
    avg = float(np.mean(vals))
    return max(0.0, 1.0 - avg / (2.0 * threshold))


def frame_signals(frame_path: str) -> tuple[float, float]:
    """``(exposure, sharpness)`` from a frame: mean luminance [0,1] + normalized blur variance."""
    import cv2
    import numpy as np

    gray = cv2.cvtColor(cv2.imread(frame_path), cv2.COLOR_BGR2GRAY)
    exposure = float(np.mean(gray)) / 255.0
    sharpness = min(1.0, float(cv2.Laplacian(gray, cv2.CV_64F).var()) / _SHARP_SCALE)
    return exposure, sharpness


def subject_present(frame_path: str) -> float:
    """1.0 if a person/pose is detected in the (correctly-oriented) frame, else 0.0.

    Uses the mediapipe Tasks PoseLandmarker. If the model is unavailable, returns 1.0
    (neutral) so the VLM pass - which also classifies no-subject - stays authoritative.
    """
    global _pose, _pose_unavailable
    if _pose_unavailable:
        return 1.0
    try:
        if _pose is None:
            from mediapipe.tasks.python import BaseOptions
            from mediapipe.tasks.python.vision import (
                PoseLandmarker, PoseLandmarkerOptions, RunningMode,
            )
            _pose = PoseLandmarker.create_from_options(PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=settings.pose_model_path),
                running_mode=RunningMode.IMAGE,
            ))
        import mediapipe as mp

        result = _pose.detect(mp.Image.create_from_file(frame_path))
        return 1.0 if result.pose_landmarks else 0.0
    except Exception:
        _pose_unavailable = True
        return 1.0


def moment_cv_signals(frame_path: str, stability: float) -> tuple[float, list[str]]:
    """Per-moment CV quality from a representative frame + the window's stability."""
    exposure, sharpness = frame_signals(frame_path)
    return cv_quality_score(stability, exposure, sharpness, subject_present(frame_path))
