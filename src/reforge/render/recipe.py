"""The LOCKED v1 framing recipe. Do not regress to v2's mistakes.

Lesson (v1 vs v2, 2026-06-27): v1's full-frame center crop looked great; v2 added
aggressive punch-in zoom + shifted "peak action" windows and clipped the subject to a
shoulder. RULE: keep the WHOLE BODY in frame. Default zoom = 1.0 (no punch-in).
Only deviate from center framing for a clearly off-center subject, and never below
~0.85 zoom without a strong reason.
"""
from __future__ import annotations

TARGET_W, TARGET_H, FPS = 1080, 1920, 30
ASPECT = 9 / 16

# brightness lift for genuinely dark night clips (the one good idea from v2)
BRIGHTNESS_EQ = {
    "none": None,
    "mild": "eq=brightness=0.05:contrast=1.06:gamma=1.12:saturation=1.05",
    "strong": "eq=brightness=0.12:contrast=1.05:gamma=1.45:saturation=1.10",
}

DEFAULTS = dict(zoom=1.0, xfrac=0.5, yfrac=0.5, brightness="none")


def display_dims(coded_w: int, coded_h: int, rotation) -> tuple[int, int]:
    """Post-rotation (display) dimensions for a clip.

    Vertically-shot clips are stored landscape-coded with a rotate=90/270 tag; ffmpeg
    autorotates on decode, so the real frame is the swapped dims. ``compute_crop`` must
    run against these display dims, not the coded ones. ``rotation`` may be int/float/
    str/None (degrees); only |rotation| of 90 or 270 swaps.
    """
    r = abs(int(float(rotation))) % 360 if rotation not in (None, "") else 0
    return (coded_h, coded_w) if r in (90, 270) else (coded_w, coded_h)


def compute_crop(w: int, h: int, zoom: float = 1.0, xfrac: float = 0.5, yfrac: float = 0.5):
    """Return (cw, ch, x, y) for a centered 9:16 crop of a w x h source.
    zoom<1.0 punches in (discouraged). xfrac/yfrac position the window (0..1)."""
    landscape = (w / h) > ASPECT
    cw0 = h * ASPECT if landscape else w
    ch0 = h if landscape else w / ASPECT
    cw = int(round(cw0 * zoom))
    cw -= cw % 2
    ch = int(round(ch0 * zoom))
    ch -= ch % 2
    x = int(round((w - cw) * xfrac))
    y = int(round((h - ch) * yfrac))
    return cw, ch, x, y
