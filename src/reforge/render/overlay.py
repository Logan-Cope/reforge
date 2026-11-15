"""Burn persistent overlay text onto a rendered reel without ffmpeg's `drawtext`.

`render.engine.render_reel` burns text via the `drawtext` filter, which needs an
ffmpeg built with libfreetype. Some environments (e.g. this Homebrew build) ship ffmpeg
without it. `burn_text` is a freetype-free alternative: PIL renders the text to a
transparent PNG and ffmpeg's core `overlay` filter composites it. The engine is unchanged.
"""
from __future__ import annotations

import os
import subprocess

from .recipe import TARGET_H, TARGET_W


def burn_text(in_path: str, text: str, out_path: str, *,
              font: str, fontsize: int = 30, y_frac: float = 0.44) -> str:
    """Composite centered ``text`` onto ``in_path`` (audio copied through); return out_path."""
    if not text:
        # nothing to burn - just pass the file through
        import shutil

        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        shutil.copy2(in_path, out_path)
        return out_path

    from PIL import Image, ImageDraw, ImageFont

    canvas = Image.new("RGBA", (TARGET_W, TARGET_H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    pil_font = ImageFont.truetype(font, fontsize)
    left, top, right, bottom = draw.textbbox((0, 0), text, font=pil_font)
    x = (TARGET_W - (right - left)) // 2 - left
    y = int(TARGET_H * y_frac)
    draw.text((x + 1, y + 1), text, font=pil_font, fill=(0, 0, 0, 242))   # shadow
    draw.text((x, y), text, font=pil_font, fill=(255, 255, 255, 242))     # white

    png = os.path.splitext(out_path)[0] + "_overlay.png"
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    canvas.save(png)

    subprocess.run(
        ["ffmpeg", "-nostdin", "-i", in_path, "-i", png,
         "-filter_complex", "[0:v][1:v]overlay=0:0",
         "-c:a", "copy", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
         "-movflags", "+faststart", "-y", out_path],
        capture_output=True, check=True)
    os.remove(png)
    return out_path
