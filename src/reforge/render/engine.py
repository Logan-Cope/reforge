"""ffmpeg render engine: an EDL (list of SlotRender) -> finished 9:16 MP4.

Ported from the proven Phase 0 build_reel.py. This piece WORKS; build around it,
don't rebuild it. Each slot is rendered to a normalized segment, segments are
concatenated, the reference audio is muxed, and the persistent text is burned in.
"""
from __future__ import annotations
import os
import subprocess
import glob
from dataclasses import dataclass
from .recipe import TARGET_W, TARGET_H, FPS, BRIGHTNESS_EQ, compute_crop, display_dims


@dataclass
class SlotRender:
    src: str            # absolute path to the source clip (use storage.ensure_local)
    in_point: float     # seconds into the source
    duration: float     # slot duration (from the reference cut grid)
    zoom: float = 1.0
    xfrac: float = 0.5
    yfrac: float = 0.5
    brightness: str = "none"


def _probe_wh(path: str) -> tuple[int, int]:
    """Return the clip's DISPLAY (post-rotation) width, height.

    Vertically-shot clips are coded landscape (e.g. 3840x2160) with a rotate=90/270 tag;
    ffmpeg autorotates on decode (autorotate stays ON), so the crop must be computed
    against the swapped display dims - otherwise a vertical clip is cropped as landscape.
    """
    import json

    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height:stream_tags=rotate:stream_side_data=rotation", "-of", "json", path],
        capture_output=True, text=True).stdout
    stream = json.loads(out)["streams"][0]
    rotation = (stream.get("tags") or {}).get("rotate")
    for side in stream.get("side_data_list") or []:
        if "rotation" in side:
            rotation = side["rotation"]
            break
    return display_dims(int(stream["width"]), int(stream["height"]), rotation)


def _render_segment(slot: SlotRender, out_path: str) -> None:
    w, h = _probe_wh(slot.src)
    cw, ch, x, y = compute_crop(w, h, slot.zoom, slot.xfrac, slot.yfrac)
    vf = f"crop={cw}:{ch}:{x}:{y}"
    eq = BRIGHTNESS_EQ.get(slot.brightness)
    if eq:
        vf += "," + eq
    vf += f",scale={TARGET_W}:{TARGET_H},setsar=1,fps={FPS},format=yuv420p"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-ss", f"{slot.in_point:.3f}", "-i", slot.src,
         "-t", f"{slot.duration:.3f}", "-an", "-vf", vf,
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-y", out_path],
        capture_output=True)


def render_reel(edl: list[SlotRender], audio_path: str, out_path: str,
                text: str = "", tmp_dir: str = "/tmp/reforge_segs",
                font: str = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf") -> str:
    """Render an EDL to a finished reel. Returns out_path."""
    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    for f in glob.glob(f"{tmp_dir}/seg*.mp4"):
        os.remove(f)

    segs = []
    for i, slot in enumerate(edl):
        seg = f"{tmp_dir}/seg{i:02d}.mp4"
        _render_segment(slot, seg)
        segs.append(seg)

    listf = f"{tmp_dir}/list.txt"
    with open(listf, "w") as f:
        f.write("".join(f"file '{s}'\n" for s in segs))
    concat = f"{tmp_dir}/concat.mp4"
    subprocess.run(["ffmpeg", "-nostdin", "-f", "concat", "-safe", "0", "-i", listf,
                    "-c", "copy", "-y", concat], capture_output=True)

    total = sum(s.duration for s in edl)
    vf = (f"drawtext=fontfile={font}:text='{text}':fontcolor=white:fontsize=30:"
          f"x=(w-text_w)/2:y=h*0.44:alpha=0.95:shadowcolor=black:shadowx=1:shadowy=1") if text else None
    cmd = ["ffmpeg", "-nostdin", "-i", concat, "-i", audio_path]
    if vf:
        cmd += ["-vf", vf]
    cmd += ["-map", "0:v", "-map", "1:a", "-t", f"{total:.2f}",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart",
            "-y", out_path]
    subprocess.run(cmd, capture_output=True)
    return out_path
