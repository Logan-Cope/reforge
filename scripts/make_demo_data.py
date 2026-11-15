"""Build the synthetic data set the public read-only demo serves.

Writes ``src/reforge/demo/fixtures.json`` plus placeholder media under
``src/reforge/demo/media/`` (labeled 9:16 frames and short preview reels). Nothing
here comes from a real library: every moment, caption, score and judge reason is
written for the demo, and every frame is drawn from a gradient and a text label.

Usage (needs Pillow and ffmpeg):  python scripts/make_demo_data.py
Deterministic: the same script produces the same fixtures.
"""
from __future__ import annotations

import json
import random
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
DEMO = ROOT / "src" / "reforge" / "demo"
THUMBS = DEMO / "media" / "thumbs"
OUTPUT = DEMO / "media" / "output"
W, H = 360, 640

# location -> (top color, bottom color)
PALETTE = {
    "track": ((18, 22, 38), (196, 92, 48)),
    "gym": ((12, 12, 14), (70, 74, 82)),
    "stairs": ((24, 18, 30), (150, 70, 90)),
    "waterfront": ((10, 24, 40), (60, 120, 150)),
    "rooftop": ((20, 20, 26), (210, 150, 90)),
}

MOMENTS = [
    ("track", "lacing up on the track at first light, low angle"),
    ("track", "full-body sprint toward camera, golden hour"),
    ("track", "wide drone pass over the lanes, subject small"),
    ("track", "hands on knees after the last rep, side light"),
    ("track", "slow jog away from camera into the haze"),
    ("track", "explosive start from a crouch, side angle"),
    ("gym", "heavy deadlift lockout, single overhead light"),
    ("gym", "chalking hands, close crop, shallow focus"),
    ("gym", "pull-up set, back to camera, silhouette"),
    ("gym", "mirror shot, subject only a reflection"),
    ("gym", "dumbbell press on a bench, dark and grainy"),
    ("gym", "rope slams, motion blur at the peak"),
    ("stairs", "stair sprint at dusk, full body in frame"),
    ("stairs", "walking down the steps, breathing hard"),
    ("stairs", "box jumps on the landing, low angle"),
    ("stairs", "stair sprint, near-duplicate of the dusk take"),
    ("waterfront", "cool-down walk along the rail, blue hour"),
    ("waterfront", "stretching against the railing, wide"),
    ("waterfront", "push-ups on the boardwalk, side profile"),
    ("waterfront", "adjusting the camera on a tripod"),
    ("rooftop", "shadow boxing at sunset, skyline behind"),
    ("rooftop", "sitting on the ledge, reflective close-up"),
    ("rooftop", "skipping rope, footwork close-up"),
    ("rooftop", "skyline wide, subject out of frame"),
]

REFERENCES = ["reference/night-run.mp4", "reference/morning-discipline.mp4",
              "reference/city-lifts.mp4"]

# Each beat: (reference ask, wanted location, judge reason)
# Each beat: (reference ask, judge's pick, runner-ups, judge reason). Picks and
# runner-ups are indexes into MOMENTS.
BEATS = {
    "reference/night-run.mp4": [
        ("opener: quiet setup before the work", 0, [4, 17],
         "calm setup at first light opens the arc without a hard cut"),
        ("build: first real effort", 5, [1, 2],
         "explosive start, full body in frame, effort reads at a glance"),
        ("build: climbing intensity", 12, [14, 15],
         "stair sprint matches the climb and keeps the warm palette"),
        ("peak: explosive rep", 10, [11, 6],
         "heaviest press in the pool for the peak"),
        ("peak: second hit", 6, [8, 11],
         "deadlift lockout keeps the peak going from a new angle"),
        ("release: breathing after", 3, [13, 16],
         "hands on knees: earned rest, subject still clearly the focus"),
        ("reflective: alone with it", 21, [23, 9],
         "quiet close-up on the ledge lands the reflective beat"),
        ("close: walk away", 4, [16, 13],
         "jogging into the haze mirrors the opener"),
    ],
    "reference/morning-discipline.mp4": [
        ("opener: before sunrise", 0, [17, 4], "first light sets the tone of the reference"),
        ("build: steady work", 18, [17, 16], "steady push-up rhythm, not a peak yet"),
        ("build: more effort", 12, [14, 15], "full-body stair work at the right energy"),
        ("peak: hardest rep", 6, [11, 8], "the heaviest lift in the pool"),
        ("release: recover", 16, [13, 3], "a calm walk along the water to recover"),
        ("close: look ahead", 21, [20, 23], "a still close that matches the reference end card"),
    ],
    "reference/city-lifts.mp4": [
        ("opener: chalk up", 7, [10, 9], "tight detail shot opens like the reference"),
        ("build: first set", 8, [6, 10], "pull-up silhouette, readable rep in clean light"),
        ("build: outside", 20, [22, 23], "moves to the roof on the same beat as the reference"),
        ("peak: footwork", 22, [20, 14], "fast footwork for the fastest cut"),
        ("peak: heavy pull", 6, [8, 11], "the lockout lands on the drop"),
        ("release: breathe", 13, [3, 16], "a breath on the steps between efforts"),
        ("close: skyline", 21, [23, 20], "ends above the city like the reference"),
    ],
}


def font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.load_default(size=size)


def wrap(draw: ImageDraw.ImageDraw, text: str, f, width: int) -> list[str]:
    lines, cur = [], ""
    for word in text.split():
        trial = f"{cur} {word}".strip()
        if draw.textlength(trial, font=f) <= width:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    return lines + [cur] if cur else lines


def draw_frame(path: Path, location: str, text: str, moment_id: int, seed: int) -> None:
    top, bottom = PALETTE[location]
    img = Image.new("RGB", (W, H))
    px = img.load()
    for y in range(H):
        t = y / (H - 1)
        row = tuple(int(top[i] + (bottom[i] - top[i]) * t) for i in range(3))
        for x in range(W):
            px[x, y] = row
    d = ImageDraw.Draw(img)
    rng = random.Random(seed)
    # a soft horizon and a sun so frames don't read as flat swatches
    hy = rng.randint(330, 450)
    d.rectangle([0, hy, W, H], fill=tuple(max(0, c - 30) for c in bottom))
    cx, r = rng.randint(70, 290), rng.randint(26, 54)
    d.ellipse([cx - r, hy - r - 40, cx + r, hy + r - 40], fill=(255, 220, 170))
    d.text((18, 18), "SYNTHETIC FRAME", font=font(13), fill=(255, 255, 255))
    d.text((18, 38), f"{location} · moment {moment_id}", font=font(13), fill=(230, 230, 230))
    f = font(20)
    y = 70  # upper third: the app's own overlays sit on the bottom of each card
    for line in wrap(d, text, f, W - 36):
        d.text((18, y), line, font=f, fill=(255, 255, 255))
        y += 26
    img.save(path, "JPEG", quality=80)


def render_reel(frames: list[tuple[Path, float]], out: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        lst = Path(tmp) / "list.txt"
        lines = []
        for p, dur in frames:
            lines += [f"file '{p}'", f"duration {dur:.2f}"]
        lines.append(f"file '{frames[-1][0]}'")
        lst.write_text("\n".join(lines) + "\n")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                        "-i", str(lst), "-vf", "fps=24,format=yuv420p",
                        "-c:v", "libx264", "-preset", "slow", "-crf", "30",
                        "-movflags", "+faststart", str(out)], check=True)


def main() -> None:
    rng = random.Random(7)
    for d in (THUMBS, OUTPUT):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)

    moments = []
    for i, (loc, desc) in enumerate(MOMENTS):
        mid = 100 + i
        key = f"thumbs/demo/m{mid}.jpg"
        (THUMBS / "demo").mkdir(exist_ok=True)
        draw_frame(THUMBS / "demo" / f"m{mid}.jpg", loc, desc, mid, seed=mid)
        weak = any(w in desc for w in ("grainy", "out of frame", "reflection", "small", "tripod"))
        q = round(rng.uniform(0.28, 0.45) if weak else rng.uniform(0.55, 0.9), 2)
        moments.append({"id": mid, "location": loc, "description": desc,
                        "thumbnail_key": key, "quality_score": q})
    # reel id -> (reference, created, grade, verdicts by beat idx, extras)
    plan = [
        (15, "reference/city-lifts.mp4", "Nov 13", None, {}, {}),
        (14, "reference/night-run.mp4", "Nov 12", None,
         {0: ("keep", None), 1: ("keep", None), 2: ("swapped", "needs more explosive energy"),
          3: ("flagged", "too dark")}, {"cut_dirty": True}),
        (13, "reference/morning-discipline.mp4", "Nov 10", "would_test",
         {i: ("keep", None) for i in range(6)} | {4: ("weak", "a bit slow")}, {}),
        (12, "reference/city-lifts.mp4", "Nov 07", "post_worthy",
         {i: ("keep", None) for i in range(7)},
         {"published": True, "posted": "Nov 08", "is_trial": True}),
        (11, "reference/night-run.mp4", "Nov 05", "not_it", {}, {}),
    ]

    reels, slot_id = [], 500
    for reel_id, ref, created, grade, verdicts, extra in plan:
        t, slots, frames = 0.0, [], []
        for idx, (ask, pick_i, runner_is, reason) in enumerate(BEATS[ref]):
            pick = moments[pick_i]
            runners = [moments[i] for i in runner_is]
            dur = round(rng.uniform(1.1, 1.9), 2)
            verdict, note = verdicts.get(idx, (None, None))
            shown = runners[0] if verdict == "swapped" else pick
            cands = [{"moment_id": m["id"], "clip_key": None, "thumbnail_key": m["thumbnail_key"],
                      "score": round(0.9 - 0.08 * n - rng.uniform(0, 0.05), 2),
                      "reason": f"runner-up: {m['description']}", "in_point": 0.4}
                     for n, m in enumerate(runners)]
            slots.append({"id": slot_id, "idx": idx, "start_s": round(t, 2),
                          "end_s": round(t + dur, 2), "ref_description": ask, "reason": reason,
                          "human_verdict": verdict, "human_note": note,
                          "chosen_moment_id": pick["id"],
                          "swapped_moment_id": shown["id"] if verdict == "swapped" else None,
                          "candidates": cands, "params": {"in_point": 0.4, "zoom": 1.0},
                          "thumbnail_key": shown["thumbnail_key"],
                          "description": shown["description"],
                          "quality_score": shown["quality_score"], "clip_key": None,
                          "moment_start": 0.0, "moment_end": 1.75})
            frames.append((THUMBS / shown["thumbnail_key"].removeprefix("thumbs/"), dur))
            slot_id += 1
            t += dur
        output_key = f"output/reel-{reel_id}.mp4"
        render_reel(frames, OUTPUT / f"reel-{reel_id}.mp4")
        reels.append({"id": reel_id, "ref": ref, "created": created, "grade": grade,
                      "approved": grade in ("post_worthy", "would_test"),
                      "output_key": output_key,
                      "caption": "Discipline is quiet. Synthetic demo caption.",
                      "slots": slots, **extra})

    data = {
        "_note": "Synthetic demo data generated by scripts/make_demo_data.py.",
        "references": REFERENCES,
        "reels": reels,
        "runs": [
            {"id": 31, "reference_key": "reference/city-lifts.mp4", "status": "done",
             "created": "Nov 13 21:02", "started": "Nov 13 21:02", "finished": "Nov 13 21:09",
             "reel_id": 15, "error": None},
            {"id": 30, "reference_key": "reference/night-run.mp4", "status": "done",
             "created": "Nov 12 19:40", "started": "Nov 12 19:40", "finished": "Nov 12 19:48",
             "reel_id": 14, "error": None},
            {"id": 29, "reference_key": "reference/night-run.mp4", "status": "error",
             "created": "Nov 12 19:31", "started": "Nov 12 19:31", "finished": "Nov 12 19:31",
             "reel_id": None, "error": "reference has burned-in captions on 6 of 8 beats"},
        ],
        "moments": [m for m in moments if 0.35 <= m["quality_score"] <= 0.75][:6]
                   or moments[:6],
        "scheduled": {
            "13": {"id": 9, "status": "scheduled", "scheduled_at": "2026-11-18T14:00:00+00:00",
                   "as_trial": True, "provider": None, "error": None, "published_at": None},
            "12": {"id": 8, "status": "published", "scheduled_at": "2026-11-08T14:00:00+00:00",
                   "as_trial": True, "provider": "dry_run", "error": None,
                   "published_at": "2026-11-08T14:00:30+00:00"},
        },
    }
    (DEMO / "fixtures.json").write_text(json.dumps(data, indent=1) + "\n")
    print(f"wrote {len(reels)} reels, {sum(len(r['slots']) for r in reels)} beats, "
          f"{len(moments)} frames -> {DEMO.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
