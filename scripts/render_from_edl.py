#!/usr/bin/env python3
"""Example entrypoint: render a finished reel from a hand-written EDL.
Mirrors the Phase 0 proof so the render path is runnable on day one.
Usage: python scripts/render_from_edl.py
"""
from reforge.render.engine import SlotRender, render_reel

# Example EDL (replace with matcher output). in_point/duration in seconds.
EDL = [
    SlotRender(src="library/example.mp4", in_point=2.0, duration=0.87),
]

if __name__ == "__main__":
    out = render_reel(EDL, audio_path="reference/audio.m4a",
                      out_path="output/draft.mp4", text="D I S C I P L I N E")
    print("wrote", out)
