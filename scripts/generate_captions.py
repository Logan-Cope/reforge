#!/usr/bin/env python3
"""Backfill ``reels.caption`` for reels that predate caption generation.

One cheap Haiku call per reel, prompt built from the deconstruct brief
(reforge.publish.captions). The render worker also runs this each cycle, so this
script is for one-off catch-ups:

    PYTHONPATH=src python scripts/generate_captions.py            # fill all missing
    PYTHONPATH=src python scripts/generate_captions.py --limit 5  # a small batch
"""
import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Backfill missing reel captions.")
    ap.add_argument("--limit", type=int, default=1000, help="max reels this run")
    args = ap.parse_args()

    from reforge.db.client import get_conn
    from reforge.publish.captions import backfill_captions

    conn = get_conn()
    try:
        n = backfill_captions(conn, limit=args.limit)
        left = conn.execute(
            "SELECT count(*) FROM reels WHERE caption IS NULL").fetchone()[0]
        print(f"captions generated: {n}; still missing: {left}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
