#!/usr/bin/env python3
"""Ingest the staged library into Supabase.

INCREMENTAL by default (Phase 9A): a plain run skips every clip already cached by content_hash
and processes only NEW footage, so adding N clips costs ~N clips of work - this is the clean
"add new footage" path (and what `scripts/watch_library.py` runs on a loop).

Usage:
    python scripts/ingest_library.py --limit 3      # test on a handful first
    python scripts/ingest_library.py                # incremental: only new clips
    python scripts/ingest_library.py --force        # re-extract Tier-0 + rebuild moments
"""
import argparse

from reforge.ingest.pipeline import ingest_library


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest the library into Supabase.")
    parser.add_argument("--prefix", default="library/", help="storage prefix to scan")
    parser.add_argument("--limit", type=int, default=None, help="ingest only the first N clips")
    parser.add_argument("--force", action="store_true", help="re-ingest even if already cataloged")
    args = parser.parse_args()

    results = ingest_library(prefix=args.prefix, limit=args.limit, force=args.force)

    by_action: dict[str, int] = {}
    moments = 0
    for r in results:
        by_action[r["action"]] = by_action.get(r["action"], 0) + 1
        moments += r.get("moments") or 0
    summary = ", ".join(f"{n} {a}" for a, n in sorted(by_action.items()))
    print(f"\nDone: {len(results)} clips ({summary}); {moments} moments written.")


if __name__ == "__main__":
    main()
