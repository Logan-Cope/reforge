"""Watched-folder / scheduled auto-ingest (Phase 9A).

Drop clips into the library folder and they ingest unattended. INCREMENTAL by content_hash:
existing clips skip instantly (Tier-0 cache hit), so a pass costs only the genuinely new clips.

  python scripts/watch_library.py --interval 60   # poll every 60s (long-running watcher)
  python scripts/watch_library.py --once          # one incremental pass (run from cron/launchd)

This is just the normal `ingest_library` (no --force) on a loop - the clean, documented
"add new footage" path. Nothing is re-decoded or re-embedded for clips already cached.
"""
from __future__ import annotations

import argparse
import time

from reforge.ingest.pipeline import ingest_library


def _pass(prefix: str) -> int:
    results = ingest_library(prefix=prefix)  # incremental: no force
    new = [r for r in results if r.get("action") in ("fresh", "reingest")]
    if new:
        moments = sum(r.get("moments") or 0 for r in new)
        print(f"[watch] ingested {len(new)} new clip(s), {moments} moments", flush=True)
    return len(new)


def main() -> None:
    ap = argparse.ArgumentParser(description="Watched-folder auto-ingest (incremental).")
    ap.add_argument("--prefix", default="library/", help="storage prefix to scan")
    ap.add_argument("--interval", type=int, default=60, help="poll seconds (ignored with --once)")
    ap.add_argument("--once", action="store_true", help="single pass then exit (for cron/launchd)")
    args = ap.parse_args()

    if args.once:
        _pass(args.prefix)
        return

    print(f"[watch] polling {args.prefix} every {args.interval}s (Ctrl-C to stop)", flush=True)
    while True:
        try:
            _pass(args.prefix)
        except Exception as exc:  # a transient error must not kill the watcher
            print(f"[watch] error: {type(exc).__name__}: {exc}", flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
