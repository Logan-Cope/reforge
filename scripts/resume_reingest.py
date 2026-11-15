"""Resume an interrupted --force re-ingest: reprocess only the clips that errored.

The full re-ingest dropped its Supabase connection mid-run (network blip); the failed
clips rolled back (old moments intact) but weren't refreshed. This re-runs ingest_clip
(force) on just the failed keys, classifying benign duplicate-path content-hash collisions
(content already cataloged under another path) apart from real network failures, and writes
any still-failing keys back out for another pass. Idempotent - safe to re-run.
"""
from __future__ import annotations

import sys

from reforge.ingest.pipeline import ingest_clip
from reforge.llm import get_captioner

FAILED = sys.argv[1] if len(sys.argv) > 1 else "scratchpad/failed_keys.txt"
STILL = sys.argv[2] if len(sys.argv) > 2 else "scratchpad/failed_keys_remaining.txt"


def main() -> None:
    with open(FAILED) as fh:
        keys = [ln.strip() for ln in fh if ln.strip()]
    print(f"resuming {len(keys)} failed clips", flush=True)

    vision = get_captioner()
    done = dup = neterr = other = 0
    remaining: list[str] = []
    for i, key in enumerate(keys):
        try:
            r = ingest_clip(key, vision=vision, force=True)
            done += 1
            print(f"[{i+1:>3}/{len(keys)}] reingest {key.split('/')[-1]}  "
                  f"({r.get('moments')} moments)", flush=True)
        except Exception as exc:
            name = type(exc).__name__
            msg = str(exc)
            if "UniqueViolation" in name or "content_hash" in msg:
                dup += 1
                print(f"[{i+1:>3}/{len(keys)}] DUP-SKIP {key.split('/')[-1]} "
                      f"(content already cataloged under another path)", flush=True)
            elif "OperationalError" in name or "resolve host" in msg or "route to host" in msg:
                neterr += 1
                remaining.append(key)
                print(f"[{i+1:>3}/{len(keys)}] NET-FAIL {key.split('/')[-1]}", flush=True)
            else:
                other += 1
                remaining.append(key)
                print(f"[{i+1:>3}/{len(keys)}] ERROR {key.split('/')[-1]}: {name}: {msg}", flush=True)

    with open(STILL, "w") as fh:
        fh.write("\n".join(remaining) + ("\n" if remaining else ""))
    print(f"\nDone: {done} reingested, {dup} benign dup-skips, {neterr} net-fail, "
          f"{other} other; {len(remaining)} still need a pass (-> {STILL})", flush=True)


if __name__ == "__main__":
    main()
