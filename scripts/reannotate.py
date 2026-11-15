#!/usr/bin/env python3
"""Re-annotate the already-cataloged library with the identity-reference captioner.

Refreshes `is_logan` + the inherited annotation on every cataloged clip's moments;
keeps the mechanical windows, thumbnails, and CLIP embeddings (no re-embedding).

Usage:
    python scripts/reannotate.py            # all cataloged clips
    python scripts/reannotate.py --limit 3  # first N
"""
import argparse

from reforge.ingest.pipeline import reannotate_library


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-annotate the cataloged library.")
    parser.add_argument("--prefix", default="library/", help="storage prefix to scan")
    parser.add_argument("--limit", type=int, default=None, help="re-annotate only the first N")
    args = parser.parse_args()

    results = reannotate_library(prefix=args.prefix, limit=args.limit)

    done = [r for r in results if r["action"] == "reannotated"]
    missing = [r for r in results if r["action"] == "missing"]
    n_true = sum(1 for r in done if r.get("is_logan"))
    print(f"\nDone: re-annotated {len(done)} clips ({n_true} is_logan=true / "
          f"{len(done) - n_true} false); {len(missing)} not cataloged (skipped).")


if __name__ == "__main__":
    main()
