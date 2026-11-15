"""Re-score the library from the Tier-0 cache - Phase 9A's "minutes, not hours" path.

  python scripts/rescore.py --scorer-version score-2 --combine-w 0.6   # recompute + promote
  python scripts/rescore.py --scorer-version score-1 --promote-only     # roll back, no recompute

Reads only cached raw signals (cache_cv_signals + cache_vlm_assessments); never decodes or
re-embeds. `--no-promote` writes the versioned scores without touching live retrieval (A/B).
"""
import argparse
import time

from reforge.ingest.rescore import promote_scorer, rescore_library


def main() -> None:
    ap = argparse.ArgumentParser(description="Tier-1 re-score (no decode / embed / VLM).")
    ap.add_argument("--scorer-version", required=True)
    ap.add_argument("--combine-w", type=float, default=0.5,
                    help="quality = combine_w*cv + (1-combine_w)*vlm  (0.5 = canonical)")
    ap.add_argument("--no-promote", action="store_true", help="write versioned scores only (A/B)")
    ap.add_argument("--promote-only", action="store_true",
                    help="activate an already-computed scorer version (no recompute)")
    args = ap.parse_args()

    t0 = time.monotonic()
    if args.promote_only:
        result = promote_scorer(args.scorer_version)
    else:
        result = rescore_library(args.scorer_version, combine_w=args.combine_w,
                                 promote=not args.no_promote)
    print(f"{result}   ({time.monotonic() - t0:.1f}s)")


if __name__ == "__main__":
    main()
