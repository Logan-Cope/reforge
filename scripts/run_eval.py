#!/usr/bin/env python3
"""Run the Reforge eval harness: retrieval metrics and/or the would-post judge.

--retrieval scores `match.retrieve` against the labeled taste-log beats (hit@k /
precision@k / recall@k / mrr; deconstructs each referenced reel once - see the
harness docstring caveat: rows are directional, not exact). --would-post judges
rendered reels against the gold references with an independent Opus judge.
Both write dated JSON + markdown into eval_results/.

--labels scores retrieval against a synthetic labeled set (explicit moment ids from
scripts/build_eval_set.py) instead of the taste log - with --retriever both, this is
the clip-vs-hybrid A/B on 30+ beats.

Usage:
    python scripts/run_eval.py --retrieval
    python scripts/run_eval.py --labels eval_results/labeled-beats.json --retriever both
    python scripts/run_eval.py --would-post output/DISCIPLINE_recreation_phase36.mp4
    python scripts/run_eval.py --retrieval --would-post output/a.mp4 output/b.mp4
"""
import argparse
import glob
import os


def main() -> None:
    ap = argparse.ArgumentParser(description="Reforge eval: retrieval metrics + would-post judge.")
    ap.add_argument("--retrieval", action="store_true",
                    help="score retrieval against the labeled taste-log beats")
    ap.add_argument("--retriever", choices=("clip", "hybrid", "both"), default=None,
                    help="retriever to score (Phase 9B A/B); 'both' runs clip AND hybrid; "
                         "default = the RETRIEVER env setting")
    ap.add_argument("--labels", metavar="PATH", default=None,
                    help="synthetic labeled-beats JSON (scripts/build_eval_set.py); "
                         "scores retrieval against explicit moment-id labels")
    ap.add_argument("--would-post", nargs="+", metavar="PATH", default=None,
                    help="rendered reel(s) to judge against the gold references")
    args = ap.parse_args()
    if not args.retrieval and not args.would_post and not args.labels:
        ap.error("nothing to run: pass --retrieval, --labels PATH, "
                 "and/or --would-post PATH [PATH ...]")

    from reforge.eval.harness import (
        _table_columns,
        run_retrieval_eval,
        run_retrieval_eval_from_labels,
        save_results,
    )
    from reforge.eval.metrics import format_results_table

    if args.retrieval:
        from reforge.config import settings

        choice = args.retriever or settings.retriever
        for mode in (("clip", "hybrid") if choice == "both" else (choice,)):
            print(f"[retrieval:{mode}] deconstruct references + score labeled beats")
            rows = run_retrieval_eval(retriever=mode)
            print(format_results_table(rows, _table_columns(rows)))
            json_path, md_path = save_results(rows, f"retrieval-{mode}")
            print(f"[retrieval:{mode}] wrote {json_path} and {md_path}")

    if args.labels:
        from reforge.config import settings

        choice = args.retriever or settings.retriever
        decon_cache: dict = {}  # frames are deterministic - share across both A/B arms
        for mode in (("clip", "hybrid") if choice == "both" else (choice,)):
            print(f"[labels:{mode}] score {args.labels} (moment-id relevance labels)")
            rows = run_retrieval_eval_from_labels(args.labels, retriever=mode,
                                                  decon_by_ref=decon_cache)
            print(format_results_table(rows, _table_columns(rows)))
            summary = rows[-1]
            print(f"[labels:{mode}] MEAN over {summary['beat']} beats: "
                  + " ".join(f"{k}={summary[k]:.3f}"
                             for k in ("hit@5", "hit@10", "hit@20", "mrr")))
            json_path, md_path = save_results(rows, f"retrieval-labeled-{mode}")
            print(f"[labels:{mode}] wrote {json_path} and {md_path}")

    if args.would_post:
        from reforge.config import settings
        from reforge.eval.would_post import judge_render

        gold_paths = sorted(glob.glob(os.path.join(settings.taste_dir, "gold-references", "*.mp4")))
        if not gold_paths:
            ap.error(f"no gold references found in {settings.taste_dir}/gold-references/*.mp4")
        print(f"[would-post] judging against {len(gold_paths)} gold reference(s)")
        rows = []
        for path in args.would_post:
            verdict = judge_render(path, gold_paths)
            print(f"  {os.path.basename(path)}: score={verdict.score} "
                  f"would_post={verdict.would_post} weak_beats={verdict.weak_beats}")
            print(f"    notes: {verdict.notes}")
            rows.append({"render": path, "score": verdict.score, "would_post": verdict.would_post,
                         "weak_beats": verdict.weak_beats, "notes": verdict.notes})
        json_path, md_path = save_results(rows, "would-post")
        print(f"[would-post] wrote {json_path} and {md_path}")


if __name__ == "__main__":
    main()
