#!/usr/bin/env python3
"""Build the synthetic labeled retrieval-eval set (pooling + Haiku relevance judge).

For each reference reel: deconstruct once (cut grid + creative read), then for EVERY
shot pool the candidate union of both retrievers (clip + hybrid, top-30 each, deduped
by moment id) and have a text-only Haiku judge mark each pooled candidate relevant /
not-relevant to the beat's ask. Writes ``eval_results/labeled-beats.json`` - explicit
``moment_id`` labels scored later by ``run_eval.py --labels``.

Usage:
    python scripts/build_eval_set.py [--out eval_results/labeled-beats.json]
"""
import argparse
import json
import os
import re

REFERENCES = ["reference/DISCIPLINE.mp4", "reference/grace.mp4", "reference/Bible-gym.mp4"]
LABEL_MODEL = "claude-haiku-4-5"


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the synthetic labeled retrieval-eval set.")
    ap.add_argument("--out", default="eval_results/labeled-beats.json",
                    help="output JSON path (default eval_results/labeled-beats.json)")
    ap.add_argument("--work-dir", default="data/eval_work",
                    help="deconstruct working directory (default data/eval_work)")
    args = ap.parse_args()

    from reforge.deconstruct.reference import deconstruct
    from reforge.eval.synthetic import label_pool, labeled_beat_record, make_client, pool_candidates
    from reforge.llm import get_captioner
    from reforge.llm.cost import estimate_cost

    vision = get_captioner()
    client = make_client()
    label_usage: dict = {}
    records: list[dict] = []

    for ref in REFERENCES:
        run = os.path.splitext(os.path.basename(ref))[0] + "-synth"
        try:
            name = re.sub(r"\W+", "_", os.path.splitext(os.path.basename(ref))[0])
            decon = deconstruct(ref, vision, os.path.join(args.work_dir, name))
        except Exception as exc:  # noqa: BLE001 - skip a broken reference, keep the rest
            print(f"[skip] {ref}: deconstruct failed: {exc}")
            continue
        print(f"[{run}] {len(decon.shots)} shots")
        for shot in decon.shots:
            ref_frame = decon.frames_per_shot[shot.idx][0]
            candidates = pool_candidates(shot, ref_frame, decon.brief)
            labels = label_pool(client, shot, decon.brief, candidates,
                                model=LABEL_MODEL, usage=label_usage)
            record = labeled_beat_record(run, shot, decon.brief, labels)
            records.append(record)
            print(f"  beat {shot.idx:2d}: pooled={record['pooled']:3d} "
                  f"relevant={record['relevant']:3d}  ({shot.query})")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=2)

    total_relevant = sum(r["relevant"] for r in records)
    print(f"\nwrote {args.out}: {len(records)} labeled beats, "
          f"{sum(r['pooled'] for r in records)} pooled candidates, "
          f"{total_relevant} marked relevant")
    decon_cost = vision.cost_summary()
    label_cost = estimate_cost({LABEL_MODEL: label_usage}) if label_usage else {"total_usd": 0.0}
    print(f"cost: deconstruct ${decon_cost['total_usd']:.4f} "
          f"({decon_cost['total_calls']} calls) + labeling ${label_cost['total_usd']:.4f} "
          f"({label_usage.get('calls', 0)} calls) = "
          f"${decon_cost['total_usd'] + label_cost['total_usd']:.4f}")


if __name__ == "__main__":
    main()
