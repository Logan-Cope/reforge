"""Recompute the headline eval numbers from the files in eval_results/ (no API, no DB).

Every number in the README's Evaluation section comes from this script, and
tests/test_eval_results.py pins them, so the README can't drift from the evidence.

Usage:  python scripts/summarize_evals.py
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "eval_results"
METRICS = ("hit@5", "hit@10", "hit@20", "precision@5", "precision@10", "precision@20",
           "recall@20", "mrr")


def retrieval_means(name: str) -> dict[str, float]:
    rows = [r for r in json.loads((RESULTS / name).read_text()) if r.get("run") != "MEAN"]
    return {"beats": len(rows),
            **{m: round(statistics.mean(float(r[m]) for r in rows), 3) for m in METRICS}}


def labeled_set() -> dict[str, int]:
    beats = json.loads((RESULTS / "labeled-beats.json").read_text())
    return {"beats": len(beats), "references": len({b["run"] for b in beats}),
            "pooled": sum(b["pooled"] for b in beats),
            "relevant": sum(b["relevant"] for b in beats)}


def would_post(name: str) -> dict[str, float]:
    rows = json.loads((RESULTS / name).read_text())
    scores = [r["score"] for r in rows]
    return {"reels": len(rows), "mean": round(statistics.mean(scores), 1),
            "median": statistics.median(scores), "min": min(scores), "max": max(scores),
            "would_post": sum(1 for r in rows if r["would_post"])}


def summary() -> dict:
    return {
        "labeled_set": labeled_set(),
        "clip": retrieval_means("2026-07-10-retrieval-labeled-clip.json"),
        "hybrid": retrieval_means("2026-07-10-retrieval-labeled-hybrid.json"),
        "would_post_batch": would_post("2026-07-16-would-post.json"),
    }


if __name__ == "__main__":
    s = summary()
    print(json.dumps(s, indent=2))
    c, h = s["clip"], s["hybrid"]
    print("\n| Metric | CLIP only | Hybrid | Change |\n| --- | --- | --- | --- |")
    for m in METRICS:
        rel = (h[m] - c[m]) / c[m] * 100 if c[m] else 0.0
        print(f"| {m} | {c[m]:.3f} | {h[m]:.3f} | {rel:+.0f}% |")
