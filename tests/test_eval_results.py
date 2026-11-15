"""The README's headline eval numbers, recomputed from eval_results/ and pinned.

If an eval file changes, this fails until the README table is updated to match.
"""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("summarize_evals",
                                              ROOT / "scripts" / "summarize_evals.py")
summarize = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summarize)


def test_labeled_set_shape():
    assert summarize.labeled_set() == {"beats": 33, "references": 3,
                                       "pooled": 1459, "relevant": 893}


def test_retrieval_arms_match_the_readme():
    s = summarize.summary()
    clip, hybrid = s["clip"], s["hybrid"]
    assert clip["beats"] == hybrid["beats"] == 33
    assert (clip["hit@5"], clip["precision@5"], clip["recall@20"], clip["mrr"]) == \
        (0.848, 0.497, 0.342, 0.647)
    assert (hybrid["hit@5"], hybrid["precision@5"], hybrid["recall@20"], hybrid["mrr"]) == \
        (0.97, 0.776, 0.542, 0.876)


def test_hybrid_beats_clip_on_every_metric():
    s = summarize.summary()
    for m in summarize.METRICS:
        assert s["hybrid"][m] >= s["clip"][m], m


def test_would_post_batch_matches_the_readme():
    assert summarize.would_post("2026-07-16-would-post.json") == {
        "reels": 15, "mean": 60.7, "median": 61, "min": 48, "max": 88, "would_post": 1}


def test_mean_rows_agree_with_per_beat_rows():
    """The MEAN row the harness wrote must equal the mean of its own per-beat rows."""
    import json
    for name in ("2026-07-10-retrieval-labeled-clip.json",
                 "2026-07-10-retrieval-labeled-hybrid.json"):
        rows = json.loads((ROOT / "eval_results" / name).read_text())
        mean_row = next(r for r in rows if r.get("run") == "MEAN")
        ours = summarize.retrieval_means(name)
        for m in ("precision@5", "mrr", "recall@20"):
            assert round(float(mean_row[m]), 3) == ours[m], (name, m)
