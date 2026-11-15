"""Unit tests for the pure retrieval-eval core in eval/harness.py (no API, no DB)."""
import json
import os
from types import SimpleNamespace

import pytest

from reforge.eval.dataset import LabeledBeat
from reforge.eval.harness import (
    REFERENCE_MAP,
    reference_for_run,
    retrieval_rows,
    save_results,
)

# --- reference_for_run --------------------------------------------------------

def test_reference_for_run_prefix_match():
    assert reference_for_run("DISCIPLINE_judge_v1") == "reference/DISCIPLINE.mp4"
    assert reference_for_run("grace_judge_v3") == "reference/grace.mp4"
    assert reference_for_run("Bible-gym_judge_v1") == "reference/Bible-gym.mp4"


def test_reference_for_run_is_case_insensitive():
    assert reference_for_run("discipline_judge_v9") == "reference/DISCIPLINE.mp4"
    assert reference_for_run("GRACE_x") == "reference/grace.mp4"


def test_reference_for_run_no_match_returns_none():
    assert reference_for_run("unknown_run") is None
    assert reference_for_run("") is None


def test_reference_for_run_longest_prefix_wins():
    mapping = {"a": "short.mp4", "ab": "long.mp4"}
    assert reference_for_run("abc_judge", mapping) == "long.mp4"


_ROOT = os.path.join(os.path.dirname(__file__), "..")


# Reference reels are other creators' videos, so they are gitignored and only exist on a
# machine that has run acquisition; a fresh clone skips this check.
@pytest.mark.skipif(not os.path.isdir(os.path.join(_ROOT, "reference")),
                    reason="reference/ reels are not in the repo")
def test_reference_map_files_exist():
    root = _ROOT
    for path in REFERENCE_MAP.values():
        assert os.path.isfile(os.path.join(root, path)), path


# --- retrieval_rows -----------------------------------------------------------

KEYS = [
    "Shorts/3/Jogging/dji_184410.MP4",
    "Shorts/3/Jogging/dji_184442.MP4",
    "Shorts/3/Gym/C1402.MP4",
]


def _decon(n_shots: int) -> SimpleNamespace:
    shots = [SimpleNamespace(idx=i, query=f"q{i}") for i in range(n_shots)]
    frames = [[f"/frames/shot{i:02d}_0.jpg", f"/frames/shot{i:02d}_1.jpg"] for i in range(n_shots)]
    return SimpleNamespace(shots=shots, frames_per_shot=frames)


def _retrieve_returning(keys: list[str]):
    def retrieve_fn(shot, ref_frame, k):
        assert ref_frame.endswith("_0.jpg")  # representative frame is [beat][0]
        return [SimpleNamespace(clip_key=key) for key in keys[:k]]
    return retrieve_fn


def test_retrieval_rows_hit_in_top_5():
    beats = [LabeledBeat(run="DISCIPLINE_judge_v1", beat=1, good=["Jogging/dji_184410"],
                         bad=["Jogging/dji_184442"])]
    decon_by_ref = {"reference/DISCIPLINE.mp4": _decon(3)}
    retrieved = ["Shorts/3/Gym/C1402.MP4", "Shorts/3/Jogging/dji_184410.MP4"]
    (row,) = retrieval_rows(beats, decon_by_ref, _retrieve_returning(retrieved), lambda: KEYS)
    assert row["run"] == "DISCIPLINE_judge_v1" and row["beat"] == 1
    assert row["hit@5"] is True
    assert row["mrr"] == 0.5                      # good key at rank 2
    assert row["precision@5"] == 1 / 5
    assert row["recall@5"] == 1.0
    assert row["resolved_fragments"] == 2         # good + bad both resolved
    assert row["unresolved_fragments"] == 0
    assert "good_not_retrieved" not in row


def test_retrieval_rows_miss():
    beats = [LabeledBeat(run="grace_judge_v1", beat=0, good=["Jogging/dji_184410"])]
    decon_by_ref = {"reference/grace.mp4": _decon(2)}
    (row,) = retrieval_rows(beats, decon_by_ref,
                            _retrieve_returning(["Shorts/3/Gym/C1402.MP4"]), lambda: KEYS)
    assert row["hit@5"] is False and row["mrr"] == 0.0 and row["recall@5"] == 0.0
    # the resolved good key never appeared anywhere in the retrieved list -> noted
    assert row["good_not_retrieved"] == ["Shorts/3/Jogging/dji_184410.MP4"]


def test_retrieval_rows_out_of_range_beat_emits_warning_row():
    beats = [LabeledBeat(run="DISCIPLINE_judge_v1", beat=15, good=["Jogging/dji_184410"])]
    decon_by_ref = {"reference/DISCIPLINE.mp4": _decon(3)}

    def retrieve_fn(shot, ref_frame, k):
        raise AssertionError("retrieve must not be called for an out-of-range beat")

    (row,) = retrieval_rows(beats, decon_by_ref, retrieve_fn, lambda: KEYS)
    assert row["run"] == "DISCIPLINE_judge_v1" and row["beat"] == 15
    assert "warning" in row and "mrr" not in row


def test_retrieval_rows_unresolvable_fragment_counted():
    beats = [LabeledBeat(run="grace_judge_v1", beat=0, good=["Nope/missing"])]
    decon_by_ref = {"reference/grace.mp4": _decon(1)}
    (row,) = retrieval_rows(beats, decon_by_ref,
                            _retrieve_returning(["Shorts/3/Gym/C1402.MP4"]), lambda: KEYS)
    assert row["resolved_fragments"] == 0 and row["unresolved_fragments"] == 1
    assert row["recall@5"] == 0.0                 # empty relevant set -> conservative 0


def test_retrieval_rows_skips_beats_without_deconstruction():
    beats = [LabeledBeat(run="unmapped_run", beat=0, good=["Jogging/dji_184410"])]
    assert retrieval_rows(beats, {}, _retrieve_returning([]), lambda: KEYS) == []


def test_retrieval_rows_asks_for_max_k_candidates():
    seen = {}

    def retrieve_fn(shot, ref_frame, k):
        seen["k"] = k
        return []

    beats = [LabeledBeat(run="grace_judge_v1", beat=0, good=["Jogging/dji_184410"])]
    retrieval_rows(beats, {"reference/grace.mp4": _decon(1)}, retrieve_fn, lambda: KEYS,
                   k_values=(5, 10, 20))
    assert seen["k"] == 20


# --- save_results ---------------------------------------------------------------

def test_save_results_writes_json_and_md(tmp_path):
    rows = [{"run": "r", "beat": 0, "hit@5": True, "mrr": 0.5},
            {"run": "r", "beat": 1, "hit@5": False, "mrr": 0.0}]
    json_path, md_path = save_results(rows, "retrieval-test", out_dir=str(tmp_path))
    assert os.path.isfile(json_path) and json_path.endswith("-retrieval-test.json")
    assert os.path.isfile(md_path) and md_path.endswith("-retrieval-test.md")
    with open(json_path, encoding="utf-8") as fh:
        assert json.load(fh) == rows              # raw rows round-trip
    with open(md_path, encoding="utf-8") as fh:
        table = fh.read()
    assert "mrr" in table and "hit@5" in table and "| --- |" in table
