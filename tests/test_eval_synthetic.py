"""Unit tests for the synthetic-labeling helpers in eval/synthetic.py (no API, no DB)."""
from types import SimpleNamespace

from reforge.eval.synthetic import (
    dedup_by_moment_id,
    label_prompt,
    labeled_beat_record,
    parse_labels,
    pool_candidates,
)
from reforge.llm.base import ClipMoment, ReelBrief, ShotSpec


def _moment(mid: int, desc: str = "d") -> ClipMoment:
    return ClipMoment(clip_key=f"Shorts/{mid}.MP4", start_s=0.0, end_s=1.5, description=desc,
                      is_logan=True, framing_notes="full body", energy="moderate",
                      location="gym", moment_id=mid, quality_score=0.9)


def _shot(idx: int = 0) -> ShotSpec:
    return ShotSpec(idx=idx, start_s=0.0, end_s=1.0, description="sprints up a hill",
                    on_screen_text="", energy="explosive", query="hill sprint",
                    role="opener", intended_feeling="grit", intended_energy="explosive")


BRIEF = ReelBrief(vibe="gritty dawn grind", emotional_register="gritty",
                  energy_profile="builds", tone="raw", theme="discipline")


# --- dedup_by_moment_id -------------------------------------------------------

def test_dedup_unions_and_keeps_first_occurrence():
    a, b, c = _moment(1), _moment(2), _moment(3)
    b_dup = _moment(2, desc="same id, other retriever")
    pooled = dedup_by_moment_id([[a, b], [b_dup, c]])
    assert [m.moment_id for m in pooled] == [1, 2, 3]
    assert pooled[1] is b                       # first occurrence wins


def test_dedup_empty_lists():
    assert dedup_by_moment_id([]) == []
    assert dedup_by_moment_id([[], []]) == []


def test_pool_candidates_unions_both_retrievers():
    calls = []

    def fake_retrieve(shot, ref_frame, *, k, retriever, brief=None):
        calls.append((retriever, k, brief))
        return [_moment(1), _moment(2)] if retriever == "clip" else [_moment(2), _moment(3)]

    pooled = pool_candidates(_shot(), "/frames/shot00_0.jpg", BRIEF, k=30,
                             retrieve_fn=fake_retrieve)
    assert [m.moment_id for m in pooled] == [1, 2, 3]
    assert calls[0][:2] == ("clip", 30) and calls[1][:2] == ("hybrid", 30)
    assert calls[1][2] is BRIEF                 # brief reaches only the hybrid arm


# --- label_prompt ---------------------------------------------------------------

def test_label_prompt_contains_ask_brief_and_candidates():
    lines = ["energy=moderate; desc=jogging", "energy=explosive; desc=box jump"]
    prompt = label_prompt(_shot(), BRIEF, lines)
    assert "ROLE: opener" in prompt and "INTENDED FEELING: grit" in prompt
    assert "sprints up a hill" in prompt        # the shot's ask
    assert "gritty dawn grind" in prompt        # the brief
    assert "[0] energy=moderate; desc=jogging" in prompt
    assert "[1] energy=explosive; desc=box jump" in prompt
    assert "PLAUSIBLY" in prompt                # generous relevance bar, not final taste


def test_label_prompt_without_brief():
    prompt = label_prompt(_shot(), None, ["line"])
    assert "This beat needs:" in prompt and "VIBE" not in prompt


# --- parse_labels ----------------------------------------------------------------

def test_parse_labels_maps_indices_and_defaults_false():
    result = {"labels": [{"candidate": 0, "relevant": True}, {"candidate": 2, "relevant": False}]}
    assert parse_labels(result, 4) == [True, False, False, False]  # 1 and 3 unmentioned


def test_parse_labels_tolerates_junk():
    result = {"labels": [
        {"candidate": 9, "relevant": True},     # out of range
        {"candidate": -1, "relevant": True},    # negative
        {"candidate": "1", "relevant": True},   # non-int index
        {"candidate": True, "relevant": True},  # bool masquerading as int
        "junk",                                  # not a dict
        {"candidate": 1},                        # missing relevant -> False
        {"candidate": 0, "relevant": 1},         # truthy coerced
    ]}
    assert parse_labels(result, 2) == [True, False]


def test_parse_labels_handles_missing_or_malformed_result():
    assert parse_labels({}, 2) == [False, False]
    assert parse_labels({"labels": None}, 2) == [False, False]
    assert parse_labels(None, 2) == [False, False]


# --- labeled_beat_record ----------------------------------------------------------

def test_labeled_beat_record_shapes_json():
    labels = [(11, True), (12, False), (13, True)]
    record = labeled_beat_record("DISCIPLINE-synth", _shot(idx=4), BRIEF, labels)
    assert record["run"] == "DISCIPLINE-synth" and record["beat"] == 4
    assert record["relevant_moment_ids"] == [11, 13]
    assert record["pooled"] == 3 and record["relevant"] == 2
    assert record["query_meta"]["query"] == "hill sprint"
    assert record["query_meta"]["role"] == "opener"
    assert record["brief"]["vibe"] == "gritty dawn grind"


def test_labeled_beat_record_without_brief():
    record = labeled_beat_record("grace-synth", _shot(), None, [])
    assert "brief" not in record
    assert record["relevant_moment_ids"] == [] and record["pooled"] == 0


# --- from-labels harness (labeled_rows + summary_row) ------------------------------

def _decon(n_shots: int) -> SimpleNamespace:
    frames = [[f"/frames/shot{i:02d}_0.jpg", f"/frames/shot{i:02d}_1.jpg"]
              for i in range(n_shots)]
    return SimpleNamespace(frames_per_shot=frames)


def _record(run: str, beat: int, relevant: list[int]) -> dict:
    return {"run": run, "beat": beat,
            "query_meta": {"description": "jogs", "query": "jog", "energy": "moderate",
                           "role": "build", "intended_feeling": "calm",
                           "intended_energy": "moderate"},
            "brief": {"vibe": "v", "emotional_register": "e", "energy_profile": "p",
                      "tone": "t", "theme": "th"},
            "relevant_moment_ids": relevant, "pooled": 5, "relevant": len(relevant)}


def _retrieve_returning(mids: list[int]):
    def fn(shot, ref_frame, brief, k):
        assert ref_frame.endswith("_0.jpg")     # representative frame is [beat][0]
        assert shot.query == "jog" and brief.vibe == "v"  # rebuilt from the record
        return [_moment(m) for m in mids[:k]]
    return fn


def test_labeled_rows_metric_math():
    from reforge.eval.harness import labeled_rows

    labels = [_record("DISCIPLINE-synth", 1, relevant=[20, 99])]
    decon_by_ref = {"reference/DISCIPLINE.mp4": _decon(3)}
    (row,) = labeled_rows(labels, decon_by_ref, _retrieve_returning([10, 20, 30, 40, 50]))
    assert row["hit@5"] is True and row["hit@10"] is True
    assert row["mrr"] == 0.5                    # first relevant id at rank 2
    assert row["precision@5"] == 1 / 5
    assert row["recall@5"] == 0.5               # 1 of 2 relevant ids found (99 never retrieved)
    assert row["n_relevant"] == 2


def test_labeled_rows_miss_and_out_of_range_and_unmapped():
    from reforge.eval.harness import labeled_rows

    labels = [_record("grace-synth", 0, relevant=[99]),      # miss
              _record("grace-synth", 7, relevant=[1]),       # out of grid range
              _record("unmapped-run", 0, relevant=[1])]      # no reference -> skipped
    decon_by_ref = {"reference/grace.mp4": _decon(2)}
    rows = labeled_rows(labels, decon_by_ref, _retrieve_returning([10, 20]))
    assert len(rows) == 2
    assert rows[0]["hit@20"] is False and rows[0]["mrr"] == 0.0 and rows[0]["recall@5"] == 0.0
    assert "warning" in rows[1] and "mrr" not in rows[1]


def test_labeled_rows_asks_for_max_k():
    from reforge.eval.harness import labeled_rows

    seen = {}

    def fn(shot, ref_frame, brief, k):
        seen["k"] = k
        return []

    labeled_rows([_record("grace-synth", 0, [1])], {"reference/grace.mp4": _decon(1)},
                 fn, k_values=(5, 10, 20))
    assert seen["k"] == 20


def test_summary_row_means():
    from reforge.eval.harness import summary_row

    rows = [
        {"run": "r", "beat": 0, "hit@5": True, "precision@5": 0.4, "recall@5": 1.0,
         "hit@10": True, "precision@10": 0.2, "recall@10": 1.0,
         "hit@20": True, "precision@20": 0.1, "recall@20": 1.0, "mrr": 1.0},
        {"run": "r", "beat": 1, "hit@5": False, "precision@5": 0.0, "recall@5": 0.0,
         "hit@10": True, "precision@10": 0.1, "recall@10": 0.5,
         "hit@20": True, "precision@20": 0.05, "recall@20": 0.5, "mrr": 0.1},
        {"run": "r", "beat": 9, "warning": "out of range"},   # excluded from means
    ]
    summary = summary_row(rows)
    assert summary["run"] == "MEAN" and summary["beat"] == 2   # scored rows only
    assert summary["hit@5"] == 0.5                             # bool -> hit rate
    assert summary["hit@10"] == 1.0
    assert summary["mrr"] == 0.55
    assert abs(summary["precision@5"] - 0.2) < 1e-9
    assert summary["recall@10"] == 0.75


def test_summary_row_empty():
    from reforge.eval.harness import summary_row

    summary = summary_row([])
    assert summary["beat"] == 0 and summary["mrr"] == 0.0
