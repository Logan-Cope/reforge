"""Unit tests for the holistic gate (pure parts only - no API, no ffmpeg)."""
import json

from reforge.llm.base import ClipMoment, GateVerdict, ShotSpec, SlotPlan
from reforge.llm.claude import _parse_gate_verdicts
from reforge.match.gate import gold_reference_paths, reel_state_excluding, slot_line
from reforge.match.matcher import MatchedSlot, exclude_moments


class _Render:
    def __init__(self, src):
        self.src = src


def _shot(idx):
    return ShotSpec(idx=idx, start_s=idx * 1.0, end_s=idx * 1.0 + 0.9,
                    description="d", on_screen_text="", energy="moderate", query="q")


def _moment(clip_id, moment_id, location="gym", description="pushups on the dock"):
    return ClipMoment("library/x.mp4", 0.0, 1.6, description, True, "fn", "moderate",
                      location=location, clip_id=clip_id, moment_id=moment_id)


def _matched(idx, clip_id, moment_id, location="gym", reason="fits the beat"):
    plan = SlotPlan(moment=_moment(clip_id, moment_id, location), in_point=0.5, reason=reason)
    return MatchedSlot(shot=_shot(idx), plan=plan, render=_Render(f"/lib/clip{clip_id}.mp4"))


# --- GateVerdict + verdict parsing -------------------------------------------

def test_gate_verdict_defaults():
    v = GateVerdict(slot=3, hold=True)
    assert v.critique == ""


def test_parse_gate_verdicts_missing_slots_default_hold():
    result = {"verdicts": [{"slot": 1, "hold": False, "critique": "too dark"}]}
    verdicts = _parse_gate_verdicts(result, 3)
    assert [v.slot for v in verdicts] == [0, 1, 2]
    assert [v.hold for v in verdicts] == [True, False, True]
    assert verdicts[1].critique == "too dark"
    assert verdicts[0].critique == ""


def test_parse_gate_verdicts_skips_out_of_range_and_junk():
    result = {"verdicts": [{"slot": 9, "hold": False, "critique": "x"},
                           {"slot": -1, "hold": False, "critique": "y"},
                           {"slot": "0", "hold": False, "critique": "z"},
                           {"slot": 0, "hold": True, "critique": ""}]}
    verdicts = _parse_gate_verdicts(result, 2)
    assert len(verdicts) == 2
    assert all(v.hold for v in verdicts)


def test_parse_gate_verdicts_empty_response_all_hold():
    verdicts = _parse_gate_verdicts({}, 2)
    assert [(v.slot, v.hold) for v in verdicts] == [(0, True), (1, True)]


# --- slot_line ---------------------------------------------------------------

def test_slot_line_format():
    m = _matched(4, clip_id=7, moment_id=70, location="dock", reason="calm payoff beat")
    assert slot_line(m) == "slot 04 | clip7.mp4 | loc=dock | reason: calm payoff beat"


def test_slot_line_missing_location():
    m = _matched(0, clip_id=1, moment_id=10)
    m.plan.moment.location = None
    assert "loc=?" in slot_line(m)


# --- reel_state_excluding ----------------------------------------------------

def test_reel_state_excluding_drops_exactly_one_slot():
    matched = [_matched(0, 1, 10, "gym"), _matched(1, 2, 20, "dock"), _matched(2, 3, 30, "gym")]
    state = reel_state_excluding(matched, 1)
    assert state["used_clip_ids"] == {1, 3}
    assert state["used_moment_ids"] == {10, 30}
    assert state["location_counts"] == {"gym": 2}
    assert state["clip_use_count"] == {1: 1, 3: 1}
    assert len(state["used_looks"]) == 2
    assert all(look.startswith("gym:") for look in state["used_looks"])


def test_reel_state_excluding_matches_build_edl_fields():
    state = reel_state_excluding([_matched(0, 1, 10)], 0)
    assert state == {"used_clip_ids": set(), "used_moment_ids": set(),
                     "location_counts": {}, "clip_use_count": {}, "used_looks": []}


# --- gold_reference_paths ----------------------------------------------------

def test_gold_reference_paths_sorted_mp4_only(tmp_path):
    (tmp_path / "b_gold.mp4").write_bytes(b"")
    (tmp_path / "a_gold.mp4").write_bytes(b"")
    (tmp_path / "notes.txt").write_text("no")
    paths = gold_reference_paths(str(tmp_path))
    assert [p.split("/")[-1] for p in paths] == ["a_gold.mp4", "b_gold.mp4"]


def test_gold_reference_paths_missing_dir_empty(tmp_path):
    assert gold_reference_paths(str(tmp_path / "nope")) == []


# --- exclude_moments (the gate re-pick candidate filter) ---------------------

def test_exclude_moments_filters_by_moment_id():
    cands = [_moment(1, 10), _moment(2, 20), _moment(3, 30)]
    out = exclude_moments(cands, {20})
    assert [c.moment_id for c in out] == [10, 30]


def test_exclude_moments_noop_when_empty():
    cands = [_moment(1, 10)]
    assert exclude_moments(cands, frozenset()) == cands


# --- match_slot: exclusion + critique passthrough (stubbed I/O) --------------

def test_match_slot_excludes_moment_and_passes_critique(monkeypatch, tmp_path):
    from reforge.ingest import frames as frames_mod
    from reforge.ingest import probe as probe_mod
    from reforge.match import matcher as matcher_mod
    from reforge.match import retrieve as retrieve_mod
    from reforge.llm.base import ReelBrief

    old = _moment(1, 10, "gym")
    fresh = _moment(2, 20, "dock")
    seen = {}

    monkeypatch.setattr(retrieve_mod, "retrieve", lambda shot, frame, k, brief=None: [old, fresh])
    monkeypatch.setattr(retrieve_mod, "shortlist",
                        lambda vision, brief, shot, cands, n, usage, freshness_weight: list(cands))
    monkeypatch.setattr(frames_mod, "extract_frame", lambda src, t, out, height=720: out)
    monkeypatch.setattr(probe_mod, "probe_media", lambda src: {"duration_s": 10.0})

    class _Storage:
        def ensure_local(self, key):
            return f"/lib/{key}"

    class _Vision:
        def rerank(self, shot, shot_frames, candidate_frames, brief,
                   avoid_looks=None, critique=""):
            seen["critique"] = critique
            seen["candidates"] = [c.moment_id for c, _ in candidate_frames]
            top = candidate_frames[0][0]
            return [SlotPlan(moment=top, in_point=0.4, reason="repicked")]

    m = matcher_mod.match_slot(
        _shot(0), ["f0.jpg"], _Vision(), ReelBrief("v", "e", "p", "t", "th"),
        k=20, shortlist_n=6, cand_frames=2, storage=_Storage(), usage={},
        fresh_w=0.0, cap=2, used_clip_ids=set(), used_moment_ids=set(),
        location_counts={}, clip_use_count={}, used_looks=[],
        exclude_moment_ids={10}, critique="too dark; find a readable subject",
    )
    assert seen["candidates"] == [20]          # the excluded moment never reached the rerank
    assert seen["critique"] == "too dark; find a readable subject"
    assert m.plan.moment.moment_id == 20
    assert m.render.zoom == 1.0


# --- append_taste_log --------------------------------------------------------

def test_append_taste_log_creates_appends_and_stamps(tmp_path):
    from reforge.taste import append_taste_log

    path = tmp_path / "sub" / "taste-log.jsonl"
    append_taste_log({"event": "gate_repick", "slot": 3}, str(path))
    append_taste_log({"event": "gate_summary", "ts": "2020-01-01"}, str(path))

    lines = path.read_text().strip().splitlines()
    assert len(lines) == 2
    first, second = (json.loads(line) for line in lines)
    assert first["event"] == "gate_repick" and first["slot"] == 3
    assert len(first["ts"]) == 10 and first["ts"][4] == "-"   # ISO date stamped
    assert second["ts"] == "2020-01-01"                        # existing ts preserved


def test_run_gate_rejections_accumulate_and_exclude_whole_clip(monkeypatch, tmp_path):
    """A rejected slot must not re-pick the same CLIP, and rejections accumulate across rounds."""
    from types import SimpleNamespace

    from reforge.match import gate as gate_mod

    matched = [_matched(0, clip_id=10, moment_id=100), _matched(1, clip_id=20, moment_id=200)]
    for m in matched:  # run_gate extracts frames at in_point/midpoint from each render
        m.render.in_point, m.render.duration = 0.5, 0.9
    decon = SimpleNamespace(brief=None, frames_per_shot=[["f0.jpg"], ["f1.jpg"]])

    # slot 0 gets rejected in round 1 AND round 2; round 3 all-hold
    verdict_rounds = iter([
        [GateVerdict(0, False, "too dark"), GateVerdict(1, True)],
        [GateVerdict(0, False, "still weak"), GateVerdict(1, True)],
        [GateVerdict(0, True), GateVerdict(1, True)],
    ])
    vision = SimpleNamespace(gate_review=lambda *a, **k: next(verdict_rounds))

    seen_calls = []
    repick_ids = iter([(30, 300), (40, 400)])  # (clip_id, moment_id) per re-pick

    def fake_match_slot(shot, frames, v, brief, **kw):
        seen_calls.append(kw)
        cid, mid = next(repick_ids)
        new = _matched(shot.idx, clip_id=cid, moment_id=mid)
        new.render.in_point, new.render.duration = 0.5, 0.9
        return new

    monkeypatch.setattr(gate_mod, "_gold_frames", lambda tmp: [])
    monkeypatch.setattr("reforge.match.matcher.match_slot", fake_match_slot)
    monkeypatch.setattr("reforge.ingest.frames.extract_frame", lambda src, t, out: out)
    monkeypatch.setattr("reforge.db.client.clip_usage", lambda: {})

    out, events = gate_mod.run_gate(decon, matched, vision, max_iters=3, storage=object())

    # round 1 re-pick: original clip 10 hard-excluded at CLIP level, moment 100 excluded
    assert 10 in seen_calls[0]["hard_exclude_clip_ids"]
    assert 100 in seen_calls[0]["exclude_moment_ids"]
    # round 2 re-pick: BOTH prior rejections excluded (10 original + 30 from round 1)
    assert {10, 30} <= seen_calls[1]["hard_exclude_clip_ids"]
    assert {100, 300} <= seen_calls[1]["exclude_moment_ids"]
    assert out[0].plan.moment.clip_id == 40
    assert len(events) == 2


def test_parse_gate_verdicts_skips_non_dict_rows():
    # Sonnet 5 emitted bare strings in a non-strict tool call; parser must not crash
    result = {"verdicts": ["junk-string", {"slot": 0, "hold": False, "critique": "x"}, 42]}
    verdicts = _parse_gate_verdicts(result, 2)
    assert verdicts[0].hold is False and verdicts[1].hold is True


def test_match_slot_hard_excludes_rejected_clips(monkeypatch, tmp_path):
    """Gate-rejected clips must be dropped at retrieval, unreachable even by the thin-beat fallback."""
    from types import SimpleNamespace

    from reforge.match import matcher as matcher_mod
    from reforge.match import retrieve as retrieve_mod

    rejected = _moment(clip_id=10, moment_id=100)
    fresh = _moment(clip_id=30, moment_id=300)
    monkeypatch.setattr(retrieve_mod, "retrieve", lambda shot, frame, k, brief=None: [rejected, fresh])
    monkeypatch.setattr(retrieve_mod, "shortlist", lambda v, b, s, c, n, usage, freshness_weight: c)
    monkeypatch.setattr("reforge.ingest.frames.extract_frame", lambda src, t, out: out)
    monkeypatch.setattr("reforge.ingest.frames.sample_timestamps", lambda dur, n: [0.1])
    monkeypatch.setattr("reforge.ingest.probe.probe_media", lambda src: {"duration_s": 5.0})

    seen_pools = []

    def fake_rerank(shot, shot_frames, candidate_frames, brief, avoid_looks=None, critique=""):
        seen_pools.append([c.clip_id for c, _f in candidate_frames])
        from reforge.llm.base import SlotPlan
        return [SlotPlan(moment=candidate_frames[0][0], in_point=0.5, score=0.9)]

    vision = SimpleNamespace(rerank=fake_rerank)
    storage = SimpleNamespace(ensure_local=lambda key: str(tmp_path / "c.mp4"))

    out = matcher_mod.match_slot(
        _shot(0), ["f.jpg"], vision, None, k=20, shortlist_n=6, cand_frames=1,
        storage=storage, usage={}, fresh_w=0.0, cap=2, used_clip_ids=set(),
        used_moment_ids=set(), location_counts={}, clip_use_count={}, used_looks=[],
        hard_exclude_clip_ids={10},
    )
    assert seen_pools == [[30]]  # rejected clip never reached the rerank pool
    assert out.plan.moment.clip_id == 30
