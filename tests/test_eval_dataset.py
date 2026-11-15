"""Unit tests for the pure taste-log dataset parsing in eval/dataset.py (no DB)."""
import os

from reforge.eval.dataset import (
    LabeledBeat,
    load_taste_log,
    parse_clip_ref,
    parse_taste_log,
    resolve_fragment,
)

# --- parse_clip_ref ----------------------------------------------------------

def test_parse_clip_ref_with_parenthetical():
    assert parse_clip_ref("clip244 Jogging/dji_184442 (drone run)") == "Jogging/dji_184442"


def test_parse_clip_ref_without_parenthetical():
    assert parse_clip_ref("clip243 Jogging/dji_184410") == "Jogging/dji_184410"


def test_parse_clip_ref_fragment_with_spaces():
    # the fragment itself can contain spaces; only the trailing parenthetical is stripped
    ref = "clip224 Candid/Copy of DJI_0013 (adjusting the drone screen)"
    assert parse_clip_ref(ref) == "Candid/Copy of DJI_0013"


def test_parse_clip_ref_no_path_token_returns_none():
    assert parse_clip_ref("clip 119 (C2111 behind-neck pull)") is None
    assert parse_clip_ref("just a plain note") is None
    assert parse_clip_ref("") is None


# --- parse_taste_log over the four real line shapes --------------------------

SWAP_LINE = (
    '{"ts":"2026-07-01","run":"DISCIPLINE_judge_v1","beat":6,"role":"run/build",'
    '"judge_pick":"clip244 Jogging/dji_184442 (drone run)","verdict":"swap",'
    '"logan_note":"not fully in the shot","rule":"subject must be fully in frame",'
    '"swapped_to":"clip243 Jogging/dji_184410 (full-body run toward camera)"}'
)
SUMMARY_LINE = (
    '{"ts":"2026-07-01","run":"DISCIPLINE_judge_v1","event":"review_summary",'
    '"kept":15,"swapped":2,"note":"first judge pass"}'
)
MULTI_BEAT_LINE = (
    '{"ts":"2026-07-01","run":"grace_judge_v1","beats":[4,5,7],'
    '"logan_note":"B4 mirror only; B5 out of frame","meta":"judge the moment"}'
)
KEEP_LINE = (
    '{"ts":"2026-07-01","run":"grace_judge_v2","beat":2,'
    '"judge_pick":"clip100 Gym/C1402 (pull-up)","verdict":"keep","logan_note":"good pick"}'
)
RULE_WITH_PICK_LINE = (
    '{"ts":"2026-07-01","run":"grace_judge_v3","beat":4,'
    '"judge_pick":"clip119 Gym/C2111 (behind-neck pull)",'
    '"logan_note":"grainy - cut it","rule":"reject grainy footage"}'
)
RULE_NOTE_ONLY_LINE = (
    '{"ts":"2026-07-01","run":"grace_judge_v3","beat":4,'
    '"logan_note":"clip 119 is grainy - cut it","rule":"reject grainy footage"}'
)
MALFORMED_LINE = '{"ts":"2026-07-01","run":"broken", this is not json}'


def test_parse_swap_line_yields_bad_pick_and_good_swap():
    beats = parse_taste_log([SWAP_LINE])
    assert beats == [
        LabeledBeat(run="DISCIPLINE_judge_v1", beat=6,
                    good=["Jogging/dji_184410"], bad=["Jogging/dji_184442"],
                    note="not fully in the shot")
    ]


def test_parse_keep_line_yields_good_pick():
    (lb,) = parse_taste_log([KEEP_LINE])
    assert lb.good == ["Gym/C1402"] and lb.bad == [] and lb.beat == 2


def test_parse_rule_line_with_pick_yields_bad():
    (lb,) = parse_taste_log([RULE_WITH_PICK_LINE])
    assert lb.bad == ["Gym/C2111"] and lb.good == []


def test_parse_skips_summary_multibeat_noteonly_and_malformed():
    lines = [SUMMARY_LINE, MULTI_BEAT_LINE, RULE_NOTE_ONLY_LINE, MALFORMED_LINE, SWAP_LINE]
    beats = parse_taste_log(lines)          # never raises
    assert len(beats) == 1 and beats[0].beat == 6


# --- resolve_fragment --------------------------------------------------------

KEYS = [
    "Shorts/3/Jogging/dji_184410.MP4",
    "Shorts/3/Walking/C2095.MP4",
    "Archive/Shorts/3/Jogging/dji_184410.MP4",
]


def test_resolve_fragment_single_match():
    assert resolve_fragment("Walking/C2095", KEYS) == "Shorts/3/Walking/C2095.MP4"


def test_resolve_fragment_zero_matches_returns_none():
    assert resolve_fragment("Gym/C9999", KEYS) is None


def test_resolve_fragment_multiple_matches_returns_shortest():
    assert resolve_fragment("Jogging/dji_184410", KEYS) == "Shorts/3/Jogging/dji_184410.MP4"


def test_resolve_fragment_is_case_insensitive():
    assert resolve_fragment("jogging/DJI_184410", KEYS) == "Shorts/3/Jogging/dji_184410.MP4"


# --- the real taste log ------------------------------------------------------

def test_load_sample_taste_log():
    path = os.path.join(os.path.dirname(__file__), "fixtures", "taste-log-sample.jsonl")
    beats = load_taste_log(path)
    # Only the two per-beat swap lines carry per-clip data; summary, multi-beat
    # note lines, and the judge_pick-less rejection line are skipped by spec.
    assert len(beats) == 2
    assert all(isinstance(b, LabeledBeat) and (b.good or b.bad) for b in beats)
    assert beats[0].good == ["Running/run_0410"]
    assert beats[0].bad == ["Running/run_0422"]


def test_resolve_fragment_token_fallback():
    # chat-judge shorthand "Jogging/dji_184442" vs the real long filename
    keys = [
        "library/Shorts/Outdoors/Jogging/dji_mimo_20251007_184442_20251007184436_x_video.mp4",
        "library/Shorts/Outdoors/Jogging/dji_mimo_20251007_174132_20251007214124_x_video.mp4",
        "library/Shorts/Outdoors/Candid/dji_mimo_20251007_184442_copy_video.mp4",
    ]
    from reforge.eval.dataset import resolve_fragment
    got = resolve_fragment("Jogging/dji_184442", keys)
    assert got == keys[0]  # all tokens (jogging, dji, 184442) present; shortest match wins


def test_resolve_fragment_token_fallback_no_match():
    from reforge.eval.dataset import resolve_fragment
    assert resolve_fragment("Jogging/dji_999999", ["library/Jogging/dji_mimo_184442.mp4"]) is None
