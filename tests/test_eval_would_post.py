"""Unit tests for the pure parts of eval/would_post.py (no API, no ffmpeg)."""
from reforge.eval.would_post import (
    WouldPostVerdict,
    _parse_would_post,
    _would_post_instruction,
)

# --- _would_post_instruction --------------------------------------------------

RUBRIC = "## Logan's rubric\n- subject fully in frame\n- no grainy footage"


def test_instruction_embeds_rubric_text():
    text = _would_post_instruction(RUBRIC)
    assert RUBRIC in text


def test_instruction_without_rubric_omits_rubric_header():
    text = _would_post_instruction("")
    assert "RUBRIC:" not in text
    assert "record_would_post" in text


def test_instruction_states_strictness_rule():
    text = _would_post_instruction(RUBRIC)
    # gold reels score 85-100; 2+ weak beats caps the score at 70
    assert "85-100" in text
    assert "70" in text
    assert "weak_beats" in text and "would_post" in text


# --- _parse_would_post ----------------------------------------------------------

def test_parse_full_result():
    verdict = _parse_would_post(
        {"score": 62, "would_post": False, "weak_beats": [1, 4], "notes": "two dark beats"})
    assert verdict == WouldPostVerdict(score=62, would_post=False,
                                       weak_beats=[1, 4], notes="two dark beats")


def test_parse_missing_fields_defaults():
    verdict = _parse_would_post({})
    assert verdict.score == 0
    assert verdict.would_post is False
    assert verdict.weak_beats == []
    assert verdict.notes == ""


def test_parse_clamps_score_to_0_100():
    assert _parse_would_post({"score": 150}).score == 100
    assert _parse_would_post({"score": -20}).score == 0


def test_parse_infers_would_post_from_score_when_missing():
    assert _parse_would_post({"score": 85}).would_post is True
    assert _parse_would_post({"score": 80}).would_post is True   # threshold is >= 80
    assert _parse_would_post({"score": 79}).would_post is False


def test_parse_explicit_would_post_wins_over_score():
    assert _parse_would_post({"score": 95, "would_post": False}).would_post is False
    assert _parse_would_post({"score": 10, "would_post": True}).would_post is True


def test_parse_weak_beats_drops_non_ints():
    verdict = _parse_would_post({"score": 50, "weak_beats": [0, "x", 2.0, None, 3]})
    assert verdict.weak_beats == [0, 2, 3]
