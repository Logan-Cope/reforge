"""Pure tests for publish/captions.py (prompt building + response cleanup; no API)."""
from reforge.publish.captions import caption_prompt, parse_caption

BRIEF = {"tone": "bold, charismatic, unapologetic",
         "vibe": "confident & aspirational gym culture",
         "theme": "body transformation & gym confidence"}


def test_prompt_carries_brief_and_rules():
    p = caption_prompt(BRIEF, "library/references/C0deSampl3y.mp4")
    assert "body transformation" in p and "bold" in p
    assert "one line" in p.lower()
    assert "hashtag" in p.lower()          # the no-hashtags rule is stated
    assert "em dash" in p.lower()          # house rule: no em dashes in published text


def test_prompt_tolerates_missing_brief():
    p = caption_prompt(None, "reference/grace.mp4")
    assert "grace" in p


def test_parse_caption_strips_quotes_and_newlines():
    assert parse_caption('"Discipline hits different at 5am."\n') == \
        "Discipline hits different at 5am."
    assert parse_caption("Caption: chase the version of you they said was impossible") == \
        "chase the version of you they said was impossible"


def test_parse_caption_takes_first_line_and_caps_length():
    multi = "line one is the caption\nline two is chatter"
    assert parse_caption(multi) == "line one is the caption"
    assert len(parse_caption("x" * 500)) <= 220


def test_parse_caption_strips_em_dashes():
    assert " - " not in parse_caption("built different - every day")


def test_parse_caption_empty_safe():
    assert parse_caption("") == ""
    assert parse_caption(None) == ""
