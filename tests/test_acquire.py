"""Unit tests for the pure acquire logic (shortcode parsing + caption filter).

No network / DB / ffmpeg - only the import-light pure helpers.
"""
from dataclasses import dataclass

import pytest

from reforge.acquire.dedup import caption_density, caption_variety, is_caption_heavy
from reforge.acquire.download import shortcode_from_url


# --- shortcode_from_url -----------------------------------------------------

@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://www.instagram.com/reel/C0deSampl3x", "C0deSampl3x"),
        ("https://www.instagram.com/reel/C0deSampl3x/", "C0deSampl3x"),
        ("https://www.instagram.com/reel/C0deSampl3x/?igshid=abc123", "C0deSampl3x"),
        ("https://instagram.com/p/CabcdEFghij/", "CabcdEFghij"),
        ("https://www.instagram.com/p/CabcdEFghij", "CabcdEFghij"),
        ("http://www.instagram.com/reels/C0deSampl3x/", "C0deSampl3x"),
        ("https://www.instagram.com/tv/C0deSampl3x/", "C0deSampl3x"),
    ],
)
def test_shortcode_from_url_extracts_code(url, expected):
    assert shortcode_from_url(url) == expected


@pytest.mark.parametrize(
    "url",
    [
        "https://www.youtube.com/watch?v=abc",
        "https://tiktok.com/@x/video/123",
        "https://www.instagram.com/somebody/",  # profile, no reel/p code
        "not a url at all",
        "",
    ],
)
def test_shortcode_from_url_none_for_non_ig_reel(url):
    assert shortcode_from_url(url) is None


# --- caption_density --------------------------------------------------------

@dataclass
class _Shot:
    on_screen_text: str


def test_caption_density_empty_shots_is_zero():
    assert caption_density([]) == 0.0


def test_caption_density_all_text_is_one():
    shots = [_Shot("HELLO"), _Shot("WORLD"), _Shot("GO")]
    assert caption_density(shots) == pytest.approx(1.0)


def test_caption_density_half_text():
    shots = [_Shot("HELLO"), _Shot(""), _Shot("GO"), _Shot("   ")]
    assert caption_density(shots) == pytest.approx(0.5)


# --- caption_variety --------------------------------------------------------

def test_caption_variety_all_same_overlay_is_zero():
    # DISCIPLINE-style: one persistent phrase on every shot -> no variety
    shots = [_Shot("I NEVER LOSE"), _Shot("I never lose."), _Shot("I NEVER LOSE")]
    assert caption_variety(shots, "I NEVER LOSE") == pytest.approx(0.0)


def test_caption_variety_no_text_is_zero():
    assert caption_variety([_Shot(""), _Shot("  ")], "") == 0.0


def test_caption_variety_varied_captions_is_high():
    # listicle / talking-head: different text per beat
    shots = [_Shot("Tip 1"), _Shot("Tip 2"), _Shot("Tip 3"), _Shot("Tip 1")]
    assert caption_variety(shots, "Tip 1") == pytest.approx(0.5)


# --- is_caption_heavy -------------------------------------------------------

def test_is_caption_heavy_variety_threshold():
    # density > 0.4 trips it even with no overlay words
    assert is_caption_heavy(0.5, 0) is True   # varied captions per beat
    assert is_caption_heavy(0.35, 0) is False  # boundary is exclusive


def test_is_caption_heavy_word_threshold():
    # a wordy overlay trips it even at low density
    assert is_caption_heavy(0.0, 9) is True
    assert is_caption_heavy(0.0, 8) is False  # boundary is exclusive


def test_is_caption_heavy_discipline_single_word_passes():
    # DISCIPLINE-style single-word overlay on a low-caption reel is recreatable B-roll
    assert is_caption_heavy(0.1, 1) is False
