"""Unit tests for rotation-aware display dimensions in render/recipe.py.

Vertically-shot phone/Sony clips are stored landscape-coded (e.g. 3840x2160) with a
rotate=90/270 tag; ffmpeg autorotates on decode, so the real DISPLAY frame is 2160x3840
(already 9:16). `display_dims` returns the post-rotation dims the crop must be computed
against.
"""
from reforge.render.recipe import display_dims


def test_no_rotation_keeps_coded_dims():
    assert display_dims(3840, 2160, 0) == (3840, 2160)
    assert display_dims(1080, 1920, 0) == (1080, 1920)


def test_90_and_270_swap_to_vertical():
    assert display_dims(3840, 2160, 90) == (2160, 3840)
    assert display_dims(3840, 2160, 270) == (2160, 3840)


def test_negative_and_float_rotation_use_abs():
    assert display_dims(3840, 2160, -90) == (2160, 3840)
    assert display_dims(3840, 2160, -90.0) == (2160, 3840)


def test_180_does_not_swap():
    assert display_dims(3840, 2160, 180) == (3840, 2160)


def test_none_rotation_treated_as_zero():
    assert display_dims(2160, 3840, None) == (2160, 3840)
