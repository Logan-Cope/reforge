"""Unit tests for the pure helpers in ingest/embed.py (no model load, no API).

L2-normalization (CLIP vectors are stored normalized so cosine distance behaves
well in pgvector) and the Phase 9B caption-lane text composition.
"""
import numpy as np
import pytest

from reforge.ingest.embed import caption_text, l2_normalize


def test_l2_normalize_makes_unit_norm():
    out = l2_normalize(np.array([3.0, 4.0]))
    assert np.linalg.norm(out) == pytest.approx(1.0)
    assert out[0] == pytest.approx(0.6)
    assert out[1] == pytest.approx(0.8)


def test_l2_normalize_zero_vector_returns_zeros():
    out = l2_normalize(np.zeros(4))
    assert np.allclose(out, 0.0)


def test_l2_normalize_returns_float32():
    out = l2_normalize(np.array([1, 2, 2], dtype=np.float64))
    assert out.dtype == np.float32


# --- Phase 9B: caption text composition for the caption-embedding lane -------

def _moment_row(**overrides):
    row = {"description": "Logan does explosive push-ups on the dock",
           "energy": "explosive", "location": "dock",
           "framing_notes": "full body, low angle"}
    row.update(overrides)
    return row


def test_caption_text_composes_all_fields():
    text = caption_text(_moment_row())
    assert "explosive push-ups on the dock" in text
    assert "explosive" in text
    assert "dock" in text
    assert "full body, low angle" in text


def test_caption_text_skips_missing_fields():
    text = caption_text(_moment_row(location=None, framing_notes=""))
    assert "push-ups" in text and "explosive" in text
    assert "Location" not in text and "Framing" not in text


def test_caption_text_empty_row_is_empty_string():
    assert caption_text({"description": None, "energy": None,
                         "location": None, "framing_notes": None}) == ""


def test_caption_text_no_double_period_on_description():
    text = caption_text(_moment_row(description="A sentence already ended."))
    assert "ended.." not in text
