"""Pure-logic tests for the Tier-0/Tier-1 split (no DB, no media).

The keystone: a re-score reads CACHED Tier-0 raw signals and recomputes the Tier-1 derived
score with a pure function - never re-decoding or re-embedding. `score_window` is that pure
function; `window_idx_from_thumbnail_key` maps an existing moment's on-disk frame back to its
cached window so the backfill can populate the cache without touching video.
"""
from reforge.ingest.cache import window_idx_from_thumbnail_key
from reforge.ingest.quality import score_window


# --- Tier-1 pure re-score ---------------------------------------------------

def test_score_window_clean_signals_high_no_flags():
    r = score_window(exposure=0.8, sharpness=0.8, stability=0.9,
                     subject_present=1.0, vlm_usable=1.0, vlm_flags=[])
    assert r["quality_score"] > 0.8
    assert r["vlm_quality"] == 1.0
    assert r["quality_flags"] == []
    assert 0.0 <= r["cv_quality"] <= 1.0


def test_score_window_merges_cv_and_vlm_flags():
    r = score_window(exposure=0.05, sharpness=0.05, stability=0.1,
                     subject_present=0.0, vlm_usable=0.1, vlm_flags=["camera_handling"])
    # CV flags (no_subject/camera_motion/blurry/dark) + the VLM flag all recorded, none dropped.
    assert "camera_handling" in r["quality_flags"]
    assert "no_subject" in r["quality_flags"]
    assert 0.0 <= r["quality_score"] <= 1.0


def test_score_window_is_monotone_in_vlm():
    lo = score_window(exposure=0.6, sharpness=0.6, stability=0.6,
                      subject_present=1.0, vlm_usable=0.1, vlm_flags=[])
    hi = score_window(exposure=0.6, sharpness=0.6, stability=0.6,
                      subject_present=1.0, vlm_usable=0.9, vlm_flags=[])
    assert hi["quality_score"] > lo["quality_score"]


def test_score_window_combine_weight_shifts_toward_cv():
    # A high cv but low vlm: weighting cv more (the tunable scorer param) raises the score.
    kw = dict(exposure=0.9, sharpness=0.9, stability=0.9, subject_present=1.0,
              vlm_usable=0.0, vlm_flags=[])
    base = score_window(**kw)                  # default 0.5/0.5
    cv_heavy = score_window(**kw, combine_w=0.8)
    assert cv_heavy["quality_score"] > base["quality_score"]
    assert "combine_w" not in base             # the returned dict carries scores, not the knob


# --- map an existing frame back to its window -------------------------------

def test_window_idx_from_thumbnail_key():
    assert window_idx_from_thumbnail_key("data/thumbs/b34d1c8fa839/004.jpg") == 4
    assert window_idx_from_thumbnail_key("data/thumbs/abc/000.jpg") == 0
    assert window_idx_from_thumbnail_key("thumbs/h/012b.jpg") == 12   # f1 frame '012b.jpg'


def test_window_idx_from_thumbnail_key_bad_input_is_none():
    assert window_idx_from_thumbnail_key("") is None
    assert window_idx_from_thumbnail_key("data/thumbs/h/cover.jpg") is None
    assert window_idx_from_thumbnail_key(None) is None
