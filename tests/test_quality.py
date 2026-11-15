"""Unit tests for the pure quality scoring in ingest/quality.py (no media / no model).

Quality detects a PROPERTY (camera motion, no subject, blur, dark), never a position.
Sub-scores are normalized [0,1] (1 = good): stability (1=still), exposure (1=well-lit),
sharpness (1=sharp), subject_present (1=person present).
"""
from reforge.ingest.quality import cv_quality_score, combine_quality


def test_clean_moment_scores_high_no_flags():
    score, flags = cv_quality_score(stability=0.9, exposure=0.8, sharpness=0.8, subject_present=1.0)
    assert score > 0.8
    assert flags == []


def test_no_subject_is_flagged():
    _score, flags = cv_quality_score(stability=0.9, exposure=0.8, sharpness=0.8, subject_present=0.0)
    assert "no_subject" in flags


def test_camera_motion_is_flagged_when_unstable():
    _score, flags = cv_quality_score(stability=0.1, exposure=0.8, sharpness=0.8, subject_present=1.0)
    assert "camera_motion" in flags


def test_blur_and_dark_are_flagged():
    _score, blur = cv_quality_score(stability=0.9, exposure=0.8, sharpness=0.05, subject_present=1.0)
    assert "blurry" in blur
    _score, dark = cv_quality_score(stability=0.9, exposure=0.05, sharpness=0.8, subject_present=1.0)
    assert "dark" in dark


def test_score_is_between_0_and_1():
    score, _ = cv_quality_score(0.5, 0.5, 0.5, 1.0)
    assert 0.0 <= score <= 1.0


# --- combine cv + vlm -------------------------------------------------------

def test_combine_quality_is_in_range_and_monotone():
    assert 0.0 <= combine_quality(0.5, 0.5) <= 1.0
    assert combine_quality(0.9, 0.9) > combine_quality(0.2, 0.2)


# Increment A drops NOTHING automatically: both mediapipe (misses dark / back-view / occluded
# subjects) and the VLM (mistakes dark + soft + handheld for camera-handling) over-dropped good
# training B-roll. Quality only down-weights; the Opus rerank is the gate and removal authority
# is the human Increment B pass. So there is no should_drop in the automatic pipeline to test -
# scoring (above) is what the pipeline relies on.
