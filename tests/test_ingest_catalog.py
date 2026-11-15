"""Unit tests for the pure catalog logic in ingest/pipeline.py.

No DB, no media, no model: just the dedup/listing/inheritance logic that decides
what gets cataloged and how clip-level annotation flows onto each moment window.
"""
import hashlib

from reforge.ingest.pipeline import (
    IS_LOGAN_EXCLUDE,
    build_moments,
    ingest_action,
    is_excluded,
    is_video_key,
    sha256_file,
    to_storage_key,
    video_keys,
)
from reforge.llm.base import ClipAnnotation


# --- video-file filtering ---------------------------------------------------

def test_is_video_key_accepts_known_video_extensions():
    assert is_video_key("library/C2060.MP4")
    assert is_video_key("a.mov")
    assert is_video_key("b/c.MOV")
    assert is_video_key("d.m4v")


def test_is_video_key_rejects_non_video():
    assert not is_video_key("library/sub")      # directory-like, no extension
    assert not is_video_key("photo.jpg")
    assert not is_video_key("notes.txt")


def test_video_keys_filters_and_preserves_order():
    listing = ["library/", "library/a.MP4", "library/b.jpg", "library/sub/c.mov"]
    assert video_keys(listing) == ["library/a.MP4", "library/sub/c.mov"]


# --- clip-level annotation inherited by every window ------------------------

def test_build_moments_inherits_clip_annotation_for_every_window():
    annotation = ClipAnnotation(
        is_logan=True,
        location="gym",
        energy="explosive",
        description="man doing weighted pull-ups",
        framing_notes="full body, centered",
    )
    windows = [(0.0, 1.75), (1.75, 3.5)]
    moments = build_moments(windows, annotation)

    assert len(moments) == 2
    for moment, (start, end) in zip(moments, windows):
        assert moment["start_s"] == start
        assert moment["end_s"] == end
        assert moment["is_logan"] is True
        assert moment["location"] == "gym"
        assert moment["energy"] == "explosive"
        assert moment["description"] == "man doing weighted pull-ups"
        assert moment["framing_notes"] == "full body, centered"


# --- dedup decision ---------------------------------------------------------

def test_ingest_action_fresh_for_new_clip():
    assert ingest_action(hash_exists=False, key_exists=False) == "fresh"


def test_ingest_action_skips_when_content_already_cataloged():
    # identical second run (same key+hash), or same bytes re-downloaded under a new key
    assert ingest_action(hash_exists=True, key_exists=True) == "skip"
    assert ingest_action(hash_exists=True, key_exists=False) == "skip"


def test_ingest_action_reingests_changed_clip():
    # same storage_key but different bytes -> content_hash not found -> refresh moments
    assert ingest_action(hash_exists=False, key_exists=True) == "reingest"


def test_ingest_action_force_reingests_or_starts_fresh():
    assert ingest_action(hash_exists=True, key_exists=True, force=True) == "reingest"
    assert ingest_action(hash_exists=True, key_exists=False, force=True) == "fresh"


# --- storage-key normalization ----------------------------------------------

def test_to_storage_key_leaves_clean_relative_key_unchanged():
    assert to_storage_key("library/C2060.MP4") == "library/C2060.MP4"


def test_to_storage_key_strips_leading_dot_slash():
    # data dir defaults to "./data", so derived thumbnail keys must be normalized
    assert to_storage_key("./data/thumbs/abc123/000.jpg") == "data/thumbs/abc123/000.jpg"


# --- is_logan exclusion (not-Logan clips that read as a solo man) -----------

def test_is_excluded_matches_filename_stem_in_any_dir():
    excl = {"IMG_5544", "IMG_5551"}
    assert is_excluded("library/Photos-3-001/IMG_5544.MOV", excl)
    assert is_excluded("library/sub/IMG_5551.mov", excl)


def test_is_excluded_ignores_extension_case_and_path():
    assert is_excluded("IMG_5544.mp4", {"img_5544"})        # case-insensitive


def test_is_excluded_false_for_other_clips():
    assert not is_excluded("library/C2060.MP4", {"IMG_5544", "IMG_5551"})
    assert not is_excluded("library/IMG_2577.MOV", {"IMG_5544"})


def test_known_exclusions_cover_the_validated_clips():
    # not-Logan (read as a solo man): IMG_5544, IMG_5551.
    # confirmed 2-person the 6-frame run missed (intermittent 2nd person): IMG_2589, IMG_2593.
    for stem in ("IMG_5544", "IMG_5551", "IMG_2589", "IMG_2593"):
        assert is_excluded(f"library/Photos-3-001/{stem}.MOV", IS_LOGAN_EXCLUDE)


# --- content hashing --------------------------------------------------------

def test_sha256_file_matches_hashlib(tmp_path):
    path = tmp_path / "clip.bin"
    data = b"reforge-content-bytes-123"
    path.write_bytes(data)
    assert sha256_file(str(path)) == hashlib.sha256(data).hexdigest()
