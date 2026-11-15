"""Pure-logic tests for publish/scheduler.py (trial-reel scheduling; no DB, no network).

Covers: parsing the schedule form's naive datetime-local string into aware UTC
(ET by default - Logan schedules in Eastern), rejecting past/garbage input, the
absolute public video URL a publisher needs, local display formatting, and the
reels-row update derived from a publish result (dry runs must NOT mark the reel
published).
"""
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pytest

from reforge.publish.scheduler import (
    ACTIVE_STATUSES,
    fmt_local,
    parse_schedule_at,
    public_video_url,
    reel_update_for,
)
from reforge.publish.providers import PublishResult

ET = ZoneInfo("America/New_York")
NOW = datetime(2026, 7, 20, 12, 0, tzinfo=timezone.utc)  # 8:00 AM ET


# --- parse_schedule_at --------------------------------------------------------

def test_parse_schedule_at_interprets_naive_input_as_eastern():
    dt = parse_schedule_at("2026-07-21T09:30", now=NOW)
    # 9:30 EDT == 13:30 UTC - proves the naive string was read as Eastern (UTC-4 in July)
    assert dt == datetime(2026, 7, 21, 9, 30, tzinfo=ET)
    assert dt == datetime(2026, 7, 21, 13, 30, tzinfo=timezone.utc)


def test_parse_schedule_at_returns_utc():
    dt = parse_schedule_at("2026-07-21T09:30", now=NOW)
    assert dt.tzinfo == timezone.utc


def test_parse_schedule_at_honors_explicit_tz():
    dt = parse_schedule_at("2026-07-21T09:30", tz_name="UTC", now=NOW)
    assert dt == datetime(2026, 7, 21, 9, 30, tzinfo=timezone.utc)


def test_parse_schedule_at_rejects_the_past():
    with pytest.raises(ValueError, match="future"):
        parse_schedule_at("2026-07-19T09:30", now=NOW)


def test_parse_schedule_at_rejects_garbage():
    for bad in ("", "not-a-date", "2026-13-40T99:99"):
        with pytest.raises(ValueError):
            parse_schedule_at(bad, now=NOW)


# --- public_video_url ---------------------------------------------------------

def test_public_video_url_joins_base_and_key():
    url = public_video_url("output/grace_run3.mp4", "https://x.supabase.co/media")
    assert url == "https://x.supabase.co/media/output/grace_run3.mp4"


def test_public_video_url_tolerates_abs_paths_and_trailing_slash():
    url = public_video_url("/Users/me/repo/output/DISCIPLINE_run7.mp4",
                           "https://x.supabase.co/media/")
    assert url == "https://x.supabase.co/media/output/DISCIPLINE_run7.mp4"


def test_public_video_url_none_without_base_or_key():
    assert public_video_url("output/a.mp4", "") is None
    assert public_video_url(None, "https://x.supabase.co/media") is None
    assert public_video_url("", "https://x.supabase.co/media") is None


# --- fmt_local ----------------------------------------------------------------

def test_fmt_local_renders_eastern():
    dt = datetime(2026, 7, 21, 13, 30, tzinfo=timezone.utc)  # 9:30 AM EDT
    assert fmt_local(dt) == "Jul 21, 9:30 AM ET"


def test_fmt_local_none_safe():
    assert fmt_local(None) == ""


# --- reel_update_for ----------------------------------------------------------

def test_live_publish_marks_reel_published():
    res = PublishResult(ok=True, media_id="1789", permalink="https://instagram.com/p/x",
                        payload={}, dry_run=False)
    upd = reel_update_for(res, as_trial=True)
    assert upd == {"status": "published", "ig_media_id": "1789", "is_trial": True}


def test_dry_run_never_marks_reel_published():
    res = PublishResult(ok=True, media_id=None, permalink=None,
                        payload={"note": "dry run"}, dry_run=True)
    assert reel_update_for(res, as_trial=True) is None


def test_failed_publish_never_marks_reel_published():
    res = PublishResult(ok=False, media_id=None, permalink=None, payload={},
                        dry_run=False, error="boom")
    assert reel_update_for(res, as_trial=True) is None


# --- statuses -----------------------------------------------------------------

def test_active_statuses_cover_scheduled_and_publishing():
    assert ACTIVE_STATUSES == ("scheduled", "publishing")
