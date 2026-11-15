"""Pure scheduling logic for trial-reel publishing (no DB, no network).

The schedule form posts a naive ``datetime-local`` string; the reviewer thinks in
Eastern time, so naive input is interpreted in ``PUBLISH_TZ`` (default
America/New_York) and stored as UTC. ``reel_update_for`` is the one place that
decides when a publish result is allowed to mark the reel published - dry runs and
failures never do.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from reforge.publish.providers import PublishResult

DEFAULT_TZ = "America/New_York"

# scheduled_posts.status values that still occupy the reel's schedule slot.
ACTIVE_STATUSES = ("scheduled", "publishing")


def _tz(tz_name: str | None) -> ZoneInfo:
    return ZoneInfo(tz_name or os.getenv("PUBLISH_TZ") or DEFAULT_TZ)


def parse_schedule_at(raw: str, tz_name: str | None = None, *,
                      now: datetime | None = None) -> datetime:
    """Parse the form's ``YYYY-MM-DDTHH:MM`` into an aware UTC datetime.

    Naive input is read in ``tz_name`` (default: PUBLISH_TZ env / Eastern).
    Raises ValueError for garbage or a time not in the future.
    """
    try:
        dt = datetime.fromisoformat((raw or "").strip())
    except (ValueError, TypeError) as exc:
        raise ValueError(f"bad schedule time: {raw!r}") from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_tz(tz_name))
    dt = dt.astimezone(timezone.utc)
    if dt <= (now or datetime.now(timezone.utc)):
        raise ValueError("schedule time must be in the future")
    return dt


def public_video_url(output_key: str | None, media_base_url: str | None) -> str | None:
    """The absolute URL a publisher can hand Instagram for this render.

    Mirrors ``queries.reel_video_url`` key normalization (anything after the last
    ``output/`` segment) but never falls back to a relative dev path - a publisher
    needs a public URL or nothing.
    """
    if not output_key or not media_base_url:
        return None
    key = output_key
    if "output/" in key:
        key = key.rsplit("output/", 1)[-1]
    if not key:
        return None
    return f"{media_base_url.rstrip('/')}/output/{key}"


def fmt_local(dt: datetime | None, tz_name: str | None = None) -> str:
    """Render a stored UTC time in the reviewer's timezone for the schedule UI."""
    if dt is None:
        return ""
    local = dt.astimezone(_tz(tz_name))
    # %-I is glibc/BSD-only but this renders on the Mac worker + local dev only.
    return local.strftime("%b %-d, %-I:%M %p ET" if "New_York" in str(local.tzinfo)
                          else "%b %-d, %-I:%M %p %Z")


def reel_update_for(result: PublishResult, *, as_trial: bool) -> dict | None:
    """The ``reels`` columns a publish result justifies updating - or None.

    Only a REAL successful publish touches the reel (status/ig_media_id/is_trial);
    dry runs and failures leave it exactly as it was.
    """
    if not result.ok or result.dry_run:
        return None
    return {"status": "published", "ig_media_id": result.media_id, "is_trial": as_trial}
