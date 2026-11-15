"""Publish providers: who actually talks to Instagram (or pretends to).

The safety latch lives in ``get_publisher``: the dry-run publisher is the default
AND the fallback - a real provider is only returned when it is explicitly named
in ``PUBLISH_PROVIDER`` and ``PUBLISH_LIVE=1`` is also set. Until both are true,
every scheduled post resolves to a dry run that records exactly what WOULD have
shipped, so the whole pipeline can be exercised end-to-end with zero risk of a
live post.

Real providers (wired once credentials exist; see docs/TRIAL-REELS.md):
  upload_post - Upload-Post's posting API (has an explicit trial-reel flag).
  graph - Meta's Instagram content-publishing API directly.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PublishResult:
    ok: bool
    media_id: str | None
    permalink: str | None
    payload: dict = field(default_factory=dict)
    dry_run: bool = False
    error: str | None = None


class DryRunPublisher:
    """Records what would ship; never touches the network."""

    name = "dry_run"

    def publish(self, *, video_url: str | None, caption: str | None,
                as_trial: bool = True) -> PublishResult:
        payload = {"video_url": video_url, "caption": caption, "as_trial": as_trial,
                   "note": "dry run - nothing was posted"}
        return PublishResult(ok=True, media_id=None, permalink=None,
                             payload=payload, dry_run=True)


class UploadPostPublisher:
    """Upload-Post posting API (api.upload-post.com) - supports the trial-reel flag."""

    name = "upload_post"

    # Meta graduation_strategy -> Upload-Post share_mode (same semantics, their names)
    _SHARE_MODES = {"MANUAL": "TRIAL_REELS_DONT_SHARE_TO_FOLLOWERS",
                    "SS_PERFORMANCE": "TRIAL_REELS_SHARE_TO_FOLLOWERS_IF_LIKED"}

    def __init__(self, api_key: str, user: str, graduation_strategy: str = "MANUAL"):
        self.api_key = api_key
        self.user = user
        self.graduation_strategy = graduation_strategy

    def build_request(self, *, video_url: str, caption: str | None,
                      as_trial: bool) -> dict:
        """The HTTP request as data (kept pure for tests; publish() sends it).

        Trial reels ride Upload-Post's ``share_mode`` (docs.upload-post.com/api/
        upload-video): DONT_SHARE_TO_FOLLOWERS = Meta's MANUAL (graduate by hand in
        the IG app); SHARE_TO_FOLLOWERS_IF_LIKED = SS_PERFORMANCE (auto-graduate a
        trial that performs in its 72h window).
        """
        data = {
            "user": self.user,
            "platform[]": "instagram",
            "video": video_url,
            "title": caption or "",
            "media_type": "REELS",
        }
        if as_trial:
            data["share_mode"] = self._SHARE_MODES.get(
                self.graduation_strategy, self._SHARE_MODES["MANUAL"])
        return {
            "url": "https://api.upload-post.com/api/upload",
            "headers": {"Authorization": f"ApiKey {self.api_key}"},
            "data": data,
        }

    def publish(self, *, video_url: str | None, caption: str | None,
                as_trial: bool = True) -> PublishResult:
        if not video_url:
            return PublishResult(ok=False, media_id=None, permalink=None,
                                 error="no public video URL (is MEDIA_BASE_URL set "
                                       "and the render synced to Supabase?)")
        import requests

        req = self.build_request(video_url=video_url, caption=caption, as_trial=as_trial)
        try:
            r = requests.post(req["url"], headers=req["headers"], data=req["data"],
                              timeout=300)
            body = r.json() if r.content else {}
        except Exception as exc:  # network/JSON failure -> recorded, retried by rescheduling
            return PublishResult(ok=False, media_id=None, permalink=None,
                                 payload={"request": req["data"]}, error=str(exc))
        if r.status_code >= 400 or not body.get("success", True):
            return PublishResult(ok=False, media_id=None, permalink=None,
                                 payload={"request": req["data"], "response": body},
                                 error=f"upload-post HTTP {r.status_code}: {body}")
        results = body.get("results", {}).get("instagram", {})
        return PublishResult(ok=True,
                             media_id=str(results.get("media_id") or "") or None,
                             permalink=results.get("url"),
                             payload={"response": body})


class GraphPublisher:
    """Meta Instagram content-publishing API, direct (container -> publish).

    Needs an IG professional account, an access token with instagram_content_publish,
    and the account's IG user id. Wired but OFF until PUBLISH_LIVE=1 + creds exist.
    """

    name = "graph"
    api = "https://graph.instagram.com/v23.0"

    def __init__(self, ig_user_id: str, access_token: str,
                 graduation_strategy: str = "MANUAL"):
        self.ig_user_id = ig_user_id
        self.access_token = access_token
        self.graduation_strategy = graduation_strategy  # MANUAL | SS_PERFORMANCE

    def build_container(self, *, video_url: str, caption: str | None,
                        as_trial: bool) -> dict:
        """The /media container params (pure for tests; publish() sends them).

        ``trial_params`` is Meta's trial-reel switch (media_type must be REELS):
        graduation_strategy MANUAL = promote to followers by hand in the IG app;
        SS_PERFORMANCE = Instagram auto-graduates a well-performing trial.
        """
        import json

        container = {"media_type": "REELS", "video_url": video_url,
                     "caption": caption or "",
                     "access_token": self.access_token}
        if as_trial:
            container["trial_params"] = json.dumps(
                {"graduation_strategy": self.graduation_strategy})
        return container

    def publish(self, *, video_url: str | None, caption: str | None,
                as_trial: bool = True) -> PublishResult:
        if not video_url:
            return PublishResult(ok=False, media_id=None, permalink=None,
                                 error="no public video URL")
        import time

        import requests

        container = self.build_container(video_url=video_url, caption=caption,
                                         as_trial=as_trial)
        try:
            r = requests.post(f"{self.api}/{self.ig_user_id}/media", data=container,
                              timeout=120)
            cid = r.json().get("id")
            if not cid:
                return PublishResult(ok=False, media_id=None, permalink=None,
                                     payload={"response": r.json()},
                                     error=f"container create failed: {r.text[:500]}")
            # poll until Instagram finishes ingesting the video, then publish
            for _ in range(60):
                s = requests.get(f"{self.api}/{cid}",
                                 params={"fields": "status_code",
                                         "access_token": self.access_token},
                                 timeout=60).json()
                if s.get("status_code") == "FINISHED":
                    break
                if s.get("status_code") == "ERROR":
                    return PublishResult(ok=False, media_id=None, permalink=None,
                                         payload={"status": s}, error="ingest error")
                time.sleep(5)
            p = requests.post(f"{self.api}/{self.ig_user_id}/media_publish",
                              data={"creation_id": cid,
                                    "access_token": self.access_token},
                              timeout=120)
            mid = p.json().get("id")
            if not mid:
                return PublishResult(ok=False, media_id=None, permalink=None,
                                     payload={"response": p.json()},
                                     error=f"publish failed: {p.text[:500]}")
            return PublishResult(ok=True, media_id=str(mid), permalink=None,
                                 payload={"creation_id": cid})
        except Exception as exc:
            return PublishResult(ok=False, media_id=None, permalink=None, error=str(exc))


def get_publisher(env=None):
    """Resolve the configured publisher - dry-run unless explicitly latched live.

    ``PUBLISH_PROVIDER`` unset/'dry_run' -> DryRunPublisher.
    A real provider name WITHOUT ``PUBLISH_LIVE=1`` -> DryRunPublisher (the latch).
    An unknown name (with the latch on) raises so a typo can't silently dry-run.
    """
    import os

    e = os.environ if env is None else env
    provider = e.get("PUBLISH_PROVIDER", "dry_run") or "dry_run"
    if provider == "dry_run":
        return DryRunPublisher()
    if e.get("PUBLISH_LIVE") != "1":
        return DryRunPublisher()
    if provider == "upload_post":
        return UploadPostPublisher(api_key=e.get("UPLOAD_POST_API_KEY", ""),
                                   user=e.get("UPLOAD_POST_USER", ""),
                                   graduation_strategy=e.get("PUBLISH_GRADUATION",
                                                             "MANUAL"))
    if provider == "graph":
        return GraphPublisher(ig_user_id=e.get("IG_USER_ID", ""),
                              access_token=e.get("IG_ACCESS_TOKEN", ""),
                              graduation_strategy=e.get("PUBLISH_GRADUATION", "MANUAL"))
    raise ValueError(f"unknown PUBLISH_PROVIDER: {provider}")
