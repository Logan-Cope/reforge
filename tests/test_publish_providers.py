"""Provider-layer tests for publish/providers.py (no network - HTTP is faked).

The safety latch is the point: get_publisher() returns the dry-run publisher unless
BOTH a real provider is configured AND PUBLISH_LIVE=1, so nothing ships live by
default no matter what else is set.
"""
import pytest

from reforge.publish.providers import (
    DryRunPublisher,
    PublishResult,
    get_publisher,
)


# --- dry-run publisher --------------------------------------------------------

def test_dry_run_publishes_nothing_and_echoes_payload():
    res = DryRunPublisher().publish(
        video_url="https://x.supabase.co/media/output/a.mp4",
        caption="God first, then the gym.", as_trial=True)
    assert res.ok is True
    assert res.dry_run is True
    assert res.media_id is None
    assert res.payload["video_url"].endswith("a.mp4")
    assert res.payload["as_trial"] is True


def test_dry_run_tolerates_missing_video_url():
    res = DryRunPublisher().publish(video_url=None, caption="x", as_trial=True)
    assert res.ok is True and res.dry_run is True
    assert "video_url" in res.payload


# --- real-provider payloads (pure; verified against vendor/Meta docs 2026-07) --

def test_upload_post_trial_rides_share_mode():
    from reforge.publish.providers import UploadPostPublisher

    req = UploadPostPublisher(api_key="k", user="logan").build_request(
        video_url="https://x/v.mp4", caption="cap", as_trial=True)
    assert req["url"] == "https://api.upload-post.com/api/upload"
    assert req["headers"]["Authorization"] == "ApiKey k"
    assert req["data"]["share_mode"] == "TRIAL_REELS_DONT_SHARE_TO_FOLLOWERS"
    assert req["data"]["media_type"] == "REELS"
    # a non-trial post must NOT send any share_mode
    req2 = UploadPostPublisher(api_key="k", user="logan").build_request(
        video_url="https://x/v.mp4", caption="cap", as_trial=False)
    assert "share_mode" not in req2["data"]


def test_upload_post_ss_performance_maps_to_share_if_liked():
    from reforge.publish.providers import UploadPostPublisher, get_publisher

    pub = get_publisher(env={"PUBLISH_PROVIDER": "upload_post", "PUBLISH_LIVE": "1",
                             "UPLOAD_POST_API_KEY": "k", "UPLOAD_POST_USER": "logan",
                             "PUBLISH_GRADUATION": "SS_PERFORMANCE"})
    assert isinstance(pub, UploadPostPublisher)
    req = pub.build_request(video_url="https://x/v.mp4", caption="", as_trial=True)
    assert req["data"]["share_mode"] == "TRIAL_REELS_SHARE_TO_FOLLOWERS_IF_LIKED"


def test_graph_container_carries_trial_params():
    import json

    from reforge.publish.providers import GraphPublisher

    pub = GraphPublisher(ig_user_id="178", access_token="tok")
    c = pub.build_container(video_url="https://x/v.mp4", caption="cap", as_trial=True)
    assert c["media_type"] == "REELS"
    assert json.loads(c["trial_params"]) == {"graduation_strategy": "MANUAL"}
    c2 = pub.build_container(video_url="https://x/v.mp4", caption="cap", as_trial=False)
    assert "trial_params" not in c2


def test_graph_graduation_strategy_configurable():
    import json

    from reforge.publish.providers import GraphPublisher, get_publisher

    pub = get_publisher(env={"PUBLISH_PROVIDER": "graph", "PUBLISH_LIVE": "1",
                             "IG_USER_ID": "178", "IG_ACCESS_TOKEN": "tok",
                             "PUBLISH_GRADUATION": "SS_PERFORMANCE"})
    assert isinstance(pub, GraphPublisher)
    c = pub.build_container(video_url="https://x/v.mp4", caption="", as_trial=True)
    assert json.loads(c["trial_params"]) == {"graduation_strategy": "SS_PERFORMANCE"}


# --- get_publisher safety latch ----------------------------------------------

def test_default_env_is_dry_run():
    assert isinstance(get_publisher(env={}), DryRunPublisher)


def test_real_provider_without_live_latch_falls_back_to_dry_run():
    pub = get_publisher(env={"PUBLISH_PROVIDER": "upload_post",
                             "UPLOAD_POST_API_KEY": "k", "UPLOAD_POST_USER": "logan"})
    assert isinstance(pub, DryRunPublisher)


def test_unknown_provider_raises():
    with pytest.raises(ValueError, match="unknown"):
        get_publisher(env={"PUBLISH_PROVIDER": "carrier_pigeon", "PUBLISH_LIVE": "1"})


def test_publish_result_error_default_none():
    res = PublishResult(ok=True, media_id="1", permalink=None, payload={}, dry_run=False)
    assert res.error is None
