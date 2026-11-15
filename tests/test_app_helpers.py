"""Pure-helper tests for the app layer: alternates_json, thumb_url, media URL mappers."""
from reforge.app.queries import alternates_json, proxy_url, reel_video_url, thumb_url
from reforge.llm.base import ClipMoment, SlotPlan
from reforge.match.matcher import MatchedSlot


def _plan(moment_id, score, reason="fits the vibe"):
    moment = ClipMoment(f"library/C{moment_id}.MP4", 0.0, 1.6, "desc", True, "fn", "moderate",
                        moment_id=moment_id, clip_id=moment_id,
                        thumbnail_key=f"thumbs/h{moment_id}/000.jpg")
    return SlotPlan(moment=moment, in_point=1.2, score=score, reason=reason)


# --- alternates_json ---------------------------------------------------------

def test_alternates_json_shape_and_order():
    out = alternates_json([_plan(901, 0.8), _plan(902, 0.6, reason="")])
    assert out == [
        {"moment_id": 901, "clip_key": "library/C901.MP4",
         "thumbnail_key": "thumbs/h901/000.jpg", "score": 0.8,
         "reason": "fits the vibe", "in_point": 1.2},
        {"moment_id": 902, "clip_key": "library/C902.MP4",
         "thumbnail_key": "thumbs/h902/000.jpg", "score": 0.6,
         "reason": "", "in_point": 1.2},
    ]


def test_alternates_json_empty():
    assert alternates_json([]) == []


# --- MatchedSlot gains alternates (additive; default stays empty) ------------

def test_matched_slot_alternates_defaults_empty():
    m = MatchedSlot(shot=object(), plan=_plan(1, 0.9), render=object())
    assert m.alternates == []


def test_matched_slot_alternates_carries_plans():
    alts = [_plan(2, 0.7), _plan(3, 0.5)]
    m = MatchedSlot(shot=object(), plan=_plan(1, 0.9), render=object(), alternates=alts)
    assert [p.moment.moment_id for p in m.alternates] == [2, 3]


# --- thumb_url ----------------------------------------------------------------

def test_thumb_url_strips_known_prefixes(monkeypatch):
    # env-independent: a developer .env may set MEDIA_BASE_URL
    from reforge.app import queries as q
    monkeypatch.setattr(q, "_media_base_url", lambda: "")
    assert thumb_url("thumbs/abc/000.jpg") == "/thumbs/abc/000.jpg"
    assert thumb_url("data/thumbs/abc/000.jpg") == "/thumbs/abc/000.jpg"
    assert thumb_url("abc/000.jpg") == "/thumbs/abc/000.jpg"


def test_thumb_url_none_for_missing():
    assert thumb_url(None) is None
    assert thumb_url("") is None


def test_thumb_url_uses_media_base_url_when_set(monkeypatch):
    from reforge.app import queries as q
    monkeypatch.setattr(q, "_media_base_url", lambda: "https://x.supabase.co/storage/v1/object/public/media")
    assert q.thumb_url("data/thumbs/abc/001.jpg") == \
        "https://x.supabase.co/storage/v1/object/public/media/thumbs/abc/001.jpg"


def test_thumb_url_local_mount_when_no_base(monkeypatch):
    from reforge.app import queries as q
    monkeypatch.setattr(q, "_media_base_url", lambda: "")
    assert q.thumb_url("thumbs/abc/001.jpg") == "/thumbs/abc/001.jpg"


# --- reel_video_url (rendered reel mp4 -> Supabase when set, else /media/output) --

def _local(monkeypatch):
    from reforge.app import queries as q
    monkeypatch.setattr(q, "_media_base_url", lambda: "")


def test_reel_video_url_maps_output_key(monkeypatch):
    _local(monkeypatch)
    assert reel_video_url("output/C0deSampl3x_recreation.mp4") == \
        "/media/output/C0deSampl3x_recreation.mp4"


def test_reel_video_url_tolerates_absolute_paths(monkeypatch):
    _local(monkeypatch)
    assert reel_video_url("/Users/me/reforge/output/x_recreation.mp4") == \
        "/media/output/x_recreation.mp4"


def test_reel_video_url_bare_filename(monkeypatch):
    _local(monkeypatch)
    assert reel_video_url("x.mp4") == "/media/output/x.mp4"


def test_reel_video_url_uses_media_base_url_when_set(monkeypatch):
    from reforge.app import queries as q
    monkeypatch.setattr(q, "_media_base_url", lambda: "https://x.supabase.co/o/media")
    assert q.reel_video_url("output/x_recreation.mp4") == \
        "https://x.supabase.co/o/media/output/x_recreation.mp4"


def test_reel_video_url_none_for_falsy():
    assert reel_video_url(None) is None
    assert reel_video_url("") is None


# --- proxy_url (library storage_key -> Supabase when set, else /media/proxies) ---

def test_proxy_url_strips_library_and_swaps_extension(monkeypatch):
    _local(monkeypatch)
    assert proxy_url("library/Shorts/Gym/Arms/C1401.MP4") == \
        "/media/proxies/Shorts/Gym/Arms/C1401.mp4"


def test_proxy_url_mov_becomes_mp4(monkeypatch):
    _local(monkeypatch)
    assert proxy_url("library/Beach/IMG_0042.MOV") == "/media/proxies/Beach/IMG_0042.mp4"


def test_proxy_url_escapes_spaces_keeps_subdirs(monkeypatch):
    _local(monkeypatch)
    assert proxy_url("library/B Roll/Sun Rise.MP4") == "/media/proxies/B%20Roll/Sun%20Rise.mp4"


def test_proxy_url_uses_media_base_url_when_set(monkeypatch):
    from reforge.app import queries as q
    monkeypatch.setattr(q, "_media_base_url", lambda: "https://x.supabase.co/o/media")
    assert q.proxy_url("library/Shorts/Gym/C1401.MP4") == \
        "https://x.supabase.co/o/media/proxies/Shorts/Gym/C1401.mp4"


def test_proxy_url_none_for_falsy_or_non_library():
    assert proxy_url(None) is None
    assert proxy_url("") is None
    assert proxy_url("thumbs/abc/000.jpg") is None
    assert proxy_url("library/") is None
