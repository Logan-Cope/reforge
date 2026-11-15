"""DEMO_READ_ONLY: the public demo can browse every page but never write.

The write-route list is discovered from the app itself, so a write route added later
is covered automatically; the expected set below also pins it, so the test can't pass
by finding nothing.
"""
import re

import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from reforge.app.fixtures import FixtureQueries
from reforge.app.main import create_app

SAFE = {"GET", "HEAD", "OPTIONS"}

EXPECTED_WRITES = {
    ("POST", "/slots/{slot_id}/keep"),
    ("POST", "/slots/{slot_id}/weak"),
    ("POST", "/slots/{slot_id}/swap"),
    ("POST", "/slots/{slot_id}/undo"),
    ("POST", "/slots/{slot_id}/flag"),
    ("POST", "/slots/{slot_id}/repick"),
    ("POST", "/reels/{reel_id}/grade"),
    ("POST", "/reels/{reel_id}/rerender"),
    ("POST", "/reels/{reel_id}/schedule"),
    ("POST", "/reels/{reel_id}/schedule/cancel"),
    ("POST", "/runs"),
    ("POST", "/moments/{moment_id}/rate"),
}

# Form bodies that would pass validation, so a 403 can only come from the gate.
FORM = {"grade": "post_worthy", "moment_id": "1", "reason": "too dark", "note": "",
        "label": "1", "reference_key": "reference/night-run.mp4",
        "date": "2030-01-01", "time": "09:00", "as_trial": "", "caption": "x"}


class RecordingQueries(FixtureQueries):
    """Fixture data plus a record of every method a request reached."""

    def __init__(self):
        super().__init__()
        self.touched = []

    def __getattribute__(self, name):
        attr = object.__getattribute__(self, name)
        if callable(attr) and not name.startswith("_") and name != "touched":
            object.__getattribute__(self, "touched").append(name)
        return attr


def iter_api_routes(routes):
    """Every APIRoute, including ones inside included routers (FastAPI >= 0.13x wraps
    them in an included-router object instead of flattening them into app.routes)."""
    for r in routes:
        if isinstance(r, APIRoute):
            yield r
        sub = getattr(r, "original_router", None)
        if sub is not None:
            yield from iter_api_routes(sub.routes)


def write_routes(app):
    return {(m, r.path) for r in iter_api_routes(app.routes) for m in r.methods - SAFE}


def concrete(path, reel_id, slot_id):
    path = path.replace("{reel_id}", str(reel_id)).replace("{slot_id}", str(slot_id))
    return re.sub(r"\{[^}]+\}", "1", path)


@pytest.fixture
def ro(monkeypatch):
    monkeypatch.setenv("DEMO_READ_ONLY", "1")
    monkeypatch.delenv("INVITE_CODE", raising=False)
    q = RecordingQueries()
    return TestClient(create_app(queries=q)), q


@pytest.fixture
def ids():
    q = FixtureQueries()
    reel = q.list_reels()[0]["id"]
    slot = q.reel_detail(reel)["slots"][0]["id"]
    return reel, slot


def test_every_write_route_is_discovered():
    app = create_app(queries=FixtureQueries())
    assert write_routes(app) == EXPECTED_WRITES


def test_every_write_route_returns_403_and_touches_no_data(ro, ids):
    client, q = ro
    for method, path in sorted(write_routes(client.app)):
        q.touched.clear()
        url = concrete(path, *ids)
        res = client.request(method, url, data=FORM)
        assert res.status_code == 403, f"{method} {url} -> {res.status_code}"
        assert q.touched == [], f"{method} {url} reached the data layer: {q.touched}"


@pytest.mark.parametrize("method", ["POST", "PUT", "PATCH", "DELETE"])
def test_unsafe_methods_are_refused_on_any_path(ro, method):
    client, _ = ro
    for url in ("/", "/reels/1", "/healthz", "/no-such-route"):
        assert client.request(method, url).status_code == 403


def test_pages_still_render_read_only(ro, ids):
    client, _ = ro
    reel, slot = ids
    for url in ("/", f"/reels/{reel}", "/posted", "/curate", f"/reels/{reel}/schedule",
                f"/reels/{reel}/schedule?open=1", "/healthz"):
        res = client.get(url)
        assert res.status_code == 200, url


def test_read_only_pages_ship_the_lock_script(ro, ids):
    client, _ = ro
    html = client.get(f"/reels/{ids[0]}").text
    assert "/static/read_only.js" in html
    assert 'class="read-only"' in html
    assert client.get("/static/read_only.js").status_code == 200


def test_writes_work_when_the_flag_is_off(monkeypatch):
    """The flag is the only switch: without it the same route reaches the data layer."""
    monkeypatch.delenv("DEMO_READ_ONLY", raising=False)
    monkeypatch.delenv("INVITE_CODE", raising=False)
    q = RecordingQueries()
    client = TestClient(create_app(queries=q), raise_server_exceptions=False)
    reel = q.list_reels()[0]["id"]
    q.touched.clear()
    client.post(f"/reels/{reel}/grade", data={"grade": "post_worthy"})
    assert "grade_reel" in q.touched  # reached the seam (which then refuses fixture writes)
    assert "/static/read_only.js" not in client.get("/").text


@pytest.mark.parametrize("value", ["0", "", "false", "no", "off"])
def test_falsey_values_leave_writes_on(monkeypatch, value):
    monkeypatch.setenv("DEMO_READ_ONLY", value)
    monkeypatch.delenv("INVITE_CODE", raising=False)
    q = RecordingQueries()
    client = TestClient(create_app(queries=q), raise_server_exceptions=False)
    res = client.post("/runs", data={"reference_key": "nope"})
    assert res.status_code == 400  # reached the route's own allowlist, not the gate


def test_fixture_seam_refuses_writes_on_its_own():
    q = FixtureQueries()
    with pytest.raises(PermissionError):
        q.grade_reel(1, "post_worthy")
    with pytest.raises(PermissionError):
        q.create_scheduled_post(1, None)
    with pytest.raises(PermissionError):
        q.create_run_request("reference/x.mp4")
