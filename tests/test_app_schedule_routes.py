"""Route tests for the trial-reel schedule surface (FakeQueries injected, no DB).

Covers the lazy-loaded schedule partial (form only on an approved reel), scheduling
(ET input stored as UTC, as_trial passed through), the guards (past time, double
schedule, unknown reel), and cancel.
"""
from datetime import datetime, timezone

from fastapi.testclient import TestClient

from reforge.app.main import create_app


class FakeQueries:
    def __init__(self, approved=True, sched=None, published=False):
        self.calls = []
        self.approved = approved
        self.published = published
        self.sched = sched  # latest_scheduled_post() result

    def ping(self):
        return True

    def reel_detail(self, reel_id):
        if reel_id == 999:
            return None
        return {"reel": {"id": reel_id, "ref": "reference/grace.mp4", "created": "Jul 20",
                         "approved": self.approved, "output_key": "output/grace_run3.mp4",
                         "playable": False, "cut_dirty": False, "grade": None,
                         "caption": "iron sharpens iron",
                         "published": self.published, "posted": "Jul 22",
                         "is_trial": True},
                "slots": []}

    def latest_scheduled_post(self, reel_id):
        self.calls.append(("latest_scheduled_post", reel_id))
        return self.sched

    def create_scheduled_post(self, reel_id, scheduled_at, as_trial=True, caption=""):
        self.calls.append(("create_scheduled_post", reel_id, scheduled_at, as_trial,
                           caption))
        self.sched = {"id": 31, "status": "scheduled", "scheduled_at": scheduled_at,
                      "as_trial": as_trial, "provider": None, "error": None,
                      "published_at": None}
        return 31

    def posted_reels(self):
        self.calls.append(("posted_reels",))
        return [{"id": 12, "ref": "reference/grace.mp4", "published": False,
                 "posted": "", "is_trial": True, "sched_status": "scheduled",
                 "scheduled_at": datetime(2026, 7, 23, 13, 30, tzinfo=timezone.utc),
                 "sched_as_trial": True, "thumbnail_key": None},
                {"id": 10, "ref": "reference/DISCIPLINE.mp4", "published": True,
                 "posted": "Jul 22", "is_trial": True, "sched_status": "published",
                 "scheduled_at": None, "sched_as_trial": True, "thumbnail_key": None}]

    def cancel_scheduled_post(self, post_id):
        self.calls.append(("cancel_scheduled_post", post_id))
        self.sched = {**self.sched, "status": "canceled"}
        return True


def _client(q):
    return TestClient(create_app(queries=q)), q


def _sched(status="scheduled", as_trial=True):
    return {"id": 31, "status": status,
            "scheduled_at": datetime(2026, 7, 22, 18, 0, tzinfo=timezone.utc),
            "as_trial": as_trial, "provider": None, "error": None,
            "published_at": None}


# --- GET (the lazy-loaded partial) -------------------------------------------

def test_schedule_collapsed_state_is_a_button_not_a_form():
    client, q = _client(FakeQueries(approved=True))
    r = client.get("/reels/5/schedule")
    assert r.status_code == 200
    assert "Schedule trial reel" in r.text
    assert 'name="date"' not in r.text  # panel only appears after the tap


def test_schedule_open_form_has_pickers_and_prefilled_caption():
    client, q = _client(FakeQueries(approved=True))
    r = client.get("/reels/5/schedule?open=1")
    assert 'type="date"' in r.text and 'type="time"' in r.text
    assert 'value="iron sharpens iron"' in r.text  # generated caption pre-filled
    assert "Back" in r.text


def test_schedule_partial_hides_form_until_approved():
    client, q = _client(FakeQueries(approved=False))
    r = client.get("/reels/5/schedule")
    assert r.status_code == 200
    assert "Schedule trial reel" not in r.text
    r2 = client.get("/reels/5/schedule?open=1")
    assert 'name="date"' not in r2.text


def test_schedule_partial_shows_active_schedule_with_cancel():
    client, q = _client(FakeQueries(sched=_sched()))
    r = client.get("/reels/5/schedule")
    assert "Trial reel" in r.text and "Cancel" in r.text
    assert "2:00 PM ET" in r.text  # 18:00 UTC July == 2 PM EDT


def test_schedule_partial_dry_run_done_says_nothing_posted_and_offers_reschedule():
    client, q = _client(FakeQueries(sched=_sched(status="dry_run_done")))
    r = client.get("/reels/5/schedule")
    assert "nothing posted" in r.text.lower()
    assert "Reschedule" in r.text  # a dry run is not terminal - the real post is next


def test_schedule_partial_404_unknown_reel():
    client, q = _client(FakeQueries())
    assert client.get("/reels/999/schedule").status_code == 404


# --- POST /schedule -----------------------------------------------------------

def test_schedule_posts_utc_and_trial_flag():
    client, q = _client(FakeQueries(approved=True))
    r = client.post("/reels/5/schedule", data={"date": "2027-01-05", "time": "09:30",
                                               "as_trial": "on"})
    assert r.status_code == 200
    call = next(c for c in q.calls if c[0] == "create_scheduled_post")
    _, reel_id, at, as_trial, _caption = call
    assert reel_id == 5 and as_trial is True
    assert at.tzinfo == timezone.utc
    assert at == datetime(2027, 1, 5, 14, 30, tzinfo=timezone.utc)  # 9:30 EST -> UTC
    assert "scheduled" in r.text.lower()


def test_schedule_passes_typed_caption():
    client, q = _client(FakeQueries(approved=True))
    client.post("/reels/5/schedule", data={"date": "2027-01-05", "time": "09:30",
                                           "as_trial": "on",
                                           "caption": "  God first, then the gym.  "})
    call = next(c for c in q.calls if c[0] == "create_scheduled_post")
    assert call[4] == "God first, then the gym."  # stripped, passed through


def test_schedule_without_trial_checkbox_is_regular_reel():
    client, q = _client(FakeQueries(approved=True))
    client.post("/reels/5/schedule", data={"date": "2027-01-05", "time": "09:30"})
    call = next(c for c in q.calls if c[0] == "create_scheduled_post")
    assert call[3] is False


def test_schedule_rejects_past_time():
    client, q = _client(FakeQueries(approved=True))
    r = client.post("/reels/5/schedule", data={"date": "2020-01-01", "time": "09:30"})
    assert r.status_code == 400


def test_schedule_rejects_unapproved_reel():
    client, q = _client(FakeQueries(approved=False))
    r = client.post("/reels/5/schedule", data={"date": "2027-01-05", "time": "09:30"})
    assert r.status_code == 400
    assert not any(c[0] == "create_scheduled_post" for c in q.calls)


def test_schedule_rejects_double_schedule():
    client, q = _client(FakeQueries(sched=_sched()))
    r = client.post("/reels/5/schedule", data={"date": "2027-01-05", "time": "09:30"})
    assert r.status_code == 400


def test_schedule_404_unknown_reel():
    client, q = _client(FakeQueries())
    r = client.post("/reels/999/schedule", data={"date": "2027-01-05", "time": "09:30"})
    assert r.status_code == 404


# --- POST /schedule/cancel ------------------------------------------------------

def test_cancel_active_schedule():
    client, q = _client(FakeQueries(sched=_sched()))
    r = client.post("/reels/5/schedule/cancel")
    assert r.status_code == 200
    assert ("cancel_scheduled_post", 31) in q.calls
    assert "Reschedule" in r.text  # back to a (re)schedule button


def test_cancel_without_active_schedule_400():
    client, q = _client(FakeQueries(sched=None))
    assert client.post("/reels/5/schedule/cancel").status_code == 400
    client2, q2 = _client(FakeQueries(sched=_sched(status="published")))
    assert client2.post("/reels/5/schedule/cancel").status_code == 400


# --- published reels are closed for scheduling ---------------------------------

def test_published_reel_shows_no_schedule_button():
    client, q = _client(FakeQueries(published=True, sched=None))
    r = client.get("/reels/5/schedule")
    assert "Schedule trial reel" not in r.text and "Reschedule" not in r.text
    r2 = client.get("/reels/5/schedule?open=1")
    assert 'name="date"' not in r2.text


def test_published_reel_rejects_schedule_post():
    client, q = _client(FakeQueries(published=True, sched=None))
    r = client.post("/reels/5/schedule", data={"date": "2027-01-05", "time": "09:30"})
    assert r.status_code == 400
    assert "already posted" in r.json()["detail"]
    assert not any(c[0] == "create_scheduled_post" for c in q.calls)


# --- the /posted page ----------------------------------------------------------

def test_posted_page_groups_scheduled_and_live():
    client, q = _client(FakeQueries())
    r = client.get("/posted")
    assert r.status_code == 200
    assert "Scheduled" in r.text and "Live on Instagram" in r.text
    assert "Reel 12" in r.text and "Reel 10" in r.text
    assert "9:30 AM ET" in r.text          # 13:30 UTC July == 9:30 AM EDT
    assert "Trial live" in r.text
