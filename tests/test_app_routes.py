"""Route tests for the review/curation app: FakeQueries injected, no DB.

Covers: dashboard + reel review render, keep/swap/rate/approve POSTs (right fake method
called AND a taste event captured via the queries seam), the run queue (POST /runs +
dashboard card), /healthz, and the invite gate. Taste writes go through
``insert_taste_event`` so the serverless app never depends on a writable filesystem;
the local-file mirror is covered in test_taste.py.
"""
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from reforge.app.main import create_app


def _slot(verdict=None, note=None, cut_dirty=False):
    return {
        "id": 71, "reel_id": 5, "idx": 3, "start_s": 2.6, "end_s": 3.5,
        "ref_description": "build beat: lifting", "reason": "matches the grind vibe",
        "human_verdict": verdict, "human_note": note,
        "chosen_moment_id": 900, "swapped_moment_id": None,
        "candidates": [
            {"moment_id": 901, "clip_key": "library/C901.MP4",
             "thumbnail_key": "thumbs/h901/000.jpg", "score": 0.8,
             "reason": "runner-up: darker gym angle", "in_point": 1.2},
        ],
        "thumbnail_key": "thumbs/h900/000.jpg",
        "description": "dark gym dumbbell press", "quality_score": 0.62,
        "clip_key": "library/Shorts/Gym/Arms/C1401.MP4",
        "params": {"in_point": 4.2, "zoom": 1.0},
        "moment_start": 0.0, "moment_end": 10.0,
        "effective_in": 4.2,  # unswapped -> the original params' in_point
        # verdict-write extras (ride on the single-round-trip CTE)
        "total": 8, "decided": 3, "cut_dirty": cut_dirty,
    }


def _swapped_slot(note=None, moment_id=901):
    """A slot swapped to candidate 901: effective_in is the CANDIDATE's in_point."""
    s = _slot(verdict="swapped", note=note, cut_dirty=True)  # a swap dirties the cut
    s["swapped_moment_id"] = moment_id
    s["effective_in"] = 1.2
    return s


class FakeQueries:
    def __init__(self):
        self.calls = []
        self.taste_events = []
        self.rerender = None      # latest_rerender() result; tests set per-scenario
        self.repick = None        # latest_repick() result; tests set per-scenario
        self.dirty = False        # reel_cut_dirty() / reel 5's cut_dirty
        self.slot_verdict = "flagged"   # slot_detail() verdict (repick routes)
        self.slot_note = "too dark: face covered"

    def ping(self):
        return True

    def list_reels(self, limit=20):
        return [{"id": 5, "ref": "reference/DISCIPLINE.mp4", "created": "Jul 01",
                 "approved": False, "slot_count": 17, "reviewed_count": 2}]

    def counts(self):
        return {"reels": 5, "approved": 1, "unrated_moments": 42, "rated_moments": 7}

    def reel_detail(self, reel_id):
        if reel_id == 999:
            return None
        if reel_id == 8:  # a reel whose rendered mp4 exists locally -> playable
            return {"reel": {"id": reel_id, "ref": "reference/DISCIPLINE.mp4",
                             "created": "Jul 01", "approved": False,
                             "output_key": "output/C0deSampl3x_recreation.mp4",
                             "playable": True, "cut_dirty": False},
                    "slots": [_slot()]}
        if reel_id == 7:  # a swapped-but-not-rerendered reel -> cut dirty, button on
            swapped = _swapped_slot()
            return {"reel": {"id": reel_id, "ref": "reference/DISCIPLINE.mp4",
                             "created": "Jul 01", "approved": False, "output_key": None,
                             "playable": False, "cut_dirty": True},
                    "slots": [swapped]}
        return {"reel": {"id": reel_id, "ref": "reference/DISCIPLINE.mp4",
                         "created": "Jul 01", "approved": False, "output_key": None,
                         "playable": False, "cut_dirty": self.dirty},
                "slots": [_slot()]}

    def keep_slot(self, slot_id, note=""):
        self.calls.append(("keep_slot", slot_id, note))
        return _slot(verdict="keep", note=note or None)

    def weak_keep_slot(self, slot_id, note=""):
        self.calls.append(("weak_keep_slot", slot_id, note))
        return _slot(verdict="weak", note=note or None)

    def swap_slot(self, slot_id, moment_id, note=""):
        self.calls.append(("swap_slot", slot_id, moment_id, note))
        return _swapped_slot(note=note or None, moment_id=moment_id)

    def flag_slot(self, slot_id, reason=""):
        self.calls.append(("flag_slot", slot_id, reason))
        return _slot(verdict="flagged", note=reason or None)

    def undo_slot(self, slot_id):
        self.calls.append(("undo_slot", slot_id))
        return None if slot_id == 999 else _slot()

    def latest_rerender(self, reel_id):
        self.calls.append(("latest_rerender", reel_id))
        return self.rerender

    def latest_repick(self, slot_id):
        self.calls.append(("latest_repick", slot_id))
        return self.repick

    def slot_detail(self, slot_id):
        self.calls.append(("slot_detail", slot_id))
        if slot_id == 999:
            return None
        return _slot(verdict=self.slot_verdict, note=self.slot_note)

    def decided_counts(self, reel_id):
        self.calls.append(("decided_counts", reel_id))
        return {"decided": 3, "total": 8}

    def reel_cut_dirty(self, reel_id):
        self.calls.append(("reel_cut_dirty", reel_id))
        return self.dirty

    def grade_reel(self, reel_id, grade):
        self.calls.append(("grade_reel", reel_id, grade))
        if grade not in ("post_worthy", "would_test", "not_it"):
            raise ValueError(f"unknown grade: {grade}")
        approved = grade in ("post_worthy", "would_test")
        accepted = [{"beat": i, "moment_id": 800 + i} for i in range(15)] if approved else []
        return {"id": reel_id, "grade": grade, "approved": approved, "slots": 17,
                "kept": 0, "swapped": 2, "flagged": 0, "accepted": accepted}

    def curation_batch(self, limit=20):
        return [{"id": 900, "thumbnail_key": "thumbs/h900/000.jpg",
                 "description": "dockside pull-up", "quality_score": 0.51}][:limit]

    def rate_moment(self, moment_id, label):
        self.calls.append(("rate_moment", moment_id, label))

    # -- serverless taste + run queue -----------------------------------------

    def insert_taste_event(self, entry):
        self.taste_events.append(entry)

    def create_run_request(self, reference_key):
        self.calls.append(("create_run_request", reference_key))
        return 31

    def list_run_requests(self, limit=10):
        return [
            {"id": 30, "reference_key": "reference/grace.mp4", "status": "queued",
             "created": "Jul 10 08:00", "started": "", "finished": "",
             "reel_id": None, "error": None},
            {"id": 29, "reference_key": "reference/DISCIPLINE.mp4", "status": "done",
             "created": "Jul 09 10:00", "started": "Jul 09 10:01", "finished": "Jul 09 10:20",
             "reel_id": 12, "error": None},
            {"id": 28, "reference_key": "reference/Bible-gym.mp4", "status": "error",
             "created": "Jul 08 09:00", "started": "Jul 08 09:01", "finished": "Jul 08 09:02",
             "reel_id": None, "error": "ffmpeg exploded"},
        ][:limit]

    def available_references(self):
        return ["reference/Bible-gym.mp4", "reference/DISCIPLINE.mp4", "reference/grace.mp4"]


@pytest.fixture
def taste_dir(tmp_path, monkeypatch):
    """Point the local taste-log mirror at a tmp dir so tests never touch taste/."""
    import reforge.taste as taste
    monkeypatch.setattr(taste, "settings", SimpleNamespace(taste_dir=str(tmp_path)))
    return tmp_path


@pytest.fixture
def fq():
    return FakeQueries()


@pytest.fixture
def client(fq, monkeypatch, taste_dir):
    monkeypatch.delenv("INVITE_CODE", raising=False)
    return TestClient(create_app(queries=fq))


# --- pages --------------------------------------------------------------------

def test_dashboard_renders_reels_and_curation_card(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "Reel 5" in r.text
    assert "DISCIPLINE.mp4" in r.text
    assert "42 unrated" in r.text


def test_dashboard_renders_new_run_form_and_only_in_flight_runs(client):
    """The dashboard leads with the poster wall; starting a run is a footer control.

    Finished and errored runs are deliberately NOT listed here - the dashboard
    answers "what needs my review", and a standing pile of old error rows was
    noise (and the first thing a visitor saw). Only in-flight runs show.
    """
    r = client.get("/")
    assert r.status_code == 200
    assert 'hx-post="/runs"' in r.text                        # the New run form survives
    assert 'name="reference_key"' in r.text
    assert 'value="reference/grace.mp4"' in r.text            # gallery options
    assert 'value="reference/Bible-gym.mp4"' in r.text

    assert "Run 30" in r.text                                 # queued run still listed
    assert "Queued" in r.text

    assert "Run 29" not in r.text                             # done run hidden
    assert "reel 12" not in r.text
    assert "Run 28" not in r.text                             # errored run hidden
    assert "ffmpeg exploded" not in r.text

    # the run queue no longer sits above the reels
    assert "Run queue" not in r.text
    assert r.text.index("poster-wall") < r.text.index("new-run")


def test_reel_page_renders_slots_and_candidates(client):
    r = client.get("/reels/5")
    assert r.status_code == 200
    assert "Beat 3" in r.text
    assert "dark gym dumbbell press" in r.text
    assert "matches the grind vibe" in r.text
    assert "runner-up: darker gym angle" in r.text          # swap candidate rendered
    assert "/thumbs/h900/000.jpg" in r.text                  # thumb_url filter applied
    assert 'hx-post="/slots/71/keep"' in r.text


def test_reel_page_404_for_missing_reel(client):
    assert client.get("/reels/999").status_code == 404


def test_reel_page_has_player_when_playable(client):
    r = client.get("/reels/8")
    assert r.status_code == 200
    assert 'id="reel-player"' in r.text
    assert "/media/output/C0deSampl3x_recreation.mp4" in r.text
    assert 'data-start="2.6"' in r.text                      # beat rail chip carries timing


def test_reel_page_no_player_when_not_playable(client):
    r = client.get("/reels/5")
    assert r.status_code == 200
    assert 'id="reel-player"' not in r.text


def test_reel_page_renders_proxy_video_for_chosen_moment(client):
    r = client.get("/reels/5")
    assert r.status_code == 200
    assert "/media/proxies/Shorts/Gym/Arms/C1401.mp4" in r.text  # lazy beat-src video
    assert 'data-in="4.2"' in r.text                             # in_point from slot params
    assert "/media/proxies/C901.mp4" in r.text                   # candidate proxy playable too
    assert 'hx-post="/slots/71/flag"' in r.text                  # flag reason chips wired


def test_reel_page_wires_instant_feedback_and_flag_notes(client):
    """Verdict buttons disable themselves during the request; flag chips carry the note."""
    r = client.get("/reels/5")
    assert r.status_code == 200
    keep = r.text.split('hx-post="/slots/71/keep"')[1][:200]
    assert 'hx-disabled-elt="this"' in keep                       # no double-taps on Keep
    flag = r.text.split('hx-post="/slots/71/flag"')[1][:220]
    assert 'hx-include="closest form"' in flag                    # flag submits the note too
    assert 'hx-disabled-elt="this"' in flag
    swap = r.text.split('hx-post="/slots/71/swap"')[1][:220]
    assert 'hx-disabled-elt="this"' in swap


def test_curate_page_renders_a_card(client):
    r = client.get("/curate")
    assert r.status_code == 200
    assert "dockside pull-up" in r.text
    assert 'hx-post="/moments/900/rate"' in r.text


# --- actions write the DB (fake) AND the taste event via the queries seam --------

def test_keep_slot_calls_fake_and_records_taste_event(client, fq):
    r = client.post("/slots/71/keep", data={"note": "solid pick"})
    assert r.status_code == 200
    assert ("keep_slot", 71, "solid pick") in fq.calls
    assert "Kept" in r.text                                  # verdict chip in the partial
    (entry,) = fq.taste_events
    assert entry["event"] == "human_keep"
    assert entry["run"] == 5 and entry["beat"] == 3
    assert entry["reason"] == "solid pick"


def test_weak_keep_records_a_soft_positive(client, fq):
    r = client.post("/slots/71/weak", data={"note": "ok not great"})
    assert r.status_code == 200
    assert ("weak_keep_slot", 71, "ok not great") in fq.calls
    assert "Could be better" in r.text                        # distinct verdict chip
    (entry,) = fq.taste_events
    assert entry["event"] == "human_keep" and entry["strength"] == "weak"
    assert entry["run"] == 5 and entry["beat"] == 3 and entry["reason"] == "ok not great"


def test_swap_slot_calls_fake_and_records_taste_event(client, fq):
    r = client.post("/slots/71/swap", data={"moment_id": "901", "note": "too dark"})
    assert r.status_code == 200
    assert ("swap_slot", 71, 901, "too dark") in fq.calls
    assert "Swapped" in r.text
    (entry,) = fq.taste_events
    assert entry["event"] == "human_swap"
    assert entry["run"] == 5 and entry["beat"] == 3
    assert entry["reason"] == "too dark"
    assert entry["old"] == 900 and entry["new"] == 901


def test_flag_slot_calls_fake_and_records_taste_event(client, fq):
    r = client.post("/slots/71/flag", data={"reason": "too dark"})
    assert r.status_code == 200
    assert ("flag_slot", 71, "too dark") in fq.calls
    assert "Flagged" in r.text                               # new verdict chip in the partial
    (entry,) = fq.taste_events
    assert entry["event"] == "human_flag"
    assert entry["run"] == 5 and entry["beat"] == 3
    assert entry["rule"] == "too dark"
    assert entry["target"] == "clip"                         # too dark blames the footage
    assert "note" not in entry                               # empty note omitted, not None


def test_flag_target_classifies_by_what_it_blames(client, fq):
    # A good clip that's wrong here (fit) or has a render issue must NOT read as a bad
    # clip - distillation uses target to protect the footage.
    for reason, target in [("too dark", "clip"), ("off-vibe", "fit"),
                           ("caption placement", "render")]:
        fq.taste_events.clear()
        client.post("/slots/71/flag", data={"reason": reason})
        (entry,) = fq.taste_events
        assert entry["target"] == target


def test_flag_target_helper_defaults_unknown_to_other():
    from reforge.app.routes import flag_target
    assert flag_target("too dark") == "clip"
    assert flag_target("off-vibe") == "fit"
    assert flag_target("caption placement") == "render"
    assert flag_target("other") == "other"


def test_flag_slot_with_note_persists_both(client, fq):
    """A note typed before tapping a flag chip must not be lost (was a data-loss bug)."""
    r = client.post("/slots/71/flag", data={"reason": "too dark", "note": "blurry face too"})
    assert r.status_code == 200
    assert ("flag_slot", 71, "too dark: blurry face too") in fq.calls
    assert "too dark: blurry face too" in r.text             # combined note in the partial
    (entry,) = fq.taste_events
    assert entry["event"] == "human_flag"
    assert entry["rule"] == "too dark"
    assert entry["note"] == "blurry face too"


def test_flag_note_composition():
    from reforge.app.routes import flag_note
    assert flag_note("too dark", "") == "too dark"
    assert flag_note("too dark", "blurry") == "too dark: blurry"


def test_grade_reel_positive_records_accepted_and_endorses_unflagged(client, fq):
    r = client.post("/reels/5/grade", data={"grade": "post_worthy"})
    assert r.status_code == 200
    assert ("grade_reel", 5, "post_worthy") in fq.calls
    assert "Post-worthy" in r.text and "15 accepted" in r.text  # unflagged beats endorsed
    (entry,) = fq.taste_events
    assert entry["event"] == "review_summary"
    assert entry["run"] == 5 and entry["grade"] == "post_worthy" and entry["approved"] is True
    assert len(entry["accepted"]) == 15  # the whole reel learned from in one tap


def test_grade_reel_not_it_rejects_concept_no_endorsement(client, fq):
    r = client.post("/reels/5/grade", data={"grade": "not_it"})
    assert r.status_code == 200
    assert ("grade_reel", 5, "not_it") in fq.calls
    (entry,) = fq.taste_events
    assert entry["event"] == "reel_reject"
    assert entry["approved"] is False and entry["accepted"] == []


def test_grade_reel_rejects_unknown_grade(client, fq):
    r = client.post("/reels/5/grade", data={"grade": "bogus"})
    assert r.status_code == 400


def test_rate_moment_calls_fake_records_and_loads_next(client, fq):
    r = client.post("/moments/900/rate", data={"label": "1"})
    assert r.status_code == 200
    assert ("rate_moment", 900, 1) in fq.calls
    assert "dockside pull-up" in r.text                      # next card auto-loaded
    (entry,) = fq.taste_events
    assert entry["event"] == "clip_label"
    assert entry["moment_id"] == 900 and entry["label"] == 1


# --- run queue -------------------------------------------------------------------

def test_post_runs_queues_and_renders_partial(client, fq):
    r = client.post("/runs", data={"reference_key": "reference/grace.mp4"})
    assert r.status_code == 200
    assert ("create_run_request", "reference/grace.mp4") in fq.calls
    assert "Run 31" in r.text                                # id from create_run_request
    assert "grace.mp4" in r.text
    assert "Queued" in r.text                                # fresh request shows queued chip


# --- undo a verdict ----------------------------------------------------------------

def test_undo_slot_calls_fake_and_records_taste_event(client, fq):
    r = client.post("/slots/71/undo")
    assert r.status_code == 200
    assert ("undo_slot", 71) in fq.calls
    assert 'hx-post="/slots/71/keep"' in r.text              # verdict buttons are back
    entry = next(e for e in fq.taste_events if e["event"] == "human_undo")
    assert entry["run"] == 5 and entry["beat"] == 3          # downstream can discount


def test_undo_404_for_missing_slot(client, fq):
    assert client.post("/slots/999/undo").status_code == 404
    assert fq.taste_events == []


def test_decided_card_shows_undo_button(client):
    r = client.get("/reels/7")                               # reel 7 has a swapped slot
    assert 'hx-post="/slots/71/undo"' in r.text


def test_undecided_card_has_no_undo_button(client):
    r = client.get("/reels/5")
    assert 'hx-post="/slots/71/undo"' not in r.text


def test_keep_response_offers_undo(client):
    r = client.post("/slots/71/keep")
    assert 'hx-post="/slots/71/undo"' in r.text


# --- post-swap card previews the SWAPPED moment's seconds ---------------------------

def test_swapped_card_uses_effective_in_not_original_params(client):
    """The beat proxy must preview the swapped candidate's in_point (1.2), not the
    original chosen moment's params.in_point (4.2) - that showed the wrong seconds."""
    r = client.get("/reels/7")
    assert r.status_code == 200
    assert 'data-in="1.2"' in r.text
    assert 'data-in="4.2"' not in r.text


def test_swap_response_card_uses_effective_in(client):
    r = client.post("/slots/71/swap", data={"moment_id": "901"})
    assert 'data-in="1.2"' in r.text


# --- cut freshness: does the video match the picks? --------------------------------

def test_reel_page_offers_rerender_when_cut_is_dirty(client):
    r = client.get("/reels/7")                                # dirty: swap not rendered yet
    assert r.status_code == 200
    assert "Video out of date - re-render" in r.text
    assert 'hx-post="/reels/7/rerender"' in r.text
    assert "Video matches the picks" not in r.text


def test_reel_page_shows_quiet_match_line_when_cut_is_clean(client):
    """Reels 8/9 live bug: a completed re-render must NOT keep offering the button."""
    r = client.get("/reels/5")                                # clean cut
    assert r.status_code == 200
    assert "Re-render" not in r.text
    assert "Video matches the picks" in r.text


def test_post_rerender_queues_and_renders_banner(client, fq):
    r = client.post("/reels/5/rerender")
    assert r.status_code == 200
    assert ("create_run_request", "rerender:5") in fq.calls
    assert "Re-render queued" in r.text
    assert 'hx-get="/reels/5/rerender/status"' in r.text      # banner polls for progress
    assert 'hx-trigger="every 4s"' in r.text
    (entry,) = fq.taste_events
    assert entry["event"] == "rerender_requested"
    assert entry["run"] == 5


def test_post_rerender_404_for_missing_reel(client, fq):
    assert client.post("/reels/999/rerender").status_code == 404
    assert not any(c[0] == "create_run_request" for c in fq.calls)


def test_swap_response_includes_oob_rerender_area(client, fq):
    """The re-render button must appear live after a swap, not after a refresh."""
    r = client.post("/slots/71/swap", data={"moment_id": "901"})
    assert r.status_code == 200
    assert 'id="rerender-area"' in r.text
    assert 'hx-swap-oob="true"' in r.text
    assert "Video out of date - re-render" in r.text          # swap dirties the cut


def test_undo_response_includes_oob_rerender_area(client, fq):
    r = client.post("/slots/71/undo")                         # undo fake: cut stays clean
    assert 'id="rerender-area"' in r.text
    assert 'hx-swap-oob="true"' in r.text
    assert "Video out of date - re-render" not in r.text      # button retracts
    assert "Video matches the picks" in r.text


def test_reel_page_shows_polling_banner_while_rerender_in_flight(client, fq):
    fq.rerender = {"status": "queued", "error": None, "output_key": None}
    r = client.get("/reels/7")
    assert 'hx-get="/reels/7/rerender/status"' in r.text
    assert "Video out of date - re-render" not in r.text      # banner replaces the button


# --- re-render status polling -------------------------------------------------------

def test_rerender_status_running_returns_polling_banner(client, fq):
    fq.rerender = {"status": "running", "error": None, "output_key": None}
    r = client.get("/reels/5/rerender/status")
    assert r.status_code == 200
    assert 'hx-get="/reels/5/rerender/status"' in r.text      # keeps polling
    assert 'hx-trigger="every 4s"' in r.text
    assert "dot-pulse" in r.text


def test_rerender_status_done_offers_load_button(client, fq, monkeypatch):
    monkeypatch.setattr("reforge.app.queries._media_base_url", lambda: "")  # local-mount form
    fq.rerender = {"status": "done", "error": None, "output_key": "output/reel5_v2.mp4"}
    r = client.get("/reels/5/rerender/status")
    assert r.status_code == 200
    assert "New cut ready" in r.text
    assert 'data-new-src="/media/output/reel5_v2.mp4"' in r.text
    assert 'hx-trigger="every 4s"' not in r.text              # polling stops


def test_rerender_status_error_shows_tail_and_retry(client, fq):
    fq.rerender = {"status": "error", "error": "ffmpeg exploded at frame 9",
                   "output_key": None}
    r = client.get("/reels/5/rerender/status")
    assert r.status_code == 200
    assert "ffmpeg exploded at frame 9" in r.text
    assert 'hx-post="/reels/5/rerender"' in r.text            # retry re-queues


def test_rerender_status_404_when_never_rerendered(client, fq):
    assert client.get("/reels/5/rerender/status").status_code == 404


# --- live progress header (OOB on every verdict) ------------------------------------

def test_reel_page_header_has_progress_ids(client):
    r = client.get("/reels/5")
    assert 'id="decided-count"' in r.text
    assert 'id="progress-fill"' in r.text


def test_every_verdict_response_updates_the_header_oob(client, fq):
    """Live bug: 6 beats decided, header still said 1/8 until a full reload."""
    for post in (lambda: client.post("/slots/71/keep"),
                 lambda: client.post("/slots/71/swap", data={"moment_id": "901"}),
                 lambda: client.post("/slots/71/flag", data={"reason": "too dark"}),
                 lambda: client.post("/slots/71/undo")):
        r = post()
        assert r.status_code == 200
        count = r.text.split('id="decided-count"')[1]
        assert 'hx-swap-oob="true"' in count[:60]
        assert "3/8 decided" in count[:200]                  # counts from the verdict row
        fill = r.text.split('id="progress-fill"')[1]
        assert 'hx-swap-oob="true"' in fill[:60]
        assert "width: 37.5%" in fill[:200]


# --- "other" custom flag --------------------------------------------------------

def test_flag_rail_has_caption_placement_and_other_chips(client):
    r = client.get("/reels/5")
    assert ">caption placement</button>" in r.text
    other = r.text.split("data-requires-note")[0].rsplit("<button", 1)[1]
    assert 'hx-vals=\'{"reason": "other"}\'' in other         # other chip posts reason=other


def test_flag_rail_groups_chips_by_blame_target(client):
    r = client.get("/reels/5")
    assert "Bad clip" in r.text and "Good clip, wrong here" in r.text
    assert "Good clip, render issue" in r.text
    assert 'data-target="render"' in r.text                   # caption placement carries its target


def test_flag_other_with_note_persists_the_full_text(client, fq):
    r = client.post("/slots/71/flag", data={"reason": "other",
                                            "note": "caption covers my face"})
    assert r.status_code == 200
    assert ("flag_slot", 71, "other: caption covers my face") in fq.calls
    (entry,) = fq.taste_events
    assert entry["rule"] == "other"
    assert entry["note"] == "caption covers my face"          # minable full text


def test_flag_other_without_note_is_rejected(client, fq):
    r = client.post("/slots/71/flag", data={"reason": "other", "note": "   "})
    assert r.status_code == 400
    assert not any(c[0] == "flag_slot" for c in fq.calls)
    assert fq.taste_events == []


# --- re-pick from a flag -----------------------------------------------------------

def test_flagged_card_offers_repick(client):
    r = client.post("/slots/71/flag", data={"reason": "too dark"})
    assert 'hx-post="/slots/71/repick"' in r.text
    assert "Re-pick this beat" in r.text


def test_kept_card_does_not_offer_repick(client):
    r = client.post("/slots/71/keep")
    assert 'hx-post="/slots/71/repick"' not in r.text


def test_post_repick_queues_and_returns_polling_card(client, fq):
    r = client.post("/slots/71/repick")
    assert r.status_code == 200
    assert ("create_run_request", "repick:71") in fq.calls
    assert 'id="slot-71"' in r.text                           # still the card anchor
    assert 'hx-get="/slots/71/repick/status"' in r.text
    assert 'hx-trigger="every 4s"' in r.text
    assert "Re-pick queued" in r.text
    (entry,) = fq.taste_events
    assert entry["event"] == "repick_requested"
    assert entry["run"] == 5 and entry["beat"] == 3
    assert entry["critique"] == "too dark: face covered"


def test_post_repick_400_unless_flagged(client, fq):
    fq.slot_verdict = "keep"
    assert client.post("/slots/71/repick").status_code == 400
    assert not any(c[0] == "create_run_request" for c in fq.calls)


def test_post_repick_404_for_missing_slot(client, fq):
    assert client.post("/slots/999/repick").status_code == 404


def test_repick_status_running_keeps_polling(client, fq):
    fq.repick = {"status": "running", "error": None}
    r = client.get("/slots/71/repick/status")
    assert r.status_code == 200
    assert "Re-picking this beat" in r.text
    assert 'hx-trigger="every 4s"' in r.text


def test_repick_status_done_returns_fresh_open_card_with_oob_refreshes(client, fq):
    fq.repick = {"status": "done", "error": None}
    fq.slot_verdict = None                                    # worker cleared the verdict
    fq.dirty = True                                           # repick dirtied the cut
    r = client.get("/slots/71/repick/status")
    assert r.status_code == 200
    assert 'hx-post="/slots/71/keep"' in r.text               # card is open to judge again
    assert 'hx-trigger="every 4s"' not in r.text              # polling stops
    assert "3/8 decided" in r.text                            # OOB header refresh
    assert "Video out of date - re-render" in r.text          # OOB rerender area (dirty)


def test_repick_status_error_keeps_flag_and_shows_quiet_error(client, fq):
    fq.repick = {"status": "error", "error": "no candidates retrieved for shot 3"}
    r = client.get("/slots/71/repick/status")
    assert r.status_code == 200
    assert "Re-pick failed" in r.text
    assert "no candidates retrieved for shot 3" in r.text
    assert "Flagged" in r.text                                # flag intact
    assert 'hx-trigger="every 4s"' not in r.text


def test_repick_status_404_when_never_repicked(client, fq):
    assert client.get("/slots/71/repick/status").status_code == 404


# --- healthz + invite gate ------------------------------------------------------

def test_healthz_ok(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"ok": True, "db": True}


def test_invite_gate_blocks_without_code(fq, monkeypatch, taste_dir):
    monkeypatch.setenv("INVITE_CODE", "secret")
    c = TestClient(create_app(queries=fq))
    assert c.get("/").status_code == 403
    assert c.get("/healthz").status_code == 200              # keep-alive stays open
    r = c.get("/", params={"code": "secret"})
    assert r.status_code == 200                              # code accepted...
    assert c.get("/").status_code == 200                     # ...cookie now set


def test_create_run_rejects_unknown_reference(client, fq):
    resp = client.post("/runs", data={"reference_key": "../../etc/passwd"})
    assert resp.status_code == 400
    assert getattr(fq, "run_requests", []) == []
