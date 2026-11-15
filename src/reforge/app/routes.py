"""Routes for the review + curation surface. Every human action feeds the taste memory.

Handlers stay thin: read/write via ``request.app.state.queries`` (injectable), record a
taste event through that same seam (durable ``taste_events`` row + best-effort local
file - see ``taste.record_taste_event``) as a BackgroundTask AFTER the response is sent
(the tap latency budget is one DB round trip), render a template or HTMX partial. The "Run"
button only queues a ``run_requests`` row; the heavy pipeline runs on the local worker.
"""
from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Form, HTTPException, Request
from fastapi.responses import HTMLResponse

from reforge.app.queries import reel_video_url
from reforge.taste import record_taste_event

router = APIRouter()


def _tpl(request: Request, name: str, ctx: dict):
    return request.app.state.templates.TemplateResponse(request, name, ctx)


def _render(request: Request, name: str, ctx: dict) -> str:
    """Render a partial to a string (for multi-fragment HTMX responses with OOB swaps)."""
    return request.app.state.templates.env.get_template(name).render(ctx)


def _queries(request: Request):
    return request.app.state.queries


def _progress_oob(request: Request, decided: int, total: int) -> str:
    """The OOB header fragments keeping "N/M decided" + the bar live after a verdict."""
    return _render(request, "partials/progress.html",
                   {"decided": decided, "total": total, "oob": True})


def _verdict_response(request: Request, q, slot: dict, *,
                      rerender_area: bool = False) -> HTMLResponse:
    """The slot partial PLUS the OOB fragments a verdict must refresh live.

    Every verdict updates the header progress; swap/undo can also move the reel's
    cut-freshness (``cut_dirty`` rides on the verdict row - no extra round trip), so
    they refresh #rerender-area too.
    """
    html = _render(request, "partials/slot.html", {"slot": slot})
    html += _progress_oob(request, slot["decided"], slot["total"])
    # keep the beat-rail chip live: its thumbnail follows the swapped clip and its color
    # follows the verdict, both of which just changed.
    html += _render(request, "partials/beat_chip.html", {"slot": slot, "oob": True})
    if rerender_area:
        html += _render(request, "partials/rerender_area.html",
                        {"reel_id": slot["reel_id"],
                         "cut_dirty": slot["cut_dirty"], "oob": True})
    return HTMLResponse(html)


def flag_note(reason: str, note: str) -> str:
    """Compose the persisted human_note for a flag: the reason, plus any typed note."""
    return f"{reason}: {note}" if note else reason


# What a flag blames - so distillation knows what to penalize:
#   clip   -> the footage itself (too dark, grainy, not-me): down-weight this moment
#             everywhere.
#   fit    -> good footage, wrong here (off-vibe, repeat look): re-pick this beat only;
#             the clip keeps full standing elsewhere.
#   render -> good footage, the render failed it (caption over the face): a render-layer
#             lesson; NEVER down-weight the clip.
# A flag with no mapping (the typed "other") stays 'other' and is read from its note.
FLAG_TARGETS = {
    "not clearly me": "clip", "too dark": "clip", "out of frame": "clip",
    "grainy": "clip", "setting up": "clip",
    "off-vibe": "fit", "repeat look": "fit",
    "caption placement": "render",
}


def flag_target(reason: str) -> str:
    """Classify a flag reason by what it blames (clip | fit | render | other)."""
    return FLAG_TARGETS.get(reason, "other")


@router.get("/")
def dashboard(request: Request):
    q = _queries(request)
    return _tpl(request, "dashboard.html",
                {"reels": q.list_reels(), "counts": q.counts(),
                 "runs": q.list_run_requests(), "references": q.available_references()})


@router.get("/reels/{reel_id}")
def reel_review(request: Request, reel_id: int):
    q = _queries(request)
    detail = q.reel_detail(reel_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="reel not found")
    # a queued/running re-render shows its live status banner instead of the button
    return _tpl(request, "reel.html", {"reel": detail["reel"], "slots": detail["slots"],
                                       "rerender": q.latest_rerender(reel_id)})


@router.post("/slots/{slot_id}/keep")
def keep_slot(request: Request, background_tasks: BackgroundTasks, slot_id: int,
              note: str = Form("")):
    q = _queries(request)
    slot = q.keep_slot(slot_id, note)
    if slot is None:
        raise HTTPException(status_code=404, detail="slot not found")
    # taste events run AFTER the response - the tap never waits on a second DB write
    background_tasks.add_task(
        record_taste_event,
        {"event": "human_keep", "run": slot["reel_id"], "beat": slot["idx"], "reason": note},
        queries=q)
    return _verdict_response(request, q, slot)


@router.post("/slots/{slot_id}/weak")
def weak_keep_slot(request: Request, background_tasks: BackgroundTasks, slot_id: int,
                   note: str = Form("")):
    """Keep the beat but flag it a weak positive ("could be better") - a per-beat
    signal finer than the reel grade."""
    q = _queries(request)
    slot = q.weak_keep_slot(slot_id, note)
    if slot is None:
        raise HTTPException(status_code=404, detail="slot not found")
    background_tasks.add_task(
        record_taste_event,
        {"event": "human_keep", "strength": "weak", "run": slot["reel_id"],
         "beat": slot["idx"], "reason": note},
        queries=q)
    return _verdict_response(request, q, slot)


@router.post("/slots/{slot_id}/swap")
def swap_slot(request: Request, background_tasks: BackgroundTasks, slot_id: int,
              moment_id: int = Form(...), note: str = Form("")):
    q = _queries(request)
    slot = q.swap_slot(slot_id, moment_id, note)
    if slot is None:
        raise HTTPException(status_code=404, detail="slot not found")
    background_tasks.add_task(
        record_taste_event,
        {"event": "human_swap", "run": slot["reel_id"], "beat": slot["idx"],
         "reason": note, "old": slot["chosen_moment_id"], "new": moment_id},
        queries=q)
    return _verdict_response(request, q, slot, rerender_area=True)


@router.post("/slots/{slot_id}/undo")
def undo_slot(request: Request, background_tasks: BackgroundTasks, slot_id: int):
    """Clear an accidental verdict; the taste log records the retraction so downstream
    consumers can discount the earlier keep/swap/flag event."""
    q = _queries(request)
    slot = q.undo_slot(slot_id)
    if slot is None:
        raise HTTPException(status_code=404, detail="slot not found")
    background_tasks.add_task(
        record_taste_event,
        {"event": "human_undo", "run": slot["reel_id"], "beat": slot["idx"]}, queries=q)
    return _verdict_response(request, q, slot, rerender_area=True)


@router.post("/slots/{slot_id}/flag")
def flag_slot(request: Request, background_tasks: BackgroundTasks, slot_id: int,
              reason: str = Form(...), note: str = Form("")):
    q = _queries(request)
    # "other" is only meaningful as the typed note (judge.js blocks this client-side too)
    if reason == "other" and not note.strip():
        raise HTTPException(status_code=400, detail="the 'other' flag needs a note")
    slot = q.flag_slot(slot_id, flag_note(reason, note))
    if slot is None:
        raise HTTPException(status_code=404, detail="slot not found")
    entry = {"event": "human_flag", "run": slot["reel_id"], "beat": slot["idx"],
             "rule": reason, "target": flag_target(reason)}
    if note:
        entry["note"] = note
    background_tasks.add_task(record_taste_event, entry, queries=q)
    return _verdict_response(request, q, slot)


@router.post("/slots/{slot_id}/repick")
def repick_slot(request: Request, background_tasks: BackgroundTasks, slot_id: int):
    """Send a flagged beat back for a fresh retrieval, the flag as the critique.

    Only queues a ``repick:<slot_id>`` run request; the local worker runs the actual
    match (scripts/repick_slot.py). The card flips to a polling "re-pick queued" state.
    """
    q = _queries(request)
    slot = q.slot_detail(slot_id)
    if slot is None:
        raise HTTPException(status_code=404, detail="slot not found")
    if slot["human_verdict"] != "flagged":
        raise HTTPException(status_code=400, detail="only flagged beats can be re-picked")
    q.create_run_request(f"repick:{slot_id}")
    background_tasks.add_task(
        record_taste_event,
        {"event": "repick_requested", "run": slot["reel_id"], "beat": slot["idx"],
         "critique": slot["human_note"] or ""},
        queries=q)
    return _tpl(request, "partials/repick_queued.html", {"slot": slot, "status": "queued"})


@router.get("/slots/{slot_id}/repick/status")
def repick_status(request: Request, slot_id: int):
    """Poll target for the re-pick card: queued/running keep the banner; done returns
    the fresh open card (new pick, verdict cleared by the worker) plus the OOB header
    progress and #rerender-area (a repick always leaves the cut dirty); error returns
    the card with the flag intact and a quiet failure line."""
    q = _queries(request)
    rp = q.latest_repick(slot_id)
    if rp is None:
        raise HTTPException(status_code=404, detail="no re-pick for this slot")
    slot = q.slot_detail(slot_id)
    if slot is None:
        raise HTTPException(status_code=404, detail="slot not found")
    if rp["status"] in ("queued", "running"):
        return _tpl(request, "partials/repick_queued.html",
                    {"slot": slot, "status": rp["status"]})
    if rp["status"] == "error":
        return _tpl(request, "partials/slot.html",
                    {"slot": slot, "repick_error": rp.get("error") or "worker error"})
    counts = q.decided_counts(slot["reel_id"])
    html = _render(request, "partials/slot.html", {"slot": slot})
    html += _progress_oob(request, counts["decided"], counts["total"])
    html += _render(request, "partials/rerender_area.html",
                    {"reel_id": slot["reel_id"],
                     "cut_dirty": q.reel_cut_dirty(slot["reel_id"]), "oob": True})
    return HTMLResponse(html)


@router.post("/reels/{reel_id}/grade")
def grade_reel(request: Request, background_tasks: BackgroundTasks, reel_id: int,
               grade: str = Form(...)):
    """One reel-level judgment. A positive grade endorses every unflagged beat (the
    ``accepted`` list) so the taste log learns from all of them without a per-beat tap;
    ``not_it`` rejects the concept (signal for the discovery side)."""
    q = _queries(request)
    try:
        res = q.grade_reel(reel_id, grade)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    background_tasks.add_task(
        record_taste_event,
        {"event": "review_summary" if res["approved"] else "reel_reject",
         "run": reel_id, "grade": grade, "slots": res["slots"], "kept": res["kept"],
         "swapped": res["swapped"], "flagged": res["flagged"],
         "accepted": res["accepted"], "approved": res["approved"]},
        queries=q)
    return _tpl(request, "partials/approve.html",
                {"reel": {"id": reel_id, "grade": grade, "approved": res["approved"]},
                 "grade": grade, "accepted_count": len(res["accepted"]),
                 "flagged": res["flagged"]})


@router.post("/reels/{reel_id}/rerender")
def rerender_reel(request: Request, reel_id: int):
    """Queue a re-render of this reel with its swaps applied; the local worker executes."""
    q = _queries(request)
    if q.reel_detail(reel_id) is None:
        raise HTTPException(status_code=404, detail="reel not found")
    q.create_run_request(f"rerender:{reel_id}")
    record_taste_event({"event": "rerender_requested", "run": reel_id}, queries=q)
    return _tpl(request, "partials/rerender_queued.html",
                {"reel_id": reel_id, "status": "queued"})


@router.get("/reels/{reel_id}/rerender/status")
def rerender_status(request: Request, reel_id: int):
    """Live status for the polling banner: queued/running keep polling; done offers to
    load the fresh cut (reels.output_key was updated by the worker); error shows why."""
    q = _queries(request)
    rr = q.latest_rerender(reel_id)
    if rr is None:
        raise HTTPException(status_code=404, detail="no re-render for this reel")
    return _tpl(request, "partials/rerender_queued.html",
                {"reel_id": reel_id, "status": rr["status"], "error": rr.get("error"),
                 "video_url": reel_video_url(rr.get("output_key"))})


@router.post("/runs")
def create_run(request: Request, reference_key: str = Form(...)):
    """Queue a pipeline run; the local worker picks it up. Returns the new queue row."""
    q = _queries(request)
    # allowlist: reference_key reaches a worker subprocess, so only known references queue
    if reference_key not in set(q.available_references()):
        raise HTTPException(status_code=400, detail="unknown reference")
    run_id = q.create_run_request(reference_key)
    run = {"id": run_id, "reference_key": reference_key, "status": "queued",
           "created": "just now", "started": "", "finished": "", "reel_id": None, "error": None}
    return _tpl(request, "partials/run_request.html", {"run": run})


@router.get("/curate")
def curate(request: Request):
    q = _queries(request)
    batch = q.curation_batch(limit=1)
    return _tpl(request, "curate.html",
                {"moment": batch[0] if batch else None, "counts": q.counts()})


@router.post("/moments/{moment_id}/rate")
def rate_moment(request: Request, background_tasks: BackgroundTasks, moment_id: int,
                label: int = Form(...)):
    q = _queries(request)
    q.rate_moment(moment_id, label)
    background_tasks.add_task(
        record_taste_event,
        {"event": "clip_label", "moment_id": moment_id, "label": label}, queries=q)
    batch = q.curation_batch(limit=1)  # auto-load the next card
    return _tpl(request, "partials/moment_card.html", {"moment": batch[0] if batch else None})
