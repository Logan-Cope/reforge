"""Routes for trial-reel publish scheduling - additive, own router (main.py includes it).

The reel page lazy-loads GET /reels/{id}/schedule into #schedule-area, so these
routes own the whole schedule surface and the existing review routes stay untouched.
Scheduling only INSERTS a scheduled_posts row; the actual publish happens on the
local publish worker (scripts/publish_worker.py), dry-run by default.
"""
from __future__ import annotations

from fastapi import APIRouter, Form, HTTPException, Request

from reforge.publish.scheduler import ACTIVE_STATUSES, fmt_local, parse_schedule_at

router = APIRouter()


def _ctx(request: Request, reel_id: int, *, open_form: bool = False) -> dict:
    q = request.app.state.queries
    detail = q.reel_detail(reel_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="reel not found")
    sched = q.latest_scheduled_post(reel_id)
    if sched:
        sched = {**sched, "when_local": fmt_local(sched.get("scheduled_at")),
                 "published_local": fmt_local(sched.get("published_at"))}
    return {"reel_id": reel_id, "approved": bool(detail["reel"]["approved"]),
            "published": bool(detail["reel"].get("published")),
            "sched": sched, "open_form": open_form,
            # pre-fill: the generated caption (worker backfills reels.caption)
            "caption": detail["reel"].get("caption") or ""}


def _partial(request: Request, reel_id: int, *, open_form: bool = False):
    return request.app.state.templates.TemplateResponse(
        request, "partials/schedule.html",
        _ctx(request, reel_id, open_form=open_form))


@router.get("/posted")
def posted_page(request: Request):
    """The Posted & scheduled page: everything that has left (or is leaving) the
    review queue, split into upcoming and already-live groups."""
    reels = request.app.state.queries.posted_reels()
    upcoming = [{**r, "when_local": fmt_local(r.get("scheduled_at"))}
                for r in reels if not r["published"]]
    live = [r for r in reels if r["published"]]
    return request.app.state.templates.TemplateResponse(
        request, "posted.html", {"upcoming": upcoming, "live": live})


@router.get("/reels/{reel_id}/schedule")
def schedule_state(request: Request, reel_id: int, open: int = 0):
    """The schedule partial. Default = current state (a normal-size button when the
    approved reel has nothing pending); ``?open=1`` = the expanded schedule panel."""
    return _partial(request, reel_id, open_form=bool(open))


@router.post("/reels/{reel_id}/schedule")
def schedule_reel(request: Request, reel_id: int, date: str = Form(...),
                  time: str = Form(...), as_trial: str = Form(""),
                  caption: str = Form("")):
    """Schedule the approved reel: ET date+time pickers -> one UTC row; the publish
    worker executes. The typed caption wins over the reel's stored one."""
    ctx = _ctx(request, reel_id)
    if not ctx["approved"]:
        raise HTTPException(status_code=400, detail="grade the reel first")
    if ctx["published"]:
        raise HTTPException(status_code=400, detail="already posted to Instagram")
    if ctx["sched"] and ctx["sched"]["status"] in ACTIVE_STATUSES:
        raise HTTPException(status_code=400, detail="already scheduled")
    try:
        when = parse_schedule_at(f"{date.strip()}T{time.strip()}")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    request.app.state.queries.create_scheduled_post(
        reel_id, when, as_trial=(as_trial == "on"), caption=caption.strip())
    return _partial(request, reel_id)


@router.post("/reels/{reel_id}/schedule/cancel")
def cancel_schedule(request: Request, reel_id: int):
    """Cancel a still-pending schedule; publishing/terminal rows can't be canceled."""
    ctx = _ctx(request, reel_id)
    sched = ctx["sched"]
    if not sched or sched["status"] != "scheduled":
        raise HTTPException(status_code=400, detail="nothing to cancel")
    request.app.state.queries.cancel_scheduled_post(sched["id"])
    return _partial(request, reel_id)
