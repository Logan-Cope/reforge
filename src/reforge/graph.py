"""LangGraph orchestration wrapper around the recreate pipeline (thin, no rewrite).

The working pipeline stays exactly as `scripts/recreate.py` runs it - this module just
wires the same functions into a checkpointed `StateGraph` with a **human-review
interrupt** before render:

    deconstruct -> match -> gate -> review (interrupt) -> render -> END

`review_node` calls `langgraph.types.interrupt()` with a compact per-slot summary
(`[{slot, clip, reason}, ...]`); the caller resumes with
``Command(resume={"approved": bool})``. Rejection records the decision and ends the run
without rendering (nothing is deleted; re-run with a new thread_id to try again).

Vision client: nodes share ONE `ClaudeVision` per process via the `get_vision()`
module-level cache - the client itself never enters graph state, so the checkpointer
only ever serializes plain dataclasses (Deconstruction / MatchedSlot), and the cost
ledger accumulates across nodes for an end-of-run summary.

`scripts/recreate.py` remains the framework-free path; `scripts/run_graph.py` drives
this graph.
"""
from __future__ import annotations

import dataclasses
import os
from typing import Any, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from reforge.deconstruct.reference import deconstruct
from reforge.llm import get_captioner
from reforge.match.gate import run_gate
from reforge.match.matcher import build_edl
from reforge.render.engine import render_reel
from reforge.render.overlay import burn_text
from reforge.taste import append_taste_log

FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"

_VISION = None


def get_vision():
    """One vision client per process (lazy) - shared cost ledger, never stored in state."""
    global _VISION
    if _VISION is None:
        _VISION = get_captioner()
    return _VISION


class ReelState(TypedDict, total=False):
    """Graph state: plain inputs + dataclass artifacts (checkpointer-serializable)."""
    reference: str
    out_path: str
    work_dir: str
    max_shots: int | None
    beat_s: float | None
    persist: bool               # default True; run_graph sets False for --no-persist
    decon: Any | None           # Deconstruction
    matched: list | None        # list[MatchedSlot]
    gate_events: list | None
    approved: bool
    output: str | None
    reel_id: int | None


# --- nodes (each a thin call into existing code) ------------------------------

def deconstruct_node(state: ReelState) -> dict:
    """Reference -> Deconstruction (+ optional first-N-shots truncation)."""
    decon = deconstruct(state["reference"], get_vision(), state["work_dir"],
                        beat_s=state.get("beat_s"))
    max_shots = state.get("max_shots")
    if max_shots is not None:
        decon = dataclasses.replace(
            decon,
            shots=decon.shots[:max_shots],
            frames_per_shot=decon.frames_per_shot[:max_shots],
        )
    return {"decon": decon}


def match_node(state: ReelState) -> dict:
    """Deconstruction -> matched slots (retrieve -> shortlist -> taste rerank)."""
    return {"matched": build_edl(state["decon"], get_vision())}


def gate_node(state: ReelState) -> dict:
    """Holistic gate vs gold references; taste-log every re-pick + a run summary."""
    run = os.path.splitext(os.path.basename(state["out_path"]))[0]
    matched, events = run_gate(state["decon"], state["matched"], get_vision())
    for e in events:
        append_taste_log({**e, "run": run})
    iters = max((e["iter"] for e in events), default=1)
    append_taste_log({"event": "gate_summary", "run": run, "slots": len(matched),
                      "repicks": len(events), "iters": iters})
    return {"matched": matched, "gate_events": events}


def review_node(state: ReelState) -> dict:
    """Human-review interrupt: pause with a per-slot summary, resume with a verdict."""
    summary = [{"slot": m.shot.idx, "clip": os.path.basename(m.render.src),
                "reason": m.plan.reason} for m in state["matched"]]
    decision = interrupt(summary)  # resumes with {"approved": bool}
    return {"approved": bool(decision.get("approved", False))}


def render_node(state: ReelState) -> dict:
    """Render + burn overlay (same block as recreate.py), then optionally persist."""
    decon, matched, out_path = state["decon"], state["matched"], state["out_path"]
    # render_reel's concat list needs an ABSOLUTE tmp_dir; burn the overlay separately
    # (this Homebrew ffmpeg lacks the `drawtext` filter - no libfreetype).
    silent = os.path.splitext(out_path)[0] + "_notext.mp4"
    render_reel([m.render for m in matched], decon.audio_path, silent, text="",
                tmp_dir=os.path.abspath(os.path.join(state["work_dir"], "segs")), font=FONT)
    out = burn_text(silent, decon.overlay_text, out_path, font=FONT)
    os.remove(silent)
    update: dict = {"output": out}
    if state.get("persist", True):
        update["reel_id"] = _persist_run(state["reference"], decon, matched, out)
    return update


def _persist_run(reference: str, decon, matched, out_path: str) -> int:
    """Write the reels/slots run record (same logic as recreate.py's `_persist`)."""
    from reforge.app.queries import alternates_json, update_slot_candidates
    from reforge.db import client as db

    total = sum(m.render.duration for m in matched)
    reel_id = db.insert_reel(
        reference_key=reference,
        total_duration=total,
        cut_grid=[[s.start_s, s.end_s] for s in decon.shots],
        brief=dataclasses.asdict(decon.brief),
        output_key=out_path,
    )
    for m in matched:
        slot_id = db.insert_slot(
            reel_id, m.shot.idx,
            start_s=m.shot.start_s, end_s=m.shot.end_s,
            ref_description=m.shot.description, ref_text_overlay=m.shot.on_screen_text,
            ref_energy=m.shot.energy, chosen_moment_id=m.plan.moment.moment_id,
            params={"in_point": m.render.in_point, "zoom": m.render.zoom,
                    "xfrac": m.render.xfrac, "yfrac": m.render.yfrac,
                    "brightness": m.render.brightness},
            reason=m.plan.reason,
        )
        alts = alternates_json(getattr(m, "alternates", []) or [])
        if alts:
            update_slot_candidates(slot_id, alts)
    return reel_id


# --- graph ---------------------------------------------------------------------

def _after_review(state: ReelState) -> str:
    return "render" if state.get("approved") else END


def build_graph(checkpointer=None):
    """Compile the checkpointed pipeline graph (default: in-memory checkpointer)."""
    g = StateGraph(ReelState)
    g.add_node("deconstruct", deconstruct_node)
    g.add_node("match", match_node)
    g.add_node("gate", gate_node)
    g.add_node("review", review_node)
    g.add_node("render", render_node)
    g.add_edge(START, "deconstruct")
    g.add_edge("deconstruct", "match")
    g.add_edge("match", "gate")
    g.add_edge("gate", "review")
    g.add_conditional_edges("review", _after_review, {"render": "render", END: END})
    g.add_edge("render", END)
    return g.compile(checkpointer=checkpointer or InMemorySaver())
