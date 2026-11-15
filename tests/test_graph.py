"""LangGraph orchestration wrapper: wiring + interrupt/resume, no API/DB/ffmpeg.

Stubs replace `reforge.graph`'s module-level seams (deconstruct/build_edl/run_gate/
render_reel/burn_text/append_taste_log/get_vision) so the graph runs end-to-end for
free, proving the human-review interrupt fires with the right payload and that resume
routes to render (approved) or straight to END (rejected).
"""
from dataclasses import dataclass, field

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

import reforge.graph as rg


# --- serializable fakes (checkpointer must be able to round-trip state) -----

@dataclass
class FakeShot:
    idx: int


@dataclass
class FakeRender:
    src: str


@dataclass
class FakePlan:
    reason: str


@dataclass
class FakeSlot:
    shot: FakeShot
    render: FakeRender
    plan: FakePlan


@dataclass
class FakeDecon:
    shots: list = field(default_factory=list)
    frames_per_shot: list = field(default_factory=list)
    audio_path: str = "audio.m4a"
    overlay_text: str = "DISCIPLINE"
    brief: str = "brief"


def _decon(n: int = 2) -> FakeDecon:
    return FakeDecon(shots=[FakeShot(i) for i in range(n)],
                     frames_per_shot=[[f"f{i}.jpg"] for i in range(n)])


def _matched(decon: FakeDecon) -> list[FakeSlot]:
    return [FakeSlot(s, FakeRender(f"/lib/clip{s.idx}.mp4"), FakePlan(f"reason {s.idx}"))
            for s in decon.shots]


@pytest.fixture
def stubbed(monkeypatch, tmp_path):
    """Wire all external seams to stubs; return a call recorder."""
    calls: dict = {"render": 0}
    monkeypatch.setattr(rg, "get_vision", lambda: None)
    monkeypatch.setattr(rg, "deconstruct",
                        lambda ref, vision, work, beat_s=None: _decon())
    monkeypatch.setattr(rg, "build_edl", lambda decon, vision: _matched(decon))
    monkeypatch.setattr(rg, "run_gate", lambda decon, matched, vision: (matched, []))
    monkeypatch.setattr(rg, "append_taste_log", lambda entry: None)

    def fake_render_reel(edl, audio, out, text="", tmp_dir="", font=""):
        calls["render"] += 1
        open(out, "wb").close()
        return out

    def fake_burn_text(in_path, text, out_path, *, font):
        open(out_path, "wb").close()
        return out_path

    monkeypatch.setattr(rg, "render_reel", fake_render_reel)
    monkeypatch.setattr(rg, "burn_text", fake_burn_text)
    return calls


def _state(tmp_path, **over):
    base = {"reference": "ref.mp4", "out_path": str(tmp_path / "out.mp4"),
            "work_dir": str(tmp_path / "work"), "max_shots": None, "beat_s": None,
            "persist": False}
    base.update(over)
    return base


def _cfg(tid: str):
    return {"configurable": {"thread_id": tid}}


# --- tests -------------------------------------------------------------------

def test_build_graph_compiles_with_expected_nodes():
    app = rg.build_graph(InMemorySaver())
    nodes = set(app.get_graph().nodes)
    assert {"deconstruct", "match", "gate", "review", "render"} <= nodes


def test_interrupts_at_review_with_slot_payload(stubbed, tmp_path):
    app = rg.build_graph(InMemorySaver())
    result = app.invoke(_state(tmp_path), _cfg("t1"))
    intr = result.get("__interrupt__")
    assert intr, "graph should pause at review"
    assert intr[0].value == [
        {"slot": 0, "clip": "clip0.mp4", "reason": "reason 0"},
        {"slot": 1, "clip": "clip1.mp4", "reason": "reason 1"},
    ]
    assert stubbed["render"] == 0  # nothing rendered before approval


def test_resume_approved_true_reaches_render(stubbed, tmp_path):
    app = rg.build_graph(InMemorySaver())
    cfg = _cfg("t2")
    app.invoke(_state(tmp_path), cfg)
    final = app.invoke(Command(resume={"approved": True}), cfg)
    assert final["approved"] is True
    assert stubbed["render"] == 1
    assert final["output"] == str(tmp_path / "out.mp4")
    assert final.get("reel_id") is None  # persist=False


def test_resume_approved_false_skips_render(stubbed, tmp_path):
    app = rg.build_graph(InMemorySaver())
    cfg = _cfg("t3")
    app.invoke(_state(tmp_path), cfg)
    final = app.invoke(Command(resume={"approved": False}), cfg)
    assert final["approved"] is False
    assert stubbed["render"] == 0
    assert final.get("output") is None


def test_max_shots_truncates_decon(stubbed, monkeypatch, tmp_path):
    monkeypatch.setattr(rg, "deconstruct",
                        lambda ref, vision, work, beat_s=None: _decon(4))
    app = rg.build_graph(InMemorySaver())
    result = app.invoke(_state(tmp_path, max_shots=2), _cfg("t4"))
    assert len(result["__interrupt__"][0].value) == 2
