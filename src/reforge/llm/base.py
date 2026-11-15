"""Model-agnostic vision/LLM interface (swap Gemini/Claude behind this)."""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class ReelBrief:
    """The reel-level creative read (Phase 3.5 taste layer).

    What a human holds in their head while picking clips, made explicit: match the reel's
    emotional register, not its pixels. A calmer clip can be right for a moody/sad reel -
    this is fit-to-vibe, NOT maximize-energy.
    """
    vibe: str
    emotional_register: str   # e.g. gritty / moody / triumphant / sad
    energy_profile: str       # how energy moves across the reel
    tone: str
    theme: str
    trend_context: str = ""   # web-search digest of what this trend is (acquire/trends.py)

@dataclass
class ShotSpec:
    """One shot of the reference reel."""
    idx: int
    start_s: float
    end_s: float
    description: str          # what the shot shows
    on_screen_text: str       # OCR'd overlay text (verbatim)
    energy: str               # e.g. "explosive", "slow", "ambient"
    query: str                # retrieval query for matching our footage
    # Phase 3.5 taste context (what a human knew about this beat's job in the reel)
    role: str = ""            # what the shot does in the arc (opener / build / payoff / breather...)
    intended_feeling: str = ""
    intended_energy: str = ""

@dataclass
class ClipMoment:
    """A retrieved candidate moment inside one of Logan's clips (Phase 3)."""
    clip_key: str
    start_s: float
    end_s: float
    description: str
    is_logan: bool
    framing_notes: str
    energy: str
    # retrieval enrichment (set by match.retrieve from the knn row)
    location: str | None = None
    moment_id: int | None = None
    clip_id: int | None = None
    thumbnail_key: str | None = None
    distance: float | None = None
    quality_score: float | None = None   # Phase 3.7 per-moment quality (None = pre-3.7)

@dataclass
class SlotPlan:
    """The rerank's decision for one reference shot: which moment + how to frame it.

    Framing stays on the LOCKED recipe - zoom is fixed at 1.0 (never punch in). Only
    xfrac/yfrac (off-center subject) and brightness (dark night) may deviate.
    """
    moment: ClipMoment
    in_point: float            # seconds into the source clip (peak action when there is one)
    zoom: float = 1.0
    xfrac: float = 0.5
    yfrac: float = 0.5
    brightness: str = "none"
    score: float = 0.0         # rerank score, higher = better
    second_person: bool = False  # use-time backstop: a 2nd person in the chosen window
    reason: str = ""           # one-line rerank rationale (to QA the taste)

@dataclass
class GateVerdict:
    """The holistic gate's call on one assembled slot (taste-architecture step 6).

    ``hold=True`` keeps the pick; ``hold=False`` sends the slot back for a re-pick with
    ``critique`` (one line: why it is weak + what to look for instead).
    """
    slot: int
    hold: bool
    critique: str = ""

@dataclass
class ClipAnnotation:
    """One clip-level annotation from the vision model (Phase 2 ingest).

    Locked design: Gemini is called once per CLIP (its timestamps are imprecise, so
    per-window semantics are deferred). Every mechanical window in the clip inherits
    this annotation; the per-window CLIP frame embedding is what the retriever ranks.
    """
    is_logan: bool
    location: str
    energy: str
    description: str
    framing_notes: str

class VisionModel(ABC):
    @abstractmethod
    def annotate_clip(self, local_path: str) -> ClipAnnotation:
        """Describe a whole library clip once (used by ingest; windows inherit it)."""

    @abstractmethod
    def assess_moment(self, frames: list[str]) -> tuple[float, list[str]]:
        """Classify a candidate moment: clean usable shot (1.0) vs transitional/awkward (0.0).

        Returns ``(vlm_quality, flags)``. One-time at ingest (Phase 3.7). This is CLASSIFICATION
        of a given window (not timestamp-finding), so VLM-timestamp imprecision doesn't apply.
        """

    @abstractmethod
    def read_reference(self, frames_per_shot: list[list[str]],
                       cut_grid: list[tuple[float, float]]) -> tuple[ReelBrief, list[ShotSpec]]:
        """Per-shot frames + cut grid -> (ReelBrief, ordered ShotSpecs) (used by deconstruct).

        The reel-level brief comes from frames spanning the WHOLE reel; ``frames_per_shot[i]``
        are the frames for shot ``i``; timing comes from the grid.
        """

    @abstractmethod
    def shortlist_score(self, brief_text: str, shot_ask: str, candidate_lines: list[str]) -> list[float]:
        """Cheap text-only fit scores (0-1) for each candidate's metadata vs brief + shot.

        No images - narrows a wide pool to finalists before the expensive vision rerank.
        Returns a score per candidate, parallel to ``candidate_lines``.
        """

    @abstractmethod
    def rerank(self, shot: ShotSpec, shot_frames: list[str],
               candidate_frames: list[tuple[ClipMoment, list[str]]],
               brief: ReelBrief, avoid_looks: list[str] | None = None,
               critique: str = "") -> list[SlotPlan]:
        """Rank finalists for a shot and choose framing + peak in_point. Quality lives here.

        Picks the clip that best SERVES this shot's role within the reel's vibe (fit-to-vibe
        + energy outweigh visual similarity); returns SlotPlans best-first with a one-line
        ``reason``. ``avoid_looks`` (3.6) softly nudges away from already-used looks this reel.
        ``critique`` is the holistic gate's note on why the PREVIOUS pick for this shot was
        rejected - the rerank must avoid that failure mode. Framing stays on the locked recipe.
        """

    def gate_review(self, brief: ReelBrief, slot_lines: list[str],
                    slot_frames: list[list[str]], gold_frames: list[str]) -> list[GateVerdict]:
        """Holistic final gate: view the WHOLE assembled pick sequence next to gold reels.

        ``slot_lines[i]`` is one text line per slot (idx, chosen clip, the rerank's reason);
        ``slot_frames[i]`` its frame paths (slot order); ``gold_frames`` are frames sampled
        from the gold-reference reels (the quality bar). Returns one ``GateVerdict`` per slot:
        hold=True keeps the pick, hold=False sends it back with a one-line critique.
        Non-abstract so eval-only backends (Gemini) keep working without it.
        """
        raise NotImplementedError("gate_review is not implemented for this VisionModel")
