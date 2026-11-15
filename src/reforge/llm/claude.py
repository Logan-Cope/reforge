"""Claude implementation of the VisionModel interface (Phase 2: PRIMARY captioner).

One Claude call per CLIP returns a structured clip-level annotation that every
mechanical window inherits (the locked design). Claude reads Logan's **identity
reference photos** plus a handful of frames sampled across the clip, and decides
`is_logan` by appearance match - biased to INCLUDE (see the decision rule in the
prompt). This fixes the earlier "single solo man" limitation, which both wrongly
included other solo men and wrongly excluded Logan when his face wasn't clear
(e.g. shadowboxing from behind read as "a woman").

Gemini + Twelve Labs remain video-native EVAL benchmarks behind this same interface.

Uses the official `anthropic` SDK with a forced, strict tool call for reliable
structured output. Model is `settings.claude_model` (CLAUDE_MODEL), kept swappable.

`is_image_file` / `image_files` are pure (unit-tested); the anthropic SDK and PIL are
imported lazily so this module imports light.
"""
from __future__ import annotations

import base64
import io
import os
import shutil
import tempfile

from ..config import settings
from ..taste import load_rubric
from .base import (
    ClipAnnotation,
    ClipMoment,
    GateVerdict,
    ReelBrief,
    ShotSpec,
    SlotPlan,
    VisionModel,
)

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")


def is_image_file(name: str) -> bool:
    """True if ``name`` is a usable reference image (by extension, case-insensitive)."""
    return name.lower().endswith(IMAGE_EXTS)


def image_files(listing: list[str]) -> list[str]:
    """Filter a directory listing to image files, sorted for a deterministic order."""
    return sorted(n for n in listing if is_image_file(n))


def _annotation_prompt(n_refs: int, n_frames: int) -> str:
    return (
        f"The FIRST {n_refs} images are reference photos of ONE specific man - call him "
        f'"Logan" (front, profile, and gym selfies; the SAME man in all of them). The '
        f"REMAINING {n_frames} images are frames sampled in order from a SINGLE short "
        f"B-roll clip. Record ONE annotation for the clip via the record_annotation tool.\n\n"
        "Decide is_logan IN THIS ORDER:\n"
        "1. COUNT the prominent people in the clip. If MORE THAN ONE person is prominently "
        "present, set is_logan=FALSE - ALWAYS, with NO exceptions, even if one of them looks "
        "like Logan. (Incidental background passersby don't count; judge who the clip is "
        "actually about.)\n"
        "2. Otherwise there is a single subject - compare that subject to the reference "
        "photos. Set is_logan=FALSE only if the subject is CLEARLY a DIFFERENT person than "
        "the reference (a face/build you are confident does not match). In EVERY other "
        "single-subject case - face not clearly visible, shot from behind or at a distance, "
        "poor lighting, or simply unsure - DEFAULT to is_logan=TRUE. Bias toward TRUE for a "
        "lone subject.\n\n"
        "- location: one short lowercase label for the setting (e.g. gym, dock, street, "
        "beach, home, outdoor).\n"
        "- energy: the clip's overall energy.\n"
        "- description: one concrete sentence describing the main action/subject (describe "
        "what is happening; do NOT assert the person's identity here).\n"
        "- framing_notes: a short note on how the subject is framed (e.g. 'full body, "
        "centered', 'mid-shot from the side', 'close on hands')."
    )


_ANNOTATION_TOOL = {
    "name": "record_annotation",
    "description": "Record the clip-level annotation for one B-roll clip.",
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "is_logan": {"type": "boolean"},
            "location": {"type": "string"},
            "energy": {"type": "string", "enum": ["explosive", "moderate", "slow", "ambient"]},
            "description": {"type": "string"},
            "framing_notes": {"type": "string"},
        },
        "required": ["is_logan", "location", "energy", "description", "framing_notes"],
        "additionalProperties": False,
    },
}

_REFERENCE_TOOL = {
    "name": "record_reference",
    "strict": True,
    "description": "Record the reel's creative brief and one entry per shot, in order.",
    "input_schema": {
        "type": "object",
        "properties": {
            "brief": {
                "type": "object",
                "properties": {
                    "vibe": {"type": "string", "description": "what the reel feels like overall"},
                    "emotional_register": {"type": "string", "description": "e.g. gritty / moody / triumphant / sad"},
                    "energy_profile": {"type": "string", "description": "how energy moves across the reel"},
                    "tone": {"type": "string"},
                    "theme": {"type": "string"},
                },
                "required": ["vibe", "emotional_register", "energy_profile", "tone", "theme"],
                "additionalProperties": False,
            },
            "shots": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "description": {"type": "string", "description": "what the shot shows"},
                        "on_screen_text": {"type": "string", "description": "verbatim overlay text, or empty"},
                        "energy": {"type": "string", "enum": ["explosive", "moderate", "slow", "ambient"]},
                        "query": {"type": "string", "description": "short search phrase to find matching B-roll"},
                        "role": {"type": "string", "description": "what this shot does in the arc (opener/build/payoff/breather...)"},
                        "intended_feeling": {"type": "string", "description": "what this beat should make the viewer feel"},
                        "intended_energy": {"type": "string", "description": "the energy this beat needs"},
                    },
                    "required": ["description", "on_screen_text", "energy", "query",
                                 "role", "intended_feeling", "intended_energy"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["brief", "shots"],
        "additionalProperties": False,
    },
}

_SHORTLIST_TOOL = {
    "name": "record_shortlist",
    "strict": True,
    "description": "Score each candidate's fit to the reel brief + this shot's role (text only).",
    "input_schema": {
        "type": "object",
        "properties": {
            "scores": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "candidate": {"type": "integer", "description": "candidate index"},
                        "fit": {"type": "number", "description": "0-1 fit to the reel's vibe + this shot's role/feeling/energy"},
                    },
                    "required": ["candidate", "fit"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["scores"],
        "additionalProperties": False,
    },
}

_RERANK_TOOL = {
    "name": "record_rerank",
    "strict": True,
    "description": "Rank the candidate clips for how well each recreates the reference shot.",
    "input_schema": {
        "type": "object",
        "properties": {
            "rankings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "candidate": {"type": "integer", "description": "candidate index"},
                        "score": {"type": "number", "description": "0-1, higher = better match"},
                        "peak_frac": {"type": "number", "description": "0-1, where the peak action is within this candidate's frames"},
                        "second_person": {"type": "boolean", "description": "true if a 2nd person appears in this candidate"},
                        "off_center": {"type": "string", "enum": ["no", "left", "right", "up", "down"]},
                        "dark_night": {"type": "boolean", "description": "true if a genuinely dark night clip"},
                        "reason": {"type": "string", "description": "one line: why this clip serves the shot's role within the reel's vibe"},
                    },
                    "required": ["candidate", "score", "peak_frac", "second_person", "off_center", "dark_night", "reason"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["rankings"],
        "additionalProperties": False,
    },
}

# Off-center hint -> a conservative xfrac/yfrac nudge (the recipe allows small deviations only).
_OFF_CENTER = {"no": (0.5, 0.5), "left": (0.4, 0.5), "right": (0.6, 0.5),
               "up": (0.5, 0.4), "down": (0.5, 0.6)}

_GATE_TOOL = {
    "name": "record_gate",
    "strict": True,
    "description": "Record the holistic quality-gate verdict for every slot of the assembled reel.",
    "input_schema": {
        "type": "object",
        "properties": {
            "verdicts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "slot": {"type": "integer", "description": "slot index"},
                        "hold": {"type": "boolean", "description": "true if this slot serves the reel as-is"},
                        "critique": {"type": "string", "description": "when hold=false: one line - why the pick is weak + what to look for instead"},
                    },
                    "required": ["slot", "hold", "critique"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["verdicts"],
        "additionalProperties": False,
    },
}

_MOMENT_QUALITY_TOOL = {
    "name": "record_moment_quality",
    "description": "Classify one ~1.5s candidate B-roll moment as a clean usable shot vs transitional junk.",
    "input_schema": {
        "type": "object",
        "properties": {
            "usable": {"type": "number", "description": "0-1: clean held usable shot (1) vs transitional/awkward/junk (0)"},
            "flags": {"type": "array", "items": {"type": "string",
                      "enum": ["clean", "transitional", "camera_handling", "no_subject", "awkward_framing", "blurry"]}},
        },
        "required": ["usable", "flags"],
        "additionalProperties": False,
    },
}


def _encode_downscaled(path: str, max_edge: int = 768, quality: int = 85) -> str:
    """Load an image, downscale to ``max_edge`` on the long side, return JPEG base64.

    References are re-sent on every clip, so shrinking them once keeps token cost down
    while preserving enough detail for face/identity matching.
    """
    from PIL import Image

    image = Image.open(path).convert("RGB")
    image.thumbnail((max_edge, max_edge))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    return base64.standard_b64encode(buffer.getvalue()).decode()


def _frame_block(path: str) -> dict:
    """Image block for an extracted video frame, downscaled to the API's 1568px cap.

    The API resizes anything larger server-side, so full-res frames only bloat the
    upload (a 44-frame gate call was ~18MB and stalled on slow connections).
    """
    return _image_block(_encode_downscaled(path, max_edge=1568))


def _image_block(b64: str) -> dict:
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}}


def _client_config(anthropic_key: str, gateway_key: str, gateway_base_url: str) -> tuple[str, str | None]:
    """(api_key, base_url) for the Anthropic client: AI Gateway when configured, else direct."""
    if gateway_key and gateway_base_url:
        return gateway_key, gateway_base_url
    return anthropic_key, None


def _maybe_trace(client):
    """Wrap an anthropic client with LangSmith tracing when configured; else return it unchanged.

    Strictly no-op unless LANGSMITH_API_KEY is set in the environment AND the `langsmith`
    package imports cleanly. When tracing activates, defaults the project to "reforge" and
    enables tracing (both only via setdefault, so explicit env config wins).
    """
    if not os.environ.get("LANGSMITH_API_KEY"):
        return client
    try:
        from langsmith.wrappers import wrap_anthropic
    except ImportError:
        return client
    os.environ.setdefault("LANGSMITH_PROJECT", "reforge")
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    return wrap_anthropic(client)


def _rerank_instruction(rubric: str, critique: str = "") -> str:
    """Final rerank instruction text; prepends the taste rubric and appends any gate critique."""
    base = (
        "Rank EVERY candidate via record_rerank. Pick the clip that best SERVES this shot's "
        "role within THIS reel's vibe: FIT-TO-VIBE and energy-match OUTWEIGH visual similarity, "
        "and a calmer/lower-intensity clip can be the right pick for a moody or sad reel. "
        "Penalize too-dark / low-visibility clips where the subject is hard to see. For each: "
        "score (0-1), peak_frac (0-1, where the peak action falls in its frames; 0.5 if none), "
        "second_person (a 2nd person present - disqualifying), off_center (subject clearly "
        "off-center), dark_night (genuinely dark night), and reason (one line on why it serves "
        "this beat). Do NOT consider zoom - framing is fixed, whole body in frame."
    )
    if critique:
        base += (
            "\n\nHOLISTIC GATE FEEDBACK - the previous pick for this shot was REJECTED by a "
            "holistic review of the assembled reel with this critique:\n" + critique +
            "\nAvoid that failure mode when picking now."
        )
    if not rubric:
        return base
    return ("LOGAN'S TASTE RUBRIC - apply every test below when judging candidates:\n"
            + rubric + "\n\n" + base)


def _parse_gate_verdicts(result: dict, n_slots: int) -> list[GateVerdict]:
    """record_gate tool output -> one GateVerdict per slot (pure).

    Unknown / out-of-range slot rows are skipped; slots the model did not mention default
    to hold=True (be conservative - re-pick only what the gate explicitly flagged).
    """
    by_slot: dict[int, GateVerdict] = {}
    for row in result.get("verdicts") or []:
        if not isinstance(row, dict):  # non-strict models have emitted bare strings here
            continue
        slot = row.get("slot")
        if not isinstance(slot, int) or isinstance(slot, bool) or not (0 <= slot < n_slots):
            continue
        by_slot[slot] = GateVerdict(slot=slot, hold=bool(row.get("hold", True)),
                                    critique=(row.get("critique") or "").strip())
    return [by_slot.get(i, GateVerdict(slot=i, hold=True)) for i in range(n_slots)]


class ClaudeVision(VisionModel):
    """VisionModel backed by Claude vision over identity references + sampled frames."""

    def __init__(self, model: str | None = None, n_frames: int = 6, ref_dir: str | None = None):
        if not (settings.anthropic_api_key or settings.ai_gateway_api_key):
            raise RuntimeError("ANTHROPIC_API_KEY (or AI_GATEWAY_API_KEY) is not set; add it to .env")
        import anthropic

        api_key, base_url = _client_config(
            settings.anthropic_api_key, settings.ai_gateway_api_key, settings.ai_gateway_base_url)
        # explicit timeout: a hung connection must fail fast and retry, not stall a run
        # (a gateway run once sat blocked on a socket read for 10+ minutes with no timeout)
        self._client = _maybe_trace(anthropic.Anthropic(api_key=api_key, base_url=base_url,
                                                        timeout=180.0, max_retries=3))
        self._model = model or settings.claude_model
        self._rerank_model = settings.rerank_model  # taste step (Opus by default, Phase 3.5)
        self._n_frames = n_frames
        self._rubric = load_rubric()  # Logan's taste rubric, injected into every rerank
        self._usage: dict[str, dict] = {}  # per-model token accounting for the cost summary

        ref_dir = ref_dir or settings.identity_ref_dir
        listing = [os.path.join(ref_dir, f) for f in os.listdir(ref_dir)] if os.path.isdir(ref_dir) else []
        ref_paths = image_files(listing)
        if not ref_paths:
            raise RuntimeError(
                f"no identity reference images in {ref_dir!r}; set IDENTITY_REF_DIR "
                "(is_logan now appearance-matches against Logan's reference photos)"
            )
        self._ref_blocks = [_image_block(_encode_downscaled(p)) for p in ref_paths]

    # --- model call + cost accounting --------------------------------------

    def _create(self, model: str, **kwargs):
        """messages.create that records token usage by model (for the cost summary)."""
        response = self._client.messages.create(model=model, **kwargs)
        acc = self._usage.setdefault(model, {"input": 0, "output": 0, "calls": 0})
        usage = getattr(response, "usage", None)
        acc["input"] += getattr(usage, "input_tokens", 0) or 0
        acc["output"] += getattr(usage, "output_tokens", 0) or 0
        acc["calls"] += 1
        return response

    def cost_summary(self) -> dict:
        """Priced per-run usage (see llm.cost.estimate_cost)."""
        from .cost import estimate_cost

        return estimate_cost(self._usage)

    def annotate_clip(self, local_path: str) -> ClipAnnotation:
        """Read references + frames sampled across the clip; return one annotation."""
        from ..ingest.frames import extract_frame, sample_timestamps
        from ..ingest.probe import probe_media

        duration_s = probe_media(local_path)["duration_s"]
        tmp_dir = tempfile.mkdtemp(prefix="reforge_caption_")
        try:
            content: list[dict] = list(self._ref_blocks)
            timestamps = sample_timestamps(duration_s, n=self._n_frames)
            for i, t in enumerate(timestamps):
                frame = extract_frame(local_path, t, os.path.join(tmp_dir, f"{i}.jpg"))
                content.append(_frame_block(frame))
            content.append({"type": "text", "text": _annotation_prompt(len(self._ref_blocks), len(timestamps))})

            response = self._create(
                self._model,
                max_tokens=1024,
                tools=[_ANNOTATION_TOOL],
                tool_choice={"type": "tool", "name": "record_annotation"},
                messages=[{"role": "user", "content": content}],
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        result = next(b.input for b in response.content if b.type == "tool_use")
        return ClipAnnotation(
            is_logan=bool(result.get("is_logan", False)),
            location=(result.get("location") or "").strip().lower(),
            energy=(result.get("energy") or "").strip().lower(),
            description=(result.get("description") or "").strip(),
            framing_notes=(result.get("framing_notes") or "").strip(),
        )

    def assess_moment(self, frames: list[str]) -> tuple[float, list[str]]:
        """One-time VLM quality classification of a candidate moment (Haiku)."""
        content: list[dict] = []
        for path in frames:
            content.append(_frame_block(path))
        content.append({"type": "text", "text": (
            "These frames are ONE ~1.5s candidate B-roll moment (in time order). Classify it via "
            "record_moment_quality. usable=1.0 for a CLEAN, held shot - a clear held subject/"
            "action, OR a clean intentional detail / equipment / ambiance shot - with decent "
            "framing. usable→0.0 ONLY for camera-handling / transitional junk: the camera being "
            "moved / picked up / set down, a tilt to the ground or ceiling, the subject walking up "
            "to or leaving frame, or heavy motion blur. flags: camera_handling (camera physically "
            "moved / set down - the ONLY truly unusable case), transitional, awkward_framing, "
            "blurry, no_subject (no person in frame - note this ALONE does NOT make a clean "
            "equipment/ambiance shot unusable), or clean. This is a QUALITY check, NOT vibe - a "
            "dim, low-energy, or person-less but cleanly-held shot is still usable."
        )})
        response = self._create(
            self._model,
            max_tokens=512,
            tools=[_MOMENT_QUALITY_TOOL],
            tool_choice={"type": "tool", "name": "record_moment_quality"},
            messages=[{"role": "user", "content": content}],
        )
        result = next(b.input for b in response.content if b.type == "tool_use")
        usable = min(1.0, max(0.0, float(result.get("usable", 0.5))))
        flags = [str(f) for f in (result.get("flags") or []) if f and f != "clean"]
        return usable, flags

    # --- Phase 3: deconstruct + rerank -------------------------------------

    def read_reference(self, frames_per_shot: list[list[str]],
                       cut_grid: list[tuple[float, float]]) -> tuple[ReelBrief, list[ShotSpec]]:
        """Read the whole reel into a creative brief + per-shot ShotSpecs (timing from grid)."""
        n = len(cut_grid)
        content: list[dict] = [{
            "type": "text",
            "text": (f"This is a {n}-shot viral B-roll Instagram Reel. The images are grouped by "
                     "shot, in order, and together span the WHOLE reel. FIRST read the reel as a "
                     "whole and fill `brief` (its vibe, emotional_register, energy_profile, tone, "
                     f"theme). THEN record exactly {n} `shots`, in order: `on_screen_text` = "
                     "verbatim overlay text (or empty); `query` = a short phrase to find matching "
                     "B-roll; `role` = what this beat does in the arc; `intended_feeling` / "
                     "`intended_energy` = what this beat should make the viewer feel/its energy. "
                     "Match the reel's emotional register - a calm beat can be right for a moody reel."),
        }]
        for i, shot_frames in enumerate(frames_per_shot):
            content.append({"type": "text", "text": f"Shot {i}:"})
            for path in shot_frames:
                content.append(_frame_block(path))

        response = self._create(
            self._model,
            max_tokens=4096,
            tools=[_REFERENCE_TOOL],
            tool_choice={"type": "tool", "name": "record_reference"},
            messages=[{"role": "user", "content": content}],
        )
        result = next(b.input for b in response.content if b.type == "tool_use")
        b = result.get("brief", {})
        brief = ReelBrief(
            vibe=(b.get("vibe") or "").strip(),
            emotional_register=(b.get("emotional_register") or "").strip(),
            energy_profile=(b.get("energy_profile") or "").strip(),
            tone=(b.get("tone") or "").strip(),
            theme=(b.get("theme") or "").strip(),
        )
        rows = result.get("shots", [])
        shots: list[ShotSpec] = []
        for i, (start, end) in enumerate(cut_grid):
            row = rows[i] if i < len(rows) else {}
            shots.append(ShotSpec(
                idx=i, start_s=start, end_s=end,
                description=(row.get("description") or "").strip(),
                on_screen_text=(row.get("on_screen_text") or "").strip(),
                energy=(row.get("energy") or "moderate").strip().lower(),
                query=(row.get("query") or row.get("description") or "").strip(),
                role=(row.get("role") or "").strip(),
                intended_feeling=(row.get("intended_feeling") or "").strip(),
                intended_energy=(row.get("intended_energy") or "").strip(),
            ))
        return brief, shots

    def shortlist_score(self, brief_text: str, shot_ask: str, candidate_lines: list[str]) -> list[float]:
        """Cheap text-only (Haiku) fit score per candidate vs the brief + shot. No images."""
        listing = "\n".join(f"[{i}] {line}" for i, line in enumerate(candidate_lines))
        prompt = (
            "Reel creative brief:\n" + brief_text + "\n\nThis shot needs:\n" + shot_ask +
            "\n\nCandidate clips (metadata only):\n" + listing +
            "\n\nScore EACH candidate 0-1 via record_shortlist for how well it would SERVE this "
            "shot within the reel's vibe - fit of energy/feeling/setting to the brief, not just "
            "keyword overlap. A calmer clip can score high for a moody/sad reel. Score every index."
        )
        response = self._create(
            self._model,
            max_tokens=1024,
            tools=[_SHORTLIST_TOOL],
            tool_choice={"type": "tool", "name": "record_shortlist"},
            messages=[{"role": "user", "content": prompt}],
        )
        rows = next(b.input for b in response.content if b.type == "tool_use")["scores"]
        scores = [0.0] * len(candidate_lines)
        for r in rows:
            if not isinstance(r, dict):
                continue
            ci = r.get("candidate")
            if isinstance(ci, int) and 0 <= ci < len(scores):
                scores[ci] = min(1.0, max(0.0, float(r.get("fit", 0.0))))
        return scores

    def rerank(self, shot: ShotSpec, shot_frames: list[str],
               candidate_frames: list[tuple[ClipMoment, list[str]]],
               brief: ReelBrief, avoid_looks: list[str] | None = None,
               critique: str = "") -> list[SlotPlan]:
        """Pick the finalist that best SERVES this shot's role within the reel's vibe."""
        from ..match.retrieve import brief_summary, shot_ask

        avoid = ""
        if avoid_looks:
            avoid = ("\n\nAVOID REPEATING these looks already used earlier in THIS reel "
                     "(prefer variety when fit is comparable): " + "; ".join(avoid_looks) + ".")
        content: list[dict] = [{
            "type": "text",
            "text": ("REEL BRIEF: " + brief_summary(brief) + "\n\nTHIS SHOT: " + shot_ask(shot) +
                     avoid + "\nReference shot frames:"),
        }]
        for path in shot_frames:
            content.append(_frame_block(path))

        for ci, (_cand, frames) in enumerate(candidate_frames):
            content.append({"type": "text", "text": f"CANDIDATE {ci} (frames in time order):"})
            for path in frames:
                content.append(_frame_block(path))

        content.append({"type": "text", "text": _rerank_instruction(self._rubric, critique)})

        response = self._create(
            self._rerank_model,
            max_tokens=2048,
            tools=[_RERANK_TOOL],
            tool_choice={"type": "tool", "name": "record_rerank"},
            messages=[{"role": "user", "content": content}],
        )
        rankings = next(b.input for b in response.content if b.type == "tool_use")["rankings"]

        plans: list[SlotPlan] = []
        for r in rankings:
            if not isinstance(r, dict):  # non-strict models have emitted bare strings here
                continue
            ci = r.get("candidate")
            if not isinstance(ci, int) or not (0 <= ci < len(candidate_frames)):
                continue
            moment = candidate_frames[ci][0]
            peak_frac = min(1.0, max(0.0, float(r.get("peak_frac", 0.5))))
            xfrac, yfrac = _OFF_CENTER.get(r.get("off_center", "no"), (0.5, 0.5))
            plans.append(SlotPlan(
                moment=moment,
                in_point=moment.start_s + peak_frac * (moment.end_s - moment.start_s),
                zoom=1.0, xfrac=xfrac, yfrac=yfrac,
                brightness="strong" if r.get("dark_night") else "none",
                score=float(r.get("score", 0.0)),
                second_person=bool(r.get("second_person", False)),
                reason=(r.get("reason") or "").strip(),
            ))
        return plans

    # --- Phase 3.8: holistic gate (taste-architecture step 6) ---------------

    def gate_review(self, brief: ReelBrief, slot_lines: list[str],
                    slot_frames: list[list[str]], gold_frames: list[str]) -> list[GateVerdict]:
        """View the WHOLE assembled pick sequence next to the gold reels; flag weak slots."""
        from ..match.retrieve import brief_summary

        content: list[dict] = [{
            "type": "text",
            "text": ("You are the final quality gate for an assembled reel: the per-slot picks "
                     "below are about to be rendered and posted. Review the sequence AS A WHOLE "
                     "against the reel's brief and the gold-standard references.\n\n"
                     "REEL BRIEF: " + brief_summary(brief)),
        }]
        if self._rubric:
            content.append({"type": "text", "text": (
                "LOGAN'S TASTE RUBRIC - apply every test below when judging slots:\n" + self._rubric)})
        content.append({"type": "text", "text": (
            "GOLD REFERENCES - the quality bar this reel must hold up against:")})
        content.extend(_frame_block(p) for p in gold_frames)
        for i, line in enumerate(slot_lines):
            content.append({"type": "text", "text": f"SLOT {i}: {line}"})
            content.extend(_frame_block(p) for p in slot_frames[i])
        content.append({"type": "text", "text": (
            "Judge HOLISTICALLY (arc, variety, no near-duplicate looks, fit to the brief) AND "
            "per-slot against the rubric. Record a verdict for EVERY slot via record_gate: "
            "hold=true when the slot serves the reel; hold=false with a one-line critique (why "
            "it is weak + what to look for instead) ONLY when it is clearly weak - unreadable "
            "subject, drifting out of frame, setting-up-not-training, technically dirty, too "
            "dark to read, or a near-duplicate of another slot. Be conservative: re-pick only "
            "clear failures, not taste ties.")})

        response = self._create(
            self._rerank_model,
            max_tokens=2048,
            tools=[_GATE_TOOL],
            tool_choice={"type": "tool", "name": "record_gate"},
            messages=[{"role": "user", "content": content}],
        )
        result = next(b.input for b in response.content if b.type == "tool_use")
        return _parse_gate_verdicts(result, len(slot_lines))
