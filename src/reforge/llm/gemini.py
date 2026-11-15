"""Gemini implementation of the VisionModel interface (Phase 2: clip annotation).

One Gemini call per CLIP returns a structured clip-level annotation that every
mechanical window inherits (the locked design - Gemini's per-moment timestamps are
imprecise, so per-window semantics are deferred). Uses the pinned `settings.gemini_model`
behind the model-agnostic `llm.base.VisionModel` so the provider stays swappable.

Built on the current `google-genai` SDK (the older `google-generativeai` package is
deprecated and its Files API rejects the new key format).
"""
from __future__ import annotations

import json
import time

import typing_extensions as typing

from ..config import settings
from .base import ClipAnnotation, ClipMoment, ReelBrief, ShotSpec, SlotPlan, VisionModel

_ANNOTATION_PROMPT = (
    "You are cataloging a single short B-roll clip from a fitness creator's library.\n"
    "Watch the whole clip and return ONE annotation describing it as a whole.\n\n"
    "Fields:\n"
    "- is_logan: true if a SINGLE man is the clear subject throughout and no other "
    "prominent people appear; false if multiple people appear, the subject is a woman, "
    "or there is no clear single human subject.\n"
    "- location: one short lowercase label for the setting (e.g. gym, dock, street, "
    "beach, home, outdoor).\n"
    "- energy: exactly one of explosive, moderate, slow, ambient.\n"
    "- description: one concrete sentence describing the main action/subject.\n"
    "- framing_notes: a short note on how the subject is framed (e.g. 'full body, "
    "centered', 'mid-shot from the side', 'close on hands')."
)


class _AnnotationSchema(typing.TypedDict):
    is_logan: bool
    location: str
    energy: str
    description: str
    framing_notes: str


class GeminiVision(VisionModel):
    """VisionModel backed by Gemini video understanding."""

    def __init__(self, model: str | None = None):
        if not settings.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is not set; add it to .env")
        from google import genai

        self._client = genai.Client(api_key=settings.gemini_api_key)
        self._model_name = model or settings.gemini_model

    def annotate_clip(self, local_path: str) -> ClipAnnotation:
        """Upload a clip to Gemini and return one clip-level annotation."""
        from google.genai import types

        uploaded = self._upload_and_wait(local_path)
        try:
            response = self._client.models.generate_content(
                model=self._model_name,
                contents=[uploaded, _ANNOTATION_PROMPT],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=_AnnotationSchema,
                    temperature=0.2,
                ),
            )
            data = json.loads(response.text)
        finally:
            self._client.files.delete(name=uploaded.name)
        return ClipAnnotation(
            is_logan=bool(data.get("is_logan", False)),
            location=(data.get("location") or "").strip().lower(),
            energy=(data.get("energy") or "").strip().lower(),
            description=(data.get("description") or "").strip(),
            framing_notes=(data.get("framing_notes") or "").strip(),
        )

    def _upload_and_wait(self, local_path: str, timeout_s: float = 180.0, poll_s: float = 2.0):
        """Upload a file to the Gemini Files API and block until it is ACTIVE."""
        uploaded = self._client.files.upload(file=local_path)
        waited = 0.0
        while uploaded.state.name == "PROCESSING" and waited < timeout_s:
            time.sleep(poll_s)
            waited += poll_s
            uploaded = self._client.files.get(name=uploaded.name)
        if uploaded.state.name != "ACTIVE":
            self._client.files.delete(name=uploaded.name)
            raise RuntimeError(f"Gemini file not ACTIVE (state={uploaded.state.name}) for {local_path}")
        return uploaded

    def assess_moment(self, frames: list[str]) -> tuple[float, list[str]]:
        raise NotImplementedError("Gemini is an eval-phase benchmark; use ClaudeVision for Phase 3.7")

    # --- Phase 3 (not yet implemented) -------------------------------------

    def read_reference(self, frames_per_shot: list[list[str]],
                       cut_grid: list[tuple[float, float]]) -> tuple[ReelBrief, list[ShotSpec]]:
        raise NotImplementedError("Gemini is an eval-phase benchmark; use ClaudeVision for Phase 3")

    def shortlist_score(self, brief_text: str, shot_ask: str, candidate_lines: list[str]) -> list[float]:
        raise NotImplementedError("Gemini is an eval-phase benchmark; use ClaudeVision for Phase 3")

    def rerank(self, shot: ShotSpec, shot_frames: list[str],
               candidate_frames: list[tuple[ClipMoment, list[str]]],
               brief: ReelBrief, avoid_looks: list[str] | None = None,
               critique: str = "") -> list[SlotPlan]:
        raise NotImplementedError("Gemini is an eval-phase benchmark; use ClaudeVision for Phase 3")
