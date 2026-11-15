"""Model-agnostic vision/LLM interface + captioner selection.

`get_captioner` returns the configured Phase 2 captioner: Claude (primary, frame-based)
or Gemini (video-native benchmark). Heavy SDKs are imported lazily inside the factory.
"""
from __future__ import annotations

from .base import ClipAnnotation, ClipMoment, ShotSpec, VisionModel

__all__ = ["ClipAnnotation", "ClipMoment", "ShotSpec", "VisionModel", "get_captioner"]


def get_captioner(name: str | None = None) -> VisionModel:
    """Return the configured captioner VisionModel (``claude`` default, or ``gemini``)."""
    from ..config import settings

    choice = (name or settings.captioner).lower()
    if choice == "gemini":
        from .gemini import GeminiVision

        return GeminiVision()
    if choice == "claude":
        from .claude import ClaudeVision

        return ClaudeVision()
    raise ValueError(f"unknown captioner {choice!r}; expected 'claude' or 'gemini'")
