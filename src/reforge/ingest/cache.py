"""Content-addressed cache plumbing for the Phase 9A tiered ingest.

TIER-0 (immutable, expensive) is keyed by ``(content_hash, <extractor>_version, window_idx)``
and computed once - a cache hit SKIPS the decode / pose / embed / VLM work. TIER-1 (mutable,
cheap) is keyed by ``(content_hash, scorer_version, window_idx)`` and recomputed from Tier-0
raw signals in minutes. Versions make every artifact append-only: a new extractor / model /
scorer writes rows under a NEW version, so old + new coexist for A/B + rollback.

This module holds the versions + pure helpers. The DB accessors live in ``db.client`` (the
single DB layer); the pure logic here imports nothing heavy so its tests stay import-light.
"""
from __future__ import annotations

import os
import re

from ..config import settings

# --- versions (bump to recompute that tier; old rows are kept for A/B + rollback) -----------
# Deterministic geometry: segmentation + frame extraction + raw CV signals (one motion pass).
EXTRACTOR_VERSION = "geom-1"
# The Tier-1 score formula (`quality.score_window`). Bump + re-score = minutes, no decode/embed.
SCORER_VERSION = "score-1"
# Audio features extractor (structure only for now; see 9B).
AUDIO_VERSION = "audio-1"


def vlm_quality_version() -> str:
    """Version of the per-window VLM quality pass (model + prompt revision)."""
    return f"{settings.claude_model}/q1"


def annotator_version() -> str:
    """Version of the per-clip VLM caption/annotation (model + prompt revision)."""
    return f"{settings.claude_model}/a1"


def embed_version() -> str:
    """Version of the base visual embedding = the CLIP model name (swap = new version)."""
    from .embed import clip_model_name

    return clip_model_name()


# --- pure helpers (unit-tested, no DB / no media) ---------------------------

def window_idx_from_thumbnail_key(key: str | None) -> int | None:
    """Map a moment's saved frame back to its window index, e.g. ``.../004.jpg`` -> ``4``.

    The pipeline writes the f0 thumbnail as ``{idx:03d}.jpg`` (and f1 as ``{idx:03d}b.jpg``),
    content-addressed under ``thumbs/<hash12>/``. Lets the backfill populate the cache for
    existing moments from disk, without touching video. Returns None if the name isn't a window.
    """
    if not key:
        return None
    stem = os.path.splitext(os.path.basename(key))[0]
    m = re.match(r"^(\d+)", stem)
    return int(m.group(1)) if m else None
