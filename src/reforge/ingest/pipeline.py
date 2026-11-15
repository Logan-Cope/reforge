"""Ingest pipeline: clip -> segment -> annotate -> embed -> Supabase.

`ingest_clip` catalogs one clip; `ingest_library` bulk-runs over staged footage.
Adding clips is a one-command step: drop them in storage, run, cataloged.

The pure catalog logic (`is_video_key`, `video_keys`, `build_moments`,
`ingest_action`, `sha256_file`) is factored out and unit-tested with no DB / media /
model. The orchestration lazily imports the heavy pieces (DB, ffmpeg, open_clip,
Gemini) so the pure helpers - and their tests - stay import-light.

Locked Phase 2 design (see PROGRESS.md): mechanical ~1.5-2s windows = moment rows;
ONE Gemini call per clip sets is_logan/location/energy/description/framing and every
window inherits it; the per-window CLIP frame embedding is the retriever; dedup by
content_hash (sha256 of bytes) + storage_key.
"""
from __future__ import annotations

import hashlib
import os
from typing import Any

from ..config import settings
from ..llm.base import ClipAnnotation
from . import cache

VIDEO_EXTS = (".mp4", ".mov", ".m4v")

# Clips force-set to is_logan=false by filename stem so they can never surface in a reel -
# cases the captioner can't reliably catch from sampled frames. Two reasons (both
# architect-confirmed against the actual footage):
#   - not Logan but read as a solo man: IMG_5544, IMG_5551
#   - genuinely 2-person, but the intermittent 2nd person was missed by the 6-frame sample
#     (so it read single-subject and bias-to-include defaulted it true): IMG_2589, IMG_2593
# Reliable filename-keyed backstop for the curated library; the general fix is the Phase 3
# use-time second-person check on the chosen window. See PROGRESS.md / BUILD_SPEC Phase 3.
IS_LOGAN_EXCLUDE = frozenset({"IMG_5544", "IMG_5551", "IMG_2589", "IMG_2593"})

# Audit note stamped on clips.notes when force-excluded.
EXCLUDE_NOTE = "is_logan forced false (manual exclusion)"


# --- pure catalog helpers (unit-tested) -------------------------------------

def is_video_key(key: str) -> bool:
    """True if ``key`` names a video file we ingest (by extension, case-insensitive)."""
    return key.lower().endswith(VIDEO_EXTS)


def video_keys(listing: list[str]) -> list[str]:
    """Filter a storage listing down to video files, preserving order."""
    return [k for k in listing if is_video_key(k)]


def build_moments(windows: list[tuple[float, float]], annotation: ClipAnnotation) -> list[dict[str, Any]]:
    """Turn mechanical windows + one clip annotation into moment rows.

    Every window inherits the clip-level annotation (the locked design): is_logan,
    location, energy, description, framing_notes. Returned dicts map 1:1 onto
    ``db.client.upsert_moment`` keyword arguments (minus thumbnail, set during I/O).
    """
    return [
        {
            "start_s": start,
            "end_s": end,
            "description": annotation.description,
            "energy": annotation.energy,
            "location": annotation.location,
            "is_logan": annotation.is_logan,
            "framing_notes": annotation.framing_notes,
        }
        for start, end in windows
    ]


def ingest_action(*, hash_exists: bool, key_exists: bool, force: bool = False) -> str:
    """Decide what to do with a clip: ``"fresh"``, ``"reingest"``, or ``"skip"``.

    - content already cataloged (same bytes, by hash) -> ``skip`` (idempotent: a second
      run over the same library adds zero rows; also catches re-downloaded duplicates).
    - storage_key present but bytes differ (changed clip) -> ``reingest`` (drop old
      moments, refresh the clip + moments).
    - otherwise -> ``fresh``. ``force`` bypasses the skip and re-ingests/refreshes.
    """
    if force:
        return "reingest" if key_exists else "fresh"
    if hash_exists:
        return "skip"
    if key_exists:
        return "reingest"
    return "fresh"


def is_excluded(storage_key: str, exclude_stems: set[str] | frozenset[str]) -> bool:
    """True if the clip's filename stem (no dir, no extension) is in ``exclude_stems``.

    Case-insensitive. Used to force is_logan=false for known not-Logan clips.
    """
    stem = os.path.splitext(os.path.basename(storage_key))[0]
    return stem.upper() in {s.upper() for s in exclude_stems}


def sha256_file(path: str, chunk_size: int = 1 << 20) -> str:
    """Streaming sha256 hex digest of a file's bytes (the content-dedup key)."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def to_storage_key(path: str) -> str:
    """Normalize a (possibly absolute, local) path to a stable relative storage key.

    Absolute local paths are made relative to the cwd; ``./`` prefixes (from
    ``LOCAL_DATA_DIR=./data``) are collapsed so thumbnail keys match the clean
    ``library/...`` clip-key style.
    """
    if os.path.isabs(path):
        path = os.path.relpath(path)
    return os.path.normpath(path)


# --- orchestration (I/O: DB + ffmpeg + open_clip + Gemini) ------------------

def extract_tier0(content_hash, local, media, *, vision, conn) -> None:
    """Compute + cache the immutable TIER-0 artifacts for a clip - the keystone skip-on-hit.

    A clip is extracted atomically: if its geometry (`cache_windows`) is present at the current
    extractor version, the whole expensive pass - decode, pose, embed, VLM, caption - is SKIPPED.
    (An embedding-model swap reuses the cached frames via a separate backfill, not this path.)
    Caches windows / frames / raw CV signals / per-window VLM / per-window embeddings / caption.
    """
    from ..db import client as db
    from .embed import clip_embed, clip_model_name
    from .frames import extract_frame
    from .quality import (
        frame_signals, motion_series, stability_threshold, subject_present, window_stability,
    )
    from .segment import windows_from_motion

    ev = cache.EXTRACTOR_VERSION
    if db.cache_has("cache_windows", content_hash, ev, conn=conn):
        return  # fully cached - this is the 2-3h pass we no longer pay on re-runs

    av, vv, embv = cache.annotator_version(), cache.vlm_quality_version(), cache.embed_version()
    model = clip_model_name()

    annotation = vision.annotate_clip(local)
    db.cache_put_annotation(content_hash, av, description=annotation.description,
                            energy=annotation.energy, location=annotation.location,
                            is_logan=annotation.is_logan, framing_notes=annotation.framing_notes,
                            conn=conn)

    try:
        motion, times = motion_series(local)
    except Exception:
        motion, times = [], []
    threshold = stability_threshold(motion) if motion else 0.0
    windows = windows_from_motion(motion, times, media["duration_s"], threshold)
    db.cache_put_windows(content_hash, ev, windows, conn=conn)

    thumb_dir = os.path.join(settings.local_data_dir, "thumbs", content_hash[:12])
    os.makedirs(thumb_dir, exist_ok=True)
    for idx, (a, b) in enumerate(windows):
        f0 = os.path.join(thumb_dir, f"{idx:03d}.jpg")     # thumbnail + CLIP + CV frame
        extract_frame(local, a + 0.35 * (b - a), f0)
        exposure, sharpness = frame_signals(f0)
        subj = subject_present(f0)
        stability = window_stability(motion, times, a, b, threshold) if motion else 1.0
        db.cache_put_frame(content_hash, ev, idx, "f0", to_storage_key(f0), conn=conn)
        db.cache_put_cv(content_hash, ev, idx, exposure=exposure, sharpness=sharpness,
                        stability=stability, subject_present=subj, conn=conn)

        f1 = os.path.join(thumb_dir, f"{idx:03d}b.jpg")
        extract_frame(local, a + 0.65 * (b - a), f1)
        usable, vflags = vision.assess_moment([f0, f1])
        os.remove(f1)
        db.cache_put_vlm(content_hash, vv, idx, usable=usable, flags=vflags, conn=conn)

        db.cache_put_embedding(content_hash, embv, idx, clip_embed(f0), model=model, conn=conn)


def ingest_clip(clip_key, *, vision=None, storage=None, force: bool = False) -> dict[str, Any]:
    """Catalog one clip; return a summary dict.

    Phase 9A: resolve -> hash + probe -> dedup -> TIER-0 extract (compute-once, skip on cache
    hit) -> build moments from the cache (TIER-1 `score_window` over cached signals, cheap).
    Returns ``{clip_key, action, clip_id?, moments?}``.
    """
    from ..db import client as db
    from ..storage.base import get_storage
    from .probe import probe_media
    from .quality import score_window

    storage = storage or get_storage()
    key = to_storage_key(clip_key)
    local = storage.ensure_local(key)

    content_hash = sha256_file(local)
    media = probe_media(local)

    conn = db.get_conn()
    try:
        existing = db.get_clip_by_storage_key(key, conn=conn)
        action = ingest_action(
            hash_exists=db.content_hash_exists(content_hash, conn=conn),
            key_exists=existing is not None,
            force=force,
        )
        if action == "skip":
            conn.close()
            return {"clip_key": key, "action": "skip"}

        if vision is None:
            from ..llm import get_captioner

            vision = get_captioner()

        # TIER-0 (expensive, compute-once): caches windows/frames/cv/vlm/embeddings/caption.
        extract_tier0(content_hash, local, media, vision=vision, conn=conn)

        ev, av, vv = cache.EXTRACTOR_VERSION, cache.annotator_version(), cache.vlm_quality_version()
        embv, sv = cache.embed_version(), cache.SCORER_VERSION
        ann = db.read_annotation(content_hash, av, conn=conn) or {}

        # Force is_logan=false for known not-Logan clips the captioner misflags as a solo man.
        excluded = is_excluded(key, IS_LOGAN_EXCLUDE)
        is_logan = False if excluded else ann.get("is_logan")

        clip_id = db.upsert_clip(
            key, content_hash=content_hash, filename=os.path.basename(key),
            duration_s=media["duration_s"], width=media["width"], height=media["height"],
            fps=media["fps"], is_logan=is_logan, notes=EXCLUDE_NOTE if excluded else None,
            conn=conn,
        )
        if action == "reingest":
            # Soft-supersede (don't delete) so prior reels/slots + freshness history survive.
            db.supersede_moments_for_clip(clip_id, conn=conn)

        # TIER-1 + build moments from the cache (cheap: score_window over cached raw signals).
        windows = db.read_windows(content_hash, ev, conn=conn)
        cv = db.read_cv_signals(content_hash, ev, conn=conn)
        vlm = db.read_vlm(content_hash, vv, conn=conn)
        frames = db.read_frames(content_hash, ev, "f0", conn=conn)
        embs = db.read_embeddings(content_hash, embv, conn=conn)

        kept = 0
        for widx, a, b in windows:
            sig, v = cv.get(widx), vlm.get(widx)
            if sig is None or v is None:
                continue
            r = score_window(exposure=sig["exposure"], sharpness=sig["sharpness"],
                             stability=sig["stability"], subject_present=sig["subject_present"],
                             vlm_usable=v["usable"], vlm_flags=v["flags"])
            moment_id = db.upsert_moment(
                clip_id, a, b,
                description=ann.get("description"), energy=ann.get("energy"),
                location=ann.get("location"), is_logan=is_logan,
                framing_notes=ann.get("framing_notes"),
                thumbnail_key=frames.get(widx), window_idx=widx, conn=conn, **r,
            )
            db.cache_put_window_score(content_hash, sv, widx, conn=conn, **r)
            emb = embs.get(widx)
            if emb is not None:
                db.upsert_embedding(moment_id, "clip", emb, model=embv, conn=conn)
            kept += 1

        conn.commit()
        return {"clip_key": key, "action": action, "clip_id": clip_id, "moments": kept}
    except Exception:
        conn.rollback()
        raise
    finally:
        if not conn.closed:
            conn.close()


def ingest_library(*, prefix: str = "library/", limit: int | None = None, force: bool = False) -> list[dict[str, Any]]:
    """Ingest every video under ``prefix`` (incremental by default; only new clips are processed).

    Phase 9A: a cataloged ``storage_key`` skips WITHOUT re-hashing its bytes - so a pass over an
    85GB library that only added a couple clips costs ~the couple clips, not a full re-hash.
    (Footage is immutable; ``--force`` re-hashes to detect byte changes and re-extract Tier-0.)
    """
    from ..db import client as db
    from ..storage.base import get_storage

    storage = get_storage()
    keys = video_keys(storage.list(prefix))
    if limit is not None:
        keys = keys[:limit]

    cataloged = set() if force else db.all_storage_keys()

    vision = None
    results: list[dict[str, Any]] = []
    for key in keys:
        sk = to_storage_key(key)
        if sk in cataloged:
            results.append({"clip_key": sk, "action": "skip"})
            print(f"[{'skip':>8}] {sk}")
            continue
        if vision is None:
            from ..llm import get_captioner

            vision = get_captioner()
        try:
            result = ingest_clip(key, vision=vision, storage=storage, force=force)
        except Exception as exc:
            # A bad file or transient API error must not abort a long batch; the clip
            # left no partial rows (ingest_clip rolls back), so a later run retries it.
            result = {"clip_key": to_storage_key(key), "action": "error",
                      "error": f"{type(exc).__name__}: {exc}"}
        results.append(result)
        if result["action"] == "error":
            print(f"[   ERROR] {result['clip_key']}: {result['error']}")
        else:
            print(f"[{result['action']:>8}] {result['clip_key']}"
                  + (f"  ({result.get('moments')} moments)" if result.get("moments") else ""))
    return results


# --- re-annotation (refresh is_logan + annotation; keep embeddings) ---------

def reannotate_clip(clip_key, *, vision=None, storage=None) -> dict[str, Any]:
    """Re-run the captioner on an already-cataloged clip and refresh its annotation.

    Updates `clips.is_logan` and every moment's inherited annotation fields in place;
    the mechanical windows, thumbnails, and CLIP embeddings are left untouched (no
    re-embedding). Returns ``{clip_key, action, clip_id?, is_logan?, moments?}``.
    """
    from ..db import client as db
    from ..storage.base import get_storage

    storage = storage or get_storage()
    key = to_storage_key(clip_key)
    local = storage.ensure_local(key)

    conn = db.get_conn()
    try:
        existing = db.get_clip_by_storage_key(key, conn=conn)
        if existing is None:
            conn.close()
            return {"clip_key": key, "action": "missing"}

        if vision is None:
            from ..llm import get_captioner

            vision = get_captioner()
        annotation = vision.annotate_clip(local)

        excluded = is_excluded(key, IS_LOGAN_EXCLUDE)
        is_logan = False if excluded else annotation.is_logan
        notes = EXCLUDE_NOTE if excluded else None

        clip_id = existing["id"]
        db.update_clip_annotation(clip_id, is_logan=is_logan, notes=notes, conn=conn)
        n = db.update_moments_annotation(
            clip_id,
            is_logan=is_logan,
            location=annotation.location,
            energy=annotation.energy,
            description=annotation.description,
            framing_notes=annotation.framing_notes,
            conn=conn,
        )
        conn.commit()
        return {"clip_key": key, "action": "reannotated", "clip_id": clip_id,
                "is_logan": is_logan, "moments": n}
    except Exception:
        conn.rollback()
        raise
    finally:
        if not conn.closed:
            conn.close()


def reannotate_library(*, prefix: str = "library/", limit: int | None = None) -> list[dict[str, Any]]:
    """Re-annotate every already-cataloged video under ``prefix`` (keeps embeddings)."""
    from ..storage.base import get_storage

    storage = get_storage()
    keys = video_keys(storage.list(prefix))
    if limit is not None:
        keys = keys[:limit]

    vision = None
    results: list[dict[str, Any]] = []
    for key in keys:
        if vision is None:
            from ..llm import get_captioner

            vision = get_captioner()
        result = reannotate_clip(key, vision=vision, storage=storage)
        results.append(result)
        flag = result.get("is_logan")
        print(f"[{result['action']:>11}] {result['clip_key']}"
              + (f"  is_logan={flag}" if flag is not None else ""))
    return results
