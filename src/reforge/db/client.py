"""Postgres / pgvector access for Reforge.

A connection helper (`get_conn`, with pgvector registered), idempotent writers for
the catalog (`upsert_clip`, `upsert_moment`, `upsert_embedding`), and a cosine KNN
reader over the moment embedding tables (`knn`). See `db/schema.sql` for the tables.

The query-building logic (`table_for_kind`, `build_filter_clause`) is factored out as
pure, import-light functions so it can be unit-tested without a database. Heavy/optional
imports (psycopg, pgvector, numpy) are loaded lazily inside the I/O functions, so the
pure helpers import even in a minimal environment.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, Sequence

from ..config import settings

# Embedding kind -> (table, dimension). Dims must match the live schema.
# caption = OpenAI text-embedding-3-small at dimensions=768 (migration 0007 altered
# the table from the schema.sql placeholder 1536 - the 768 truncation is documented
# and keeps the caption lane the same dim/storage as the CLIP lane).
EMBEDDING_TABLES: dict[str, tuple[str, int]] = {
    "clip": ("moment_clip_embeddings", 768),
    "caption": ("moment_caption_embeddings", 768),
    "marengo": ("moment_marengo_embeddings", 1024),
}

# Tables the schema is expected to define (used by `missing_tables`).
EXPECTED_TABLES: tuple[str, ...] = (
    "clips",
    "moments",
    "moment_clip_embeddings",
    "moment_caption_embeddings",
    "moment_marengo_embeddings",
    "reels",
    "slots",
    "eval_scores",
)

# Moment columns returned by `knn`, in select order.
_MOMENT_COLS: tuple[str, ...] = (
    "id",
    "clip_id",
    "start_s",
    "end_s",
    "description",
    "energy",
    "location",
    "is_logan",
    "framing_notes",
    "thumbnail_key",
    "quality_score",
)

# Filter keys honored by `build_filter_clause`, mapped to their moments column.
_FILTER_COLS: dict[str, str] = {
    "is_logan": "m.is_logan",
    "location": "m.location",
}


# --- pure query-building helpers (unit-tested, no DB) -----------------------

def table_for_kind(kind: str) -> tuple[str, int]:
    """Map an embedding kind to its (table_name, dimension).

    Raises ValueError for an unknown kind so a typo can't silently target the
    wrong table.
    """
    try:
        return EMBEDDING_TABLES[kind]
    except KeyError:
        valid = ", ".join(sorted(EMBEDDING_TABLES))
        raise ValueError(f"unknown embedding kind {kind!r}; expected one of: {valid}") from None


def build_filter_clause(filters: dict[str, Any] | None) -> tuple[str, dict[str, Any]]:
    """Build the SQL filter fragment + params for a moments query.

    Returns ``(sql, params)`` where ``sql`` is appended after a base ``WHERE TRUE``
    (e.g. ``" AND m.is_logan = %(f_is_logan)s"``). Keys whose value is None are
    skipped so callers can pass a sparse dict, and ``False`` is preserved (it is a
    real filter value, unlike None). Unknown keys are ignored.
    """
    if not filters:
        return "", {}
    clauses: list[str] = []
    params: dict[str, Any] = {}
    for key, column in _FILTER_COLS.items():
        value = filters.get(key)
        if value is None:
            continue
        param = f"f_{key}"
        clauses.append(f"{column} = %({param})s")
        params[param] = value
    sql = "".join(f" AND {c}" for c in clauses)
    return sql, params


def bm25_rank_to_distance(rank: float) -> float:
    """Convert a ``ts_rank`` score (DESC, bigger = better) to a distance-like value (pure).

    ``1 / (1 + rank)`` maps rank 0 -> 1.0 and grows asymptotically toward 0 for strong
    matches, so BM25 rows sort/merge the same way as kNN rows (smaller = nearer) in
    ``rrf_fuse`` / ``merge_candidates``.
    """
    return 1.0 / (1.0 + max(rank, 0.0))


# --- connection -------------------------------------------------------------

def get_conn():
    """Open a psycopg3 connection to ``settings.database_url`` with pgvector registered.

    The caller owns the returned connection (commit/close). The catalog writers below
    open and manage their own connection when one isn't passed in.
    """
    import psycopg
    from pgvector.psycopg import register_vector

    if not settings.database_url:
        raise RuntimeError("DATABASE_URL is not set; add it to .env")
    conn = psycopg.connect(settings.database_url)
    register_vector(conn)
    return conn


@contextmanager
def _session(conn=None) -> Iterator[Any]:
    """Yield a connection. If we opened it, commit on success / rollback on error, always close.

    If a connection is passed in, the caller owns its transaction lifecycle.
    """
    own = conn is None
    c = conn or get_conn()
    try:
        yield c
        if own:
            c.commit()
    except Exception:
        if own:
            c.rollback()
        raise
    finally:
        if own:
            c.close()


# --- writers ----------------------------------------------------------------

def upsert_clip(
    storage_key: str,
    *,
    content_hash: str | None = None,
    filename: str | None = None,
    duration_s: float | None = None,
    width: int | None = None,
    height: int | None = None,
    fps: float | None = None,
    is_logan: bool | None = None,
    notes: str | None = None,
    conn=None,
) -> int:
    """Insert or update a clip by its unique ``storage_key``; return the clip id.

    Re-ingesting refreshes the computed media fields and ``content_hash``. ``is_logan``
    and ``notes`` are preserved (COALESCE) when re-ingested without a value, so a manual
    label isn't wiped by a re-run that doesn't know it.
    """
    sql = """
        INSERT INTO clips (storage_key, content_hash, filename, duration_s, width, height,
                           fps, is_logan, notes)
        VALUES (%(storage_key)s, %(content_hash)s, %(filename)s, %(duration_s)s, %(width)s,
                %(height)s, %(fps)s, %(is_logan)s, %(notes)s)
        ON CONFLICT (storage_key) DO UPDATE SET
            content_hash = COALESCE(EXCLUDED.content_hash, clips.content_hash),
            filename     = EXCLUDED.filename,
            duration_s   = EXCLUDED.duration_s,
            width        = EXCLUDED.width,
            height       = EXCLUDED.height,
            fps          = EXCLUDED.fps,
            is_logan     = COALESCE(EXCLUDED.is_logan, clips.is_logan),
            notes        = COALESCE(EXCLUDED.notes, clips.notes)
        RETURNING id
    """
    params = dict(
        storage_key=storage_key,
        content_hash=content_hash,
        filename=filename,
        duration_s=duration_s,
        width=width,
        height=height,
        fps=fps,
        is_logan=is_logan,
        notes=notes,
    )
    with _session(conn) as c, c.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchone()[0]


def get_clip_by_storage_key(storage_key: str, conn=None) -> dict[str, Any] | None:
    """Return ``{id, content_hash}`` for a clip with this storage_key, or None."""
    sql = "SELECT id, content_hash FROM clips WHERE storage_key = %(k)s"
    with _session(conn) as c, c.cursor() as cur:
        cur.execute(sql, {"k": storage_key})
        row = cur.fetchone()
    return {"id": row[0], "content_hash": row[1]} if row else None


def all_storage_keys(conn=None) -> set[str]:
    """Every cataloged clip storage_key - for the incremental fast-skip (avoid re-hashing known files)."""
    with _session(conn) as c, c.cursor() as cur:
        cur.execute("SELECT storage_key FROM clips")
        return {r[0] for r in cur.fetchall()}


def content_hash_exists(content_hash: str, conn=None) -> bool:
    """True if any clip already has this content hash (content-level dedup)."""
    sql = "SELECT 1 FROM clips WHERE content_hash = %(h)s LIMIT 1"
    with _session(conn) as c, c.cursor() as cur:
        cur.execute(sql, {"h": content_hash})
        return cur.fetchone() is not None


def delete_moments_for_clip(clip_id: int, conn=None) -> int:
    """Delete all moments for a clip (cascades to their embeddings); return the count.

    Used to refresh a changed clip on re-ingest, since ``moments`` has no unique key.
    """
    sql = "DELETE FROM moments WHERE clip_id = %(id)s"
    with _session(conn) as c, c.cursor() as cur:
        cur.execute(sql, {"id": clip_id})
        return cur.rowcount


def supersede_moments_for_clip(clip_id: int, conn=None) -> int:
    """Mark a clip's existing moments superseded (re-ingest) without deleting them.

    Preserves prior reels/slots references + their freshness history; retrieval ignores
    superseded moments. Returns the count marked.
    """
    sql = "UPDATE moments SET superseded = true WHERE clip_id = %(id)s AND superseded = false"
    with _session(conn) as c, c.cursor() as cur:
        cur.execute(sql, {"id": clip_id})
        return cur.rowcount


def update_clip_annotation(clip_id: int, *, is_logan: bool | None, notes: str | None = None, conn=None) -> None:
    """Overwrite a clip's ``is_logan`` (and ``notes``) - used by re-annotation.

    Unlike ``upsert_clip`` (which COALESCEs to preserve labels), this is a deliberate
    refresh, so it sets the values directly.
    """
    sql = "UPDATE clips SET is_logan = %(is_logan)s, notes = %(notes)s WHERE id = %(id)s"
    with _session(conn) as c, c.cursor() as cur:
        cur.execute(sql, {"id": clip_id, "is_logan": is_logan, "notes": notes})


def update_moments_annotation(
    clip_id: int,
    *,
    is_logan: bool | None,
    location: str | None,
    energy: str | None,
    description: str | None,
    framing_notes: str | None,
    conn=None,
) -> int:
    """Refresh the inherited annotation fields on every moment of a clip; return the count.

    Leaves ``start_s``/``end_s``/``thumbnail_key`` and the CLIP embeddings untouched, so
    re-annotation does not require re-embedding.
    """
    sql = """
        UPDATE moments SET
            is_logan      = %(is_logan)s,
            location      = %(location)s,
            energy        = %(energy)s,
            description   = %(description)s,
            framing_notes = %(framing_notes)s
        WHERE clip_id = %(clip_id)s
    """
    params = dict(
        clip_id=clip_id,
        is_logan=is_logan,
        location=location,
        energy=energy,
        description=description,
        framing_notes=framing_notes,
    )
    with _session(conn) as c, c.cursor() as cur:
        cur.execute(sql, params)
        return cur.rowcount


def upsert_moment(
    clip_id: int,
    start_s: float,
    end_s: float,
    *,
    description: str | None = None,
    energy: str | None = None,
    location: str | None = None,
    is_logan: bool | None = None,
    framing_notes: str | None = None,
    thumbnail_key: str | None = None,
    cv_quality: float | None = None,
    vlm_quality: float | None = None,
    quality_score: float | None = None,
    quality_flags: list[str] | None = None,
    window_idx: int | None = None,
    conn=None,
) -> int:
    """Insert a moment for a clip (with Phase 3.7 quality fields); return its id.

    ``moments`` has no natural unique key in the schema, so this is an insert.
    Re-ingest idempotency is handled at the clip level (skip already-ingested by
    ``storage_key``) in the Phase 2 pipeline, per BUILD_SPEC. ``window_idx`` (Phase 9A)
    links the moment to its cached Tier-0 window for re-scoring.
    """
    from psycopg.types.json import Json

    sql = """
        INSERT INTO moments (clip_id, start_s, end_s, description, energy, location,
                             is_logan, framing_notes, thumbnail_key,
                             cv_quality, vlm_quality, quality_score, quality_flags, window_idx)
        VALUES (%(clip_id)s, %(start_s)s, %(end_s)s, %(description)s, %(energy)s,
                %(location)s, %(is_logan)s, %(framing_notes)s, %(thumbnail_key)s,
                %(cv_quality)s, %(vlm_quality)s, %(quality_score)s, %(quality_flags)s, %(window_idx)s)
        RETURNING id
    """
    params = dict(
        clip_id=clip_id,
        start_s=start_s,
        end_s=end_s,
        description=description,
        energy=energy,
        location=location,
        is_logan=is_logan,
        framing_notes=framing_notes,
        thumbnail_key=thumbnail_key,
        cv_quality=cv_quality,
        vlm_quality=vlm_quality,
        quality_score=quality_score,
        quality_flags=Json(quality_flags) if quality_flags is not None else None,
        window_idx=window_idx,
    )
    with _session(conn) as c, c.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchone()[0]


def upsert_embedding(
    moment_id: int,
    kind: str,
    vec: Sequence[float],
    *,
    model: str | None = None,
    conn=None,
) -> None:
    """Upsert one embedding for a moment. ``kind`` is one of clip / caption / marengo.

    The vector is validated against the kind's expected dimension before write, so a
    mismatched embedding fails loudly instead of erroring deep in Postgres.
    """
    import numpy as np

    table, dim = table_for_kind(kind)
    arr = np.asarray(vec, dtype=np.float32)
    if arr.ndim != 1 or arr.shape[0] != dim:
        raise ValueError(f"{kind} embedding must be 1-D length {dim}, got shape {arr.shape}")
    # `table` comes from the validated mapping, never user input - safe to interpolate.
    sql = f"""
        INSERT INTO {table} (moment_id, model, embedding)
        VALUES (%(moment_id)s, %(model)s, %(embedding)s)
        ON CONFLICT (moment_id) DO UPDATE SET
            model     = EXCLUDED.model,
            embedding = EXCLUDED.embedding
    """
    params = dict(moment_id=moment_id, model=model, embedding=arr)
    with _session(conn) as c, c.cursor() as cur:
        cur.execute(sql, params)


# --- run records (reels / slots) --------------------------------------------

def insert_reel(
    *,
    reference_key: str | None = None,
    total_duration: float | None = None,
    cut_grid: Any | None = None,
    brief: dict[str, Any] | None = None,
    status: str = "draft",
    output_key: str | None = None,
    conn=None,
) -> int:
    """Insert a recreation-run `reels` row (with the Phase 3.5 brief); return its id."""
    from psycopg.types.json import Json

    sql = """
        INSERT INTO reels (reference_key, total_duration, cut_grid, brief, status, output_key)
        VALUES (%(reference_key)s, %(total_duration)s, %(cut_grid)s, %(brief)s, %(status)s, %(output_key)s)
        RETURNING id
    """
    params = dict(
        reference_key=reference_key,
        total_duration=total_duration,
        cut_grid=Json(cut_grid) if cut_grid is not None else None,
        brief=Json(brief) if brief is not None else None,
        status=status,
        output_key=output_key,
    )
    with _session(conn) as c, c.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchone()[0]


def insert_slot(
    reel_id: int,
    idx: int,
    *,
    start_s: float | None = None,
    end_s: float | None = None,
    ref_description: str | None = None,
    ref_text_overlay: str | None = None,
    ref_energy: str | None = None,
    chosen_moment_id: int | None = None,
    params: dict[str, Any] | None = None,
    reason: str | None = None,
    conn=None,
) -> int:
    """Insert one `slots` row (the chosen moment + framing params + rerank reason); return id."""
    from psycopg.types.json import Json

    sql = """
        INSERT INTO slots (reel_id, idx, start_s, end_s, ref_description, ref_text_overlay,
                           ref_energy, chosen_moment_id, params, reason)
        VALUES (%(reel_id)s, %(idx)s, %(start_s)s, %(end_s)s, %(ref_description)s,
                %(ref_text_overlay)s, %(ref_energy)s, %(chosen_moment_id)s, %(params)s, %(reason)s)
        RETURNING id
    """
    values = dict(
        reel_id=reel_id, idx=idx, start_s=start_s, end_s=end_s,
        ref_description=ref_description, ref_text_overlay=ref_text_overlay, ref_energy=ref_energy,
        chosen_moment_id=chosen_moment_id,
        params=Json(params) if params is not None else None,
        reason=reason,
    )
    with _session(conn) as c, c.cursor() as cur:
        cur.execute(sql, values)
        return cur.fetchone()[0]


def clip_usage(conn=None) -> dict[int, dict[str, Any]]:
    """Per-clip historical usage from past runs: ``{clip_id: {count, recency}}``.

    ``count`` = how many slots across all reels chose a moment of that clip; ``recency`` in
    (0,1], 1.0 = used in the most recent reel. Feeds the Phase 3.6 freshness term so picks
    rotate across reels (explore/exploit). Empty when no runs are recorded yet.
    """
    sql = """
        SELECT m.clip_id, count(*) AS n, max(s.reel_id) AS last_reel
        FROM slots s JOIN moments m ON m.id = s.chosen_moment_id
        WHERE s.chosen_moment_id IS NOT NULL
        GROUP BY m.clip_id
    """
    with _session(conn) as c, c.cursor() as cur:
        cur.execute(sql, ())
        rows = cur.fetchall()
        cur.execute("SELECT max(id) FROM reels")
        max_reel = cur.fetchone()[0] or 1
    return {
        clip_id: {"count": n, "recency": (last_reel / max_reel) if last_reel else 0.0}
        for clip_id, n, last_reel in rows
    }


# --- reader -----------------------------------------------------------------

def knn(
    query_vec: Sequence[float],
    kind: str,
    k: int = 5,
    filters: dict[str, Any] | None = None,
    conn=None,
) -> list[dict[str, Any]]:
    """Cosine KNN over the ``kind`` embedding table, joined to moments and filtered.

    Returns up to ``k`` moment dicts (each with a ``distance``, smaller = nearer),
    nearest first. ``filters`` supports ``is_logan`` and ``location`` on the moment.
    """
    import numpy as np

    table, dim = table_for_kind(kind)
    arr = np.asarray(query_vec, dtype=np.float32)
    if arr.ndim != 1 or arr.shape[0] != dim:
        raise ValueError(f"{kind} query vector must be 1-D length {dim}, got shape {arr.shape}")

    filter_sql, filter_params = build_filter_clause(filters)
    cols = ", ".join(f"m.{c}" for c in _MOMENT_COLS)
    if kind == "clip":
        # Phase 9A: read the materialized retrieval view (current embedding + current scores).
        # Pre-filtered to superseded=false; identical columns/order to the prior live join.
        sql = f"""
            SELECT {cols}, m.storage_key, m.embedding <=> %(qv)s AS distance
            FROM mv_moment_retrieval m
            WHERE TRUE{filter_sql}
            ORDER BY distance
            LIMIT %(k)s
        """
    else:
        sql = f"""
            SELECT {cols}, c.storage_key, e.embedding <=> %(qv)s AS distance
            FROM {table} e
            JOIN moments m ON m.id = e.moment_id
            JOIN clips c ON c.id = m.clip_id
            WHERE m.superseded = false{filter_sql}
            ORDER BY distance
            LIMIT %(k)s
        """
    params = {"qv": arr, "k": k, **filter_params}
    with _session(conn) as c, c.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
    keys = (*_MOMENT_COLS, "storage_key", "distance")
    return [dict(zip(keys, row)) for row in rows]


def caption_knn(
    query_vec: Sequence[float],
    k: int = 5,
    filters: dict[str, Any] | None = None,
    conn=None,
) -> list[dict[str, Any]]:
    """Cosine KNN over the Phase 9B caption-text embeddings (same row shape as `knn`).

    Joined to moments + clips with ``superseded = false`` and the same filter support
    (``is_logan`` / ``location``), so a hybrid lane can be fused 1:1 with the CLIP lane.
    """
    return knn(query_vec, "caption", k=k, filters=filters, conn=conn)


def bm25(
    query_text: str,
    k: int = 5,
    filters: dict[str, Any] | None = None,
    conn=None,
) -> list[dict[str, Any]]:
    """Full-text search over moment captions (``moments.caption_tsv``, migration 0007).

    Same row shape as `knn`: ``ts_rank`` is DESC (bigger = better), so it is converted
    to a distance-like value via ``bm25_rank_to_distance`` for merge compatibility.
    Respects ``superseded = false`` and the ``is_logan`` / ``location`` filters. Uses
    ``websearch_to_tsquery`` so free-form shot queries never raise on syntax.
    """
    if not (query_text or "").strip():
        return []
    filter_sql, filter_params = build_filter_clause(filters)
    cols = ", ".join(f"m.{c}" for c in _MOMENT_COLS)
    sql = f"""
        SELECT {cols}, c.storage_key, ts_rank(m.caption_tsv, q.query) AS rank
        FROM moments m
        JOIN clips c ON c.id = m.clip_id
        CROSS JOIN websearch_to_tsquery('english', %(q)s) AS q(query)
        WHERE m.superseded = false AND m.caption_tsv @@ q.query{filter_sql}
        ORDER BY rank DESC
        LIMIT %(k)s
    """
    params = {"q": query_text, "k": k, **filter_params}
    with _session(conn) as c, c.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
    keys = (*_MOMENT_COLS, "storage_key", "distance")
    return [dict(zip(keys, (*row[:-1], bm25_rank_to_distance(row[-1])))) for row in rows]


# --- schema verification ----------------------------------------------------

def missing_tables(conn=None) -> list[str]:
    """Return any EXPECTED_TABLES not present in the public schema (empty == all present)."""
    sql = """
        SELECT table_name FROM information_schema.tables
        WHERE table_schema = 'public' AND table_name = ANY(%(names)s)
    """
    with _session(conn) as c, c.cursor() as cur:
        cur.execute(sql, {"names": list(EXPECTED_TABLES)})
        present = {r[0] for r in cur.fetchall()}
    return [t for t in EXPECTED_TABLES if t not in present]


# --- Phase 9A: Tier-0/Tier-1 content-addressed cache ------------------------
# Append-only writers (ON CONFLICT DO NOTHING - a (content_hash, version) row is immutable),
# skip-on-hit predicates, and Tier-0 raw-signal readers for the cheap Tier-1 re-score.
# Table names come from this fixed internal map (never user input), so `_cache_has` interpolates safely.
_CACHE_VERSION_COL: dict[str, str] = {
    "cache_windows": "extractor_version",
    "cache_cv_signals": "extractor_version",
    "cache_annotations": "annotator_version",
    "cache_embeddings": "embed_version",
    "cache_vlm_assessments": "vlm_version",
    "window_scores": "scorer_version",
}


def cache_has(table: str, content_hash: str, version: str, conn=None) -> bool:
    """Skip-on-hit: True if this clip already has cached rows for ``table`` at ``version``."""
    col = _CACHE_VERSION_COL[table]  # KeyError on an unknown table = a bug, not user input
    sql = f"SELECT 1 FROM {table} WHERE content_hash = %s AND {col} = %s LIMIT 1"
    with _session(conn) as c, c.cursor() as cur:
        cur.execute(sql, (content_hash, version))
        return cur.fetchone() is not None


def cache_put_windows(content_hash, extractor_version, windows, conn=None) -> None:
    """Append-only: cache segmentation windows ``[(start_s, end_s), ...]`` (window_idx = order)."""
    sql = """INSERT INTO cache_windows (content_hash, extractor_version, window_idx, start_s, end_s)
             VALUES (%s, %s, %s, %s, %s) ON CONFLICT DO NOTHING"""
    with _session(conn) as c, c.cursor() as cur:
        for i, (a, b) in enumerate(windows):
            cur.execute(sql, (content_hash, extractor_version, i, a, b))


def cache_put_frame(content_hash, extractor_version, window_idx, frame_role, storage_key, conn=None) -> None:
    """Append-only: record an extracted frame artifact's storage key."""
    sql = """INSERT INTO cache_frames (content_hash, extractor_version, window_idx, frame_role, storage_key)
             VALUES (%s, %s, %s, %s, %s) ON CONFLICT DO NOTHING"""
    with _session(conn) as c, c.cursor() as cur:
        cur.execute(sql, (content_hash, extractor_version, window_idx, frame_role, storage_key))


def cache_put_cv(content_hash, extractor_version, window_idx, *, exposure, sharpness,
                 stability, subject_present, conn=None) -> None:
    """Append-only: cache the raw CV signals for one window (read at re-score time)."""
    sql = """INSERT INTO cache_cv_signals
               (content_hash, extractor_version, window_idx, exposure, sharpness, stability, subject_present)
             VALUES (%s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING"""
    with _session(conn) as c, c.cursor() as cur:
        cur.execute(sql, (content_hash, extractor_version, window_idx,
                          exposure, sharpness, stability, subject_present))


def cache_put_vlm(content_hash, vlm_version, window_idx, *, usable, flags, conn=None) -> None:
    """Append-only: cache the raw VLM assessment (usable + flags) for one window."""
    from psycopg.types.json import Json

    sql = """INSERT INTO cache_vlm_assessments (content_hash, vlm_version, window_idx, usable, flags)
             VALUES (%s, %s, %s, %s, %s) ON CONFLICT DO NOTHING"""
    with _session(conn) as c, c.cursor() as cur:
        cur.execute(sql, (content_hash, vlm_version, window_idx, usable, Json(list(flags or []))))


def cache_put_annotation(content_hash, annotator_version, *, description, energy, location,
                         is_logan, framing_notes, conn=None) -> None:
    """Append-only: cache the per-clip VLM caption/annotation."""
    sql = """INSERT INTO cache_annotations
               (content_hash, annotator_version, description, energy, location, is_logan, framing_notes)
             VALUES (%s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING"""
    with _session(conn) as c, c.cursor() as cur:
        cur.execute(sql, (content_hash, annotator_version, description, energy, location,
                          is_logan, framing_notes))


def cache_put_embedding(content_hash, embed_version, window_idx, vec, *, model=None, conn=None) -> None:
    """Append-only: cache a base visual embedding for one window (shadow = a new embed_version)."""
    import numpy as np

    arr = np.asarray(vec, dtype=np.float32)
    if arr.ndim != 1 or arr.shape[0] != 768:
        raise ValueError(f"cache embedding must be 1-D length 768, got shape {arr.shape}")
    sql = """INSERT INTO cache_embeddings (content_hash, embed_version, window_idx, model, embedding)
             VALUES (%s, %s, %s, %s, %s) ON CONFLICT DO NOTHING"""
    with _session(conn) as c, c.cursor() as cur:
        cur.execute(sql, (content_hash, embed_version, window_idx, model, arr))


def cache_put_window_score(content_hash, scorer_version, window_idx, *, cv_quality, vlm_quality,
                           quality_score, quality_flags, conn=None) -> None:
    """TIER-1 write: upsert the derived score for one window (re-running a scorer overwrites it)."""
    from psycopg.types.json import Json

    sql = """INSERT INTO window_scores
               (content_hash, scorer_version, window_idx, cv_quality, vlm_quality, quality_score, quality_flags)
             VALUES (%s, %s, %s, %s, %s, %s, %s)
             ON CONFLICT (content_hash, scorer_version, window_idx) DO UPDATE SET
               cv_quality = EXCLUDED.cv_quality, vlm_quality = EXCLUDED.vlm_quality,
               quality_score = EXCLUDED.quality_score, quality_flags = EXCLUDED.quality_flags"""
    with _session(conn) as c, c.cursor() as cur:
        cur.execute(sql, (content_hash, scorer_version, window_idx, cv_quality, vlm_quality,
                          quality_score, Json(list(quality_flags or []))))


def read_cv_signals(content_hash, extractor_version, conn=None) -> dict[int, dict[str, Any]]:
    """Tier-0 read: ``{window_idx: {exposure, sharpness, stability, subject_present}}``."""
    sql = """SELECT window_idx, exposure, sharpness, stability, subject_present
             FROM cache_cv_signals WHERE content_hash = %s AND extractor_version = %s"""
    with _session(conn) as c, c.cursor() as cur:
        cur.execute(sql, (content_hash, extractor_version))
        return {r[0]: {"exposure": r[1], "sharpness": r[2], "stability": r[3],
                       "subject_present": r[4]} for r in cur.fetchall()}


def read_vlm(content_hash, vlm_version, conn=None) -> dict[int, dict[str, Any]]:
    """Tier-0 read: ``{window_idx: {usable, flags}}``."""
    sql = "SELECT window_idx, usable, flags FROM cache_vlm_assessments WHERE content_hash = %s AND vlm_version = %s"
    with _session(conn) as c, c.cursor() as cur:
        cur.execute(sql, (content_hash, vlm_version))
        return {r[0]: {"usable": r[1], "flags": r[2] or []} for r in cur.fetchall()}


def read_windows(content_hash, extractor_version, conn=None) -> list[tuple[int, float, float]]:
    """Tier-0 read: ``[(window_idx, start_s, end_s), ...]`` in window order."""
    sql = """SELECT window_idx, start_s, end_s FROM cache_windows
             WHERE content_hash = %s AND extractor_version = %s ORDER BY window_idx"""
    with _session(conn) as c, c.cursor() as cur:
        cur.execute(sql, (content_hash, extractor_version))
        return [(r[0], r[1], r[2]) for r in cur.fetchall()]


def read_frames(content_hash, extractor_version, frame_role="f0", conn=None) -> dict[int, str]:
    """Tier-0 read: ``{window_idx: storage_key}`` for one frame role."""
    sql = """SELECT window_idx, storage_key FROM cache_frames
             WHERE content_hash = %s AND extractor_version = %s AND frame_role = %s"""
    with _session(conn) as c, c.cursor() as cur:
        cur.execute(sql, (content_hash, extractor_version, frame_role))
        return {r[0]: r[1] for r in cur.fetchall()}


def read_annotation(content_hash, annotator_version, conn=None) -> dict[str, Any] | None:
    """Tier-0 read: the cached per-clip annotation, or None."""
    sql = """SELECT description, energy, location, is_logan, framing_notes
             FROM cache_annotations WHERE content_hash = %s AND annotator_version = %s"""
    with _session(conn) as c, c.cursor() as cur:
        cur.execute(sql, (content_hash, annotator_version))
        row = cur.fetchone()
    if row is None:
        return None
    return {"description": row[0], "energy": row[1], "location": row[2],
            "is_logan": row[3], "framing_notes": row[4]}


def read_embeddings(content_hash, embed_version, conn=None) -> dict[int, Any]:
    """Tier-0 read: ``{window_idx: embedding}`` for one embed version."""
    sql = """SELECT window_idx, embedding FROM cache_embeddings
             WHERE content_hash = %s AND embed_version = %s"""
    with _session(conn) as c, c.cursor() as cur:
        cur.execute(sql, (content_hash, embed_version))
        return {r[0]: r[1] for r in cur.fetchall()}


def read_all_cv_signals(extractor_version, conn=None) -> dict[tuple[str, int], dict[str, Any]]:
    """Bulk Tier-0 read for the whole-library re-score: ``{(content_hash, window_idx): signals}``."""
    sql = """SELECT content_hash, window_idx, exposure, sharpness, stability, subject_present
             FROM cache_cv_signals WHERE extractor_version = %s"""
    with _session(conn) as c, c.cursor() as cur:
        cur.execute(sql, (extractor_version,))
        return {(r[0], r[1]): {"exposure": r[2], "sharpness": r[3], "stability": r[4],
                               "subject_present": r[5]} for r in cur.fetchall()}


def read_all_vlm(vlm_version, conn=None) -> dict[tuple[str, int], dict[str, Any]]:
    """Bulk Tier-0 read: ``{(content_hash, window_idx): {usable, flags}}``."""
    sql = "SELECT content_hash, window_idx, usable, flags FROM cache_vlm_assessments WHERE vlm_version = %s"
    with _session(conn) as c, c.cursor() as cur:
        cur.execute(sql, (vlm_version,))
        return {(r[0], r[1]): {"usable": r[2], "flags": r[3] or []} for r in cur.fetchall()}


def bulk_put_window_scores(scorer_version, rows, conn=None) -> int:
    """Batch-upsert Tier-1 scores. ``rows`` = ``[(content_hash, window_idx, cv, vlm, score, flags), ...]``.

    One ``executemany`` instead of a round-trip per moment - this is what makes a whole-library
    re-score seconds, not minutes, over a remote DB.
    """
    from psycopg.types.json import Json

    sql = """INSERT INTO window_scores
               (content_hash, scorer_version, window_idx, cv_quality, vlm_quality, quality_score, quality_flags)
             VALUES (%s, %s, %s, %s, %s, %s, %s)
             ON CONFLICT (content_hash, scorer_version, window_idx) DO UPDATE SET
               cv_quality = EXCLUDED.cv_quality, vlm_quality = EXCLUDED.vlm_quality,
               quality_score = EXCLUDED.quality_score, quality_flags = EXCLUDED.quality_flags"""
    payload = [(h, scorer_version, w, cvq, vlmq, sc, Json(list(fl or [])))
               for (h, w, cvq, vlmq, sc, fl) in rows]
    with _session(conn) as c, c.cursor() as cur:
        cur.executemany(sql, payload)
    return len(payload)


def cached_content_hashes(table: str, version: str, conn=None) -> set[str]:
    """All content_hashes already cached in ``table`` at ``version`` (for embedding-swap backfill)."""
    col = _CACHE_VERSION_COL[table]
    sql = f"SELECT DISTINCT content_hash FROM {table} WHERE {col} = %s"
    with _session(conn) as c, c.cursor() as cur:
        cur.execute(sql, (version,))
        return {r[0] for r in cur.fetchall()}


def set_moment_window_idx(moment_id, window_idx, conn=None) -> None:
    """Link an existing moment to its cached window (backfill)."""
    with _session(conn) as c, c.cursor() as cur:
        cur.execute("UPDATE moments SET window_idx = %s WHERE id = %s", (window_idx, moment_id))


def promote_window_scores(scorer_version: str, conn=None) -> int:
    """Activate a versioned Tier-1 score set onto the live moments (one UPDATE, seconds).

    Copies window_scores@scorer_version onto each active moment via (content_hash, window_idx).
    Caller REFRESHes the retrieval view after. Returns rows updated.
    """
    sql = """
        UPDATE moments m
        SET cv_quality = ws.cv_quality, vlm_quality = ws.vlm_quality,
            quality_score = ws.quality_score, quality_flags = ws.quality_flags
        FROM clips c, window_scores ws
        WHERE m.clip_id = c.id AND m.superseded = false AND m.window_idx IS NOT NULL
          AND ws.content_hash = c.content_hash AND ws.window_idx = m.window_idx
          AND ws.scorer_version = %s
    """
    with _session(conn) as c, c.cursor() as cur:
        cur.execute(sql, (scorer_version,))
        return cur.rowcount


def refresh_retrieval_view(conn=None, concurrently: bool = True) -> None:
    """REFRESH the retrieval materialized view (after a scorer promote / new ingest)."""
    mode = "CONCURRENTLY " if concurrently else ""
    with _session(conn) as c, c.cursor() as cur:
        cur.execute(f"REFRESH MATERIALIZED VIEW {mode}mv_moment_retrieval")


def active_moments_for_rescore(conn=None) -> list[dict[str, Any]]:
    """Every active moment with its (content_hash, window_idx) for the Tier-1 re-score join."""
    sql = """SELECT m.id, c.content_hash, m.window_idx
             FROM moments m JOIN clips c ON c.id = m.clip_id
             WHERE m.superseded = false AND m.window_idx IS NOT NULL AND c.content_hash IS NOT NULL"""
    with _session(conn) as c, c.cursor() as cur:
        cur.execute(sql)
        return [{"moment_id": r[0], "content_hash": r[1], "window_idx": r[2]} for r in cur.fetchall()]
