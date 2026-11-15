"""App-layer DB access: a thin, injectable read/write seam over ``db.client.get_conn``.

Deliberately its OWN module - ``db/client.py`` (owned elsewhere) stays untouched; only
its stable ``get_conn`` is used. ``Queries`` is what ``create_app`` injects (tests pass
a fake with the same method names). ``alternates_json`` and ``thumb_url`` are pure
helpers (unit-tested, no DB).
"""
from __future__ import annotations

import os
import random
import threading
from typing import Any
from urllib.parse import quote

from reforge.db.client import get_conn

# Borderline quality band: moments here are where a human label is worth the tap.
QUALITY_BAND = (0.35, 0.75)

# Reel-level grades (the reviewer's confidence dial). Positive grades endorse every
# unflagged beat, so one tap teaches the judge without a per-beat Keep.
POSITIVE_GRADES = ("post_worthy", "would_test")
GRADES = (*POSITIVE_GRADES, "not_it")

# Known reference reels (repo-local mp4s) - the fallback run gallery when the DB has
# no prior runs for them yet. Unioned with distinct reels.reference_key.
KNOWN_REFERENCES = (
    "reference/Bible-gym.mp4",
    "reference/DISCIPLINE.mp4",
    "reference/grace.mp4",
)

# One slot row as the app sees it (chosen moment joined in; swapped moment wins).
# Columns + joins are factored so the UPDATE..RETURNING verdict writes reuse the exact
# same shape (one round trip returns the refreshed joined row - no drift with _SLOT_SQL).
_SLOT_COLS = """s.id, s.reel_id, s.idx, s.start_s, s.end_s, s.ref_description, s.reason,
           s.human_verdict, s.human_note, s.chosen_moment_id, s.swapped_moment_id,
           s.candidates, s.params, m.thumbnail_key, m.description, m.quality_score,
           cl.storage_key AS clip_key, m.start_s AS moment_start, m.end_s AS moment_end"""
_SLOT_JOINS = """LEFT JOIN moments m ON m.id = COALESCE(s.swapped_moment_id, s.chosen_moment_id)
    LEFT JOIN clips cl ON cl.id = m.clip_id"""
_SLOT_SQL = f"""
    SELECT {_SLOT_COLS}
    FROM slots s
    {_SLOT_JOINS}
"""
_SLOT_KEYS = ("id", "reel_id", "idx", "start_s", "end_s", "ref_description", "reason",
              "human_verdict", "human_note", "chosen_moment_id", "swapped_moment_id",
              "candidates", "params", "thumbnail_key", "description", "quality_score",
              "clip_key", "moment_start", "moment_end")


# --- pure helpers (no DB) ----------------------------------------------------

def effective_in_point(params: dict | None, candidates: list[dict] | None,
                       swapped_moment_id: int | None,
                       moment_start: float, moment_end: float) -> float:
    """The in_point for one slot, swap-aware (single source of truth: the review app's
    beat cards AND scripts/rerender_reel.py both use this, so previews match renders).

    Not swapped -> the original render params' in_point. Swapped -> the matching
    candidate's in_point (the swap options carry their own preview in_point); if no
    candidate matches (e.g. legacy rows), fall back to the swapped moment's midpoint.
    """
    if swapped_moment_id is None:
        return float((params or {}).get("in_point") or 0.0)
    for c in candidates or []:
        if c.get("moment_id") == swapped_moment_id and c.get("in_point") is not None:
            return float(c["in_point"])
    return (float(moment_start) + float(moment_end)) / 2.0


def _slot_dict(row: tuple) -> dict[str, Any]:
    """Map one joined slot row to the template dict, with ``effective_in`` computed."""
    d = dict(zip(_SLOT_KEYS, row))
    d["effective_in"] = effective_in_point(
        d["params"], d["candidates"], d["swapped_moment_id"],
        d["moment_start"] or 0.0, d["moment_end"] or 0.0)
    return d

def alternates_json(plans) -> list[dict]:
    """Serialize runner-up ``SlotPlan``s to the ``slots.candidates`` jsonb shape (pure)."""
    out: list[dict] = []
    for p in plans:
        m = p.moment
        out.append({
            "moment_id": m.moment_id,
            "clip_key": m.clip_key,
            "thumbnail_key": m.thumbnail_key,
            "score": p.score,
            "reason": p.reason,
            "in_point": p.in_point,
        })
    return out


def _media_base_url() -> str:
    """Public media base (Supabase Storage bucket) for deployed serving; env MEDIA_BASE_URL."""
    import os

    return os.getenv("MEDIA_BASE_URL", "").rstrip("/")


def thumb_url(thumbnail_key: str | None) -> str | None:
    """Map a moment's ``thumbnail_key`` to a servable URL (pure apart from one env read).

    Local dev serves from the ``/thumbs`` static mount; when MEDIA_BASE_URL is set
    (the public Supabase Storage bucket), thumbnails come from the cloud instead so
    the deployed app needs no local media. Keys are stored relative to the data dir
    (e.g. ``thumbs/<hash>/000.jpg``); tolerate a bare key or a ``data/`` prefix.
    """
    if not thumbnail_key:
        return None
    key = thumbnail_key
    for prefix in ("data/thumbs/", "thumbs/"):
        if key.startswith(prefix):
            key = key[len(prefix):]
            break
    base = _media_base_url()
    if base:
        return f"{base}/thumbs/{key}"
    return f"/thumbs/{key}"


def reel_video_url(output_key: str | None) -> str | None:
    """Map ``reels.output_key`` to a servable URL (pure apart from one env read).

    Uses the Supabase Storage bucket when MEDIA_BASE_URL is set (deployed serving),
    else the local ``/media/output`` mount (dev). Keys are normally repo-relative
    (``output/x.mp4``); tolerate absolute paths by taking everything after the last
    ``output/`` segment, and a bare filename as-is.
    """
    if not output_key:
        return None
    key = output_key
    if "output/" in key:
        key = key.rsplit("output/", 1)[-1]
    if not key:
        return None
    rel = quote(key)
    base = _media_base_url()
    if base:
        return f"{base}/output/{rel}"
    return "/media/output/" + rel


def proxy_url(storage_key: str | None) -> str | None:
    """Map a clip ``storage_key`` to its servable 480p proxy URL (one env read).

    Uses the Supabase Storage bucket when MEDIA_BASE_URL is set (deployed serving),
    else the local ``/media/proxies`` mount (dev). ``library/A/B.MP4`` -> ``.../proxies/A/B.mp4``:
    strip the leading ``library/``, swap the extension to ``.mp4`` (proxies are always mp4),
    keep subdirs, URL-escape spaces. Non-library or falsy keys return None.
    """
    if not storage_key or not storage_key.startswith("library/"):
        return None
    rel = storage_key[len("library/"):]
    stem, _ext = os.path.splitext(rel)
    if not stem:
        return None
    path = quote(stem + ".mp4")
    base = _media_base_url()
    if base:
        return f"{base}/proxies/{path}"
    return "/media/proxies/" + path


# --- module-level writer used by scripts/recreate.py -------------------------

def update_slot_candidates(slot_id: int, candidates: list[dict], conn=None) -> None:
    """Persist the runner-up plans for one slot (targeted UPDATE; insert_slot stays as-is)."""
    from psycopg.types.json import Json

    own = conn is None
    c = conn or get_conn()
    try:
        c.execute("UPDATE slots SET candidates = %s WHERE id = %s", (Json(candidates), slot_id))
        if own:
            c.commit()
    finally:
        if own:
            c.close()


# --- the injectable queries seam ---------------------------------------------

class Queries:
    """Live read/write helpers for the app routes, over ONE cached connection.

    The DB is remote (Supabase over TLS), so a fresh connection per call costs ~1s of
    handshake - that was the whole tap latency. The app is single-user and local, so a
    simple lazily-created cached autocommit connection beats a pool: no pool tuning, no
    idle churn, and a one-shot reconnect+retry covers the dropped-conn case. uvicorn may
    run sync handlers in a threadpool, so a lock serializes execute+fetch. The
    constructor stays lazy (no connection at import/create_app time; tests inject fakes).
    """

    def __init__(self) -> None:
        self._conn: Any = None  # psycopg.Connection once connected (lazy import)
        self._lock = threading.Lock()

    def _connect(self):
        conn = get_conn()
        conn.autocommit = True  # every _rows call is a single statement; no commit trip
        return conn

    def _rows(self, sql: str, params: tuple = ()) -> list[tuple]:
        import psycopg  # lazy: keep module import light (tests fake this seam)

        with self._lock:
            if self._conn is None or getattr(self._conn, "closed", False):
                self._conn = self._connect()
            try:
                return self._conn.execute(sql, params).fetchall()
            except (psycopg.OperationalError, psycopg.InterfaceError):
                try:  # the cached conn died (idle timeout, network blip): reconnect ONCE
                    self._conn.close()
                except Exception:
                    pass
                self._conn = self._connect()
                return self._conn.execute(sql, params).fetchall()

    def ping(self) -> bool:
        """Quick ``select 1`` - /healthz doubles as the Supabase keep-alive."""
        try:
            return self._rows("SELECT 1")[0][0] == 1
        except Exception:
            return False

    # -- reels / review -------------------------------------------------------

    # 200 ~= "all of them" at current volume; the dashboard felt truncated at 20.
    def list_reels(self, limit: int = 200) -> list[dict[str, Any]]:
        rows = self._rows("""
            SELECT r.id, r.reference_key, r.created_at, r.approved, r.grade,
                   count(s.id) AS slot_count,
                   count(*) FILTER (WHERE s.human_verdict IS NOT NULL) AS reviewed_count,
                   t.thumbnail_key,
                   r.status = 'published' AS published, r.posted_at, r.is_trial,
                   EXISTS (SELECT 1 FROM scheduled_posts sp WHERE sp.reel_id = r.id
                           AND sp.status IN ('scheduled', 'publishing')) AS pending_post
            FROM reels r LEFT JOIN slots s ON s.reel_id = r.id
            LEFT JOIN LATERAL (
                SELECT m.thumbnail_key FROM slots s2
                JOIN moments m ON m.id = COALESCE(s2.swapped_moment_id, s2.chosen_moment_id)
                WHERE s2.reel_id = r.id AND m.thumbnail_key IS NOT NULL
                ORDER BY s2.idx LIMIT 1
            ) t ON true
            GROUP BY r.id, t.thumbnail_key ORDER BY r.id DESC LIMIT %s
        """, (limit,))
        return [{"id": r[0], "ref": r[1], "created": r[2].strftime("%b %d") if r[2] else "",
                 "approved": bool(r[3]), "grade": r[4], "slot_count": r[5],
                 "reviewed_count": r[6], "thumbnail_key": r[7],
                 "published": bool(r[8]),
                 "posted": r[9].strftime("%b %d") if r[9] else "",
                 "is_trial": bool(r[10]), "pending_post": bool(r[11])}
                for r in rows]

    def reel_detail(self, reel_id: int) -> dict[str, Any] | None:
        reel_rows = self._rows(
            "SELECT id, reference_key, created_at, approved, output_key, cut_dirty, grade,"
            " caption, status = 'published' AS published, posted_at, is_trial"
            " FROM reels WHERE id = %s",
            (reel_id,))
        if not reel_rows:
            return None
        r = reel_rows[0]
        reel = {"id": r[0], "ref": r[1], "created": r[2].strftime("%b %d") if r[2] else "",
                "approved": bool(r[3]), "output_key": r[4], "cut_dirty": bool(r[5]),
                "grade": r[6], "caption": r[7], "published": bool(r[8]),
                "posted": r[9].strftime("%b %d") if r[9] else "",
                "is_trial": bool(r[10]),
                # watch-first review: the player renders when the mp4 is reachable -
                # from Supabase Storage when deployed (MEDIA_BASE_URL set), else local disk
                "playable": bool(r[4]) and (bool(_media_base_url()) or os.path.exists(r[4]))}
        slot_rows = self._rows(_SLOT_SQL + " WHERE s.reel_id = %s ORDER BY s.idx", (reel_id,))
        return {"reel": reel, "slots": [_slot_dict(row) for row in slot_rows]}

    def _slot(self, slot_id: int) -> dict[str, Any] | None:
        rows = self._rows(_SLOT_SQL + " WHERE s.id = %s", (slot_id,))
        return _slot_dict(rows[0]) if rows else None

    def _update_slot(self, set_sql: str, set_params: tuple, slot_id: int,
                     dirty: str | None = None) -> dict[str, Any] | None:
        """UPDATE a slot and return the refreshed joined row in ONE round trip.

        The CTE updates, then the outer SELECT re-joins moments/clips on the updated
        row - same columns as ``_SLOT_SQL`` (factored ``_SLOT_COLS``/``_SLOT_JOINS``).
        The same statement also carries the reel bookkeeping (still one round trip):

        - ``dirty`` marks the reel's rendered cut stale (``reels.cut_dirty``) when the
          write changes what a re-render would produce: ``"always"`` (swap) or
          ``"if_was_swapped"`` (undo - only undoing a swap moves the cut). ``before``
          reads the pre-update verdict (all CTEs share the statement snapshot).
        - decided/total for the reel ride along for the live progress header. The
          snapshot predates ``upd``, so the updated slot is counted from ``upd`` itself
          and the siblings from the table.

        The returned dict is the slot row plus ``total``/``decided``/``cut_dirty``.
        """
        if dirty == "always":
            dirty_where = "WHERE id = (SELECT reel_id FROM before)"
            dirty_col = "true"
        elif dirty == "if_was_swapped":
            dirty_where = "WHERE id = (SELECT reel_id FROM before WHERE old_verdict = 'swapped')"
            dirty_col = ("(SELECT old_verdict FROM before) IS NOT DISTINCT FROM 'swapped' "
                         "OR COALESCE((SELECT cut_dirty FROM reels WHERE id = s.reel_id), false)")
        else:
            dirty_where = None
            dirty_col = "COALESCE((SELECT cut_dirty FROM reels WHERE id = s.reel_id), false)"
        dirty_cte = (f", dirty AS (UPDATE reels SET cut_dirty = true {dirty_where} RETURNING id)"
                     if dirty_where else "")
        rows = self._rows(f"""
            WITH before AS (SELECT reel_id, human_verdict AS old_verdict
                            FROM slots WHERE id = %s),
                 upd AS (UPDATE slots SET {set_sql} WHERE id = %s RETURNING *){dirty_cte},
                 agg AS (SELECT count(*) AS others,
                                count(*) FILTER (WHERE s2.human_verdict IS NOT NULL) AS others_decided
                         FROM slots s2 JOIN upd u ON s2.reel_id = u.reel_id AND s2.id <> u.id)
            SELECT {_SLOT_COLS},
                   agg.others + 1,
                   agg.others_decided + (s.human_verdict IS NOT NULL)::int,
                   {dirty_col}
            FROM upd s {_SLOT_JOINS} CROSS JOIN agg
        """, (slot_id, *set_params, slot_id))
        if not rows:
            return None
        d = _slot_dict(rows[0][:len(_SLOT_KEYS)])
        d["total"], d["decided"], d["cut_dirty"] = rows[0][len(_SLOT_KEYS):len(_SLOT_KEYS) + 3]
        return d

    def keep_slot(self, slot_id: int, note: str = "") -> dict[str, Any] | None:
        """Mark a beat kept (+ optional note); return the refreshed slot row."""
        return self._update_slot(
            "human_verdict = 'keep', human_note = NULLIF(%s, '')", (note,), slot_id)

    def weak_keep_slot(self, slot_id: int, note: str = "") -> dict[str, Any] | None:
        """Keep the beat but mark it a weak positive ("could be better"); return the row.

        Still in the reel, so it does NOT dirty the cut - but a distinct verdict so the
        judge learns this pick was passable, not a favorite (finer than the reel grade).
        """
        return self._update_slot(
            "human_verdict = 'weak', human_note = NULLIF(%s, '')", (note,), slot_id)

    def swap_slot(self, slot_id: int, moment_id: int, note: str = "") -> dict[str, Any] | None:
        """Swap a beat to a runner-up moment; return the refreshed slot row.

        A swap changes what a re-render would cut, so the reel goes cut-dirty.
        """
        return self._update_slot(
            "swapped_moment_id = %s, human_verdict = 'swapped', human_note = NULLIF(%s, '')",
            (moment_id, note), slot_id, dirty="always")

    def flag_slot(self, slot_id: int, reason: str = "") -> dict[str, Any] | None:
        """Flag a beat as wrong (reason [+ note] becomes the note); return the slot row.

        Routes compose ``reason`` with any typed note ("reason: note") before calling.
        """
        return self._update_slot(
            "human_verdict = 'flagged', human_note = NULLIF(%s, '')", (reason,), slot_id)

    def undo_slot(self, slot_id: int) -> dict[str, Any] | None:
        """Clear a verdict (accidental tap): back to undecided; return the refreshed row.

        The note survives - it is user data, not part of the verdict. Undoing a SWAP
        moves the effective pick back, so the reel goes cut-dirty (undoing keep/flag
        does not change the cut). The taste log gets a ``human_undo`` event alongside
        (route's job) so downstream consumers can discount the earlier verdict.
        """
        return self._update_slot(
            "human_verdict = NULL, swapped_moment_id = NULL", (), slot_id,
            dirty="if_was_swapped")

    def slot_detail(self, slot_id: int) -> dict[str, Any] | None:
        """One slot row for the repick routes (public read; same shape as ``_slot``)."""
        return self._slot(slot_id)

    def decided_counts(self, reel_id: int) -> dict[str, int]:
        """Decided/total beats for the live progress header (non-tap paths)."""
        row = self._rows("""
            SELECT count(*) FILTER (WHERE human_verdict IS NOT NULL), count(*)
            FROM slots WHERE reel_id = %s
        """, (reel_id,))[0]
        return {"decided": row[0], "total": row[1]}

    def reel_cut_dirty(self, reel_id: int) -> bool:
        """Does the reel's rendered mp4 lag behind the reviewer's current picks?"""
        rows = self._rows("SELECT cut_dirty FROM reels WHERE id = %s", (reel_id,))
        return bool(rows[0][0]) if rows else False

    def latest_rerender(self, reel_id: int) -> dict[str, Any] | None:
        """The newest ``rerender:<id>`` run_requests row for this reel (or None).

        Carries the reel's CURRENT output_key too: when the worker finishes it points
        ``reels.output_key`` at the new cut, so the status banner can offer to load it.
        """
        rows = self._rows("""
            SELECT status, error, (SELECT output_key FROM reels WHERE id = %s)
            FROM run_requests WHERE reference_key = %s
            ORDER BY id DESC LIMIT 1
        """, (reel_id, f"rerender:{reel_id}"))
        if not rows:
            return None
        return {"status": rows[0][0], "error": rows[0][1], "output_key": rows[0][2]}

    def latest_repick(self, slot_id: int) -> dict[str, Any] | None:
        """The newest ``repick:<slot_id>`` run_requests row for this slot (or None)."""
        rows = self._rows("""
            SELECT status, error FROM run_requests WHERE reference_key = %s
            ORDER BY id DESC LIMIT 1
        """, (f"repick:{slot_id}",))
        if not rows:
            return None
        return {"status": rows[0][0], "error": rows[0][1]}

    def grade_reel(self, reel_id: int, grade: str) -> dict[str, Any]:
        """Grade a reel in one tap; return the data the review_summary event needs.

        A positive grade (post_worthy | would_test) endorses every beat the reviewer
        did NOT flag - so ``accepted`` carries each such beat with the exact moment it
        played, and the judge learns from all of them without a per-beat tap. ``not_it``
        rejects the concept: nothing is endorsed. ``approved`` stays the derived boolean.
        """
        if grade not in GRADES:
            raise ValueError(f"unknown grade: {grade}")
        approved = grade in POSITIVE_GRADES
        self._rows(
            "UPDATE reels SET grade = %s, approved = %s, graded_at = now(),"
            " approved_at = CASE WHEN %s THEN now() ELSE approved_at END"
            " WHERE id = %s RETURNING id",
            (grade, approved, approved, reel_id))
        rows = self._rows(
            "SELECT idx, human_verdict, COALESCE(swapped_moment_id, chosen_moment_id)"
            " FROM slots WHERE reel_id = %s ORDER BY idx", (reel_id,))
        # A positive grade endorses every non-flagged beat at its played moment; the
        # per-beat verdict rides along so a 'weak' keep stays a weak positive even here.
        accepted = ([{"beat": idx, "moment_id": mid, "verdict": verdict}
                     for idx, verdict, mid in rows if verdict != "flagged"]
                    if approved else [])
        return {
            "id": reel_id, "grade": grade, "approved": approved, "slots": len(rows),
            "flagged": sum(1 for _, v, _ in rows if v == "flagged"),
            "swapped": sum(1 for _, v, _ in rows if v == "swapped"),
            "kept": sum(1 for _, v, _ in rows if v == "keep"),
            "weak": sum(1 for _, v, _ in rows if v == "weak"),
            "accepted": accepted,
        }

    # -- curation --------------------------------------------------------------

    def curation_batch(self, limit: int = 20) -> list[dict[str, Any]]:
        """Unrated moments to label: mostly the borderline quality band + a few random others."""
        lo, hi = QUALITY_BAND
        n_rand = max(0, limit // 4) if limit > 1 else 0
        n_band = limit - n_rand
        base = "SELECT id, thumbnail_key, description, quality_score FROM moments " \
               "WHERE human_label IS NULL AND superseded = false"
        rows = self._rows(
            f"({base} AND quality_score BETWEEN %s AND %s ORDER BY random() LIMIT %s)", (lo, hi, n_band))
        if n_rand:
            rows += self._rows(
                f"({base} AND (quality_score IS NULL OR quality_score < %s OR quality_score > %s) "
                f"ORDER BY random() LIMIT %s)", (lo, hi, n_rand))
        batch = [{"id": r[0], "thumbnail_key": r[1], "description": r[2], "quality_score": r[3]}
                 for r in rows]
        random.shuffle(batch)
        return batch[:limit]

    def rate_moment(self, moment_id: int, label: int) -> None:
        """Record a human thumbs label (1 = up, 0 = down)."""
        self._rows("UPDATE moments SET human_label = %s WHERE id = %s RETURNING id",
                   (label, moment_id))

    # -- serverless taste + run queue -------------------------------------------

    def insert_taste_event(self, entry: dict) -> None:
        """Durable app-layer taste write (the worker later syncs it to the local file)."""
        from psycopg.types.json import Json

        self._rows("INSERT INTO taste_events (entry) VALUES (%s) RETURNING id", (Json(entry),))

    def create_run_request(self, reference_key: str) -> int:
        """Queue a pipeline run for the local worker; return the request id."""
        rows = self._rows(
            "INSERT INTO run_requests (reference_key) VALUES (%s) RETURNING id",
            (reference_key,))
        return rows[0][0]

    def list_run_requests(self, limit: int = 10) -> list[dict[str, Any]]:
        """Most recent run requests for the dashboard queue card, newest first."""
        def fmt(t) -> str:
            return t.strftime("%b %d %H:%M") if t else ""

        rows = self._rows("""
            SELECT id, reference_key, status, created_at, started_at, finished_at, reel_id, error
            FROM run_requests ORDER BY id DESC LIMIT %s
        """, (limit,))
        return [{"id": r[0], "reference_key": r[1], "status": r[2], "created": fmt(r[3]),
                 "started": fmt(r[4]), "finished": fmt(r[5]), "reel_id": r[6], "error": r[7]}
                for r in rows]

    def available_references(self) -> list[str]:
        """The curated run gallery: references already run, unioned with the known repo mp4s."""
        rows = self._rows("SELECT DISTINCT reference_key FROM reels WHERE reference_key IS NOT NULL")
        return sorted({r[0] for r in rows} | set(KNOWN_REFERENCES))

    # -- trial-reel publish scheduling ----------------------------------------

    _SCHED_KEYS = ("id", "status", "scheduled_at", "as_trial", "provider", "error",
                   "published_at")

    def latest_scheduled_post(self, reel_id: int) -> dict[str, Any] | None:
        """The newest scheduled_posts row for this reel (or None) - the schedule UI state."""
        rows = self._rows("""
            SELECT id, status, scheduled_at, as_trial, provider, error, published_at
            FROM scheduled_posts WHERE reel_id = %s ORDER BY id DESC LIMIT 1
        """, (reel_id,))
        return dict(zip(self._SCHED_KEYS, rows[0])) if rows else None

    def create_scheduled_post(self, reel_id: int, scheduled_at,
                              as_trial: bool = True, caption: str = "") -> int | None:
        """Queue a publish for the local publish worker. A non-empty ``caption``
        (typed in the schedule form) wins; else the reel's stored caption is
        snapshotted. Approved reels only (the SELECT guard); None if not."""
        rows = self._rows("""
            INSERT INTO scheduled_posts (reel_id, scheduled_at, as_trial, caption)
            SELECT id, %s, %s, COALESCE(NULLIF(%s, ''), caption)
            FROM reels WHERE id = %s AND approved
            RETURNING id
        """, (scheduled_at, as_trial, caption, reel_id))
        return rows[0][0] if rows else None

    def posted_reels(self) -> list[dict[str, Any]]:
        """Reels for the /posted page: anything published, plus anything with an
        active (scheduled/publishing) post. Newest schedule state per reel wins."""
        rows = self._rows("""
            SELECT r.id, r.reference_key, r.status = 'published' AS published,
                   r.posted_at, r.is_trial, sp.status, sp.scheduled_at, sp.as_trial,
                   t.thumbnail_key
            FROM reels r
            LEFT JOIN LATERAL (
                SELECT status, scheduled_at, as_trial FROM scheduled_posts
                WHERE reel_id = r.id ORDER BY id DESC LIMIT 1
            ) sp ON true
            LEFT JOIN LATERAL (
                SELECT m.thumbnail_key FROM slots s2
                JOIN moments m ON m.id = COALESCE(s2.swapped_moment_id, s2.chosen_moment_id)
                WHERE s2.reel_id = r.id AND m.thumbnail_key IS NOT NULL
                ORDER BY s2.idx LIMIT 1
            ) t ON true
            WHERE r.status = 'published' OR sp.status IN ('scheduled', 'publishing')
            ORDER BY (r.status = 'published'), sp.scheduled_at, r.posted_at DESC
        """)
        return [{"id": r[0], "ref": r[1], "published": bool(r[2]),
                 "posted": r[3].strftime("%b %d") if r[3] else "",
                 "is_trial": bool(r[4]), "sched_status": r[5], "scheduled_at": r[6],
                 "sched_as_trial": bool(r[7]) if r[7] is not None else True,
                 "thumbnail_key": r[8]}
                for r in rows]

    def cancel_scheduled_post(self, post_id: int) -> bool:
        """Cancel a still-pending schedule (a claimed/terminal row stays as it is)."""
        rows = self._rows("""
            UPDATE scheduled_posts SET status = 'canceled'
            WHERE id = %s AND status = 'scheduled' RETURNING id
        """, (post_id,))
        return bool(rows)

    # -- dashboard -------------------------------------------------------------

    def counts(self) -> dict[str, int]:
        reels, approved = self._rows(
            "SELECT count(*), count(*) FILTER (WHERE approved) FROM reels")[0]
        unrated, rated = self._rows("""
            SELECT count(*) FILTER (WHERE human_label IS NULL),
                   count(*) FILTER (WHERE human_label IS NOT NULL)
            FROM moments WHERE superseded = false
        """)[0]
        return {"reels": reels, "approved": approved,
                "unrated_moments": unrated, "rated_moments": rated}
