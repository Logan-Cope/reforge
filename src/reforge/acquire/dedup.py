"""Reforge's own reference-acquisition dedup store + the caption-heavy filter.

Two pure helpers (unit-tested) decide whether a source reel is recreatable B-roll:
Reforge can only recreate low/no-caption reels (it swaps in Logan's OWN footage), so a
caption-heavy reference (dense on-screen text or a wordy overlay) is skipped. A
DISCIPLINE-style single-word overlay passes.

The DB helpers are thin wrappers over `db.client.get_conn` against the `source_reels`
table (migration 0010). This is Reforge's OWN ledger - it does NOT touch the upstream
discovery tool that supplied the URLs.
"""
from __future__ import annotations

import re

from ..db import client as db


# --- pure caption-filter logic (unit-tested, no I/O) ------------------------

def caption_density(shots) -> float:
    """Fraction of shots whose ``on_screen_text`` is non-empty (0.0-1.0).

    ``shots`` is a list of objects with an ``.on_screen_text`` attribute. Empty list -> 0.0.
    """
    if not shots:
        return 0.0
    with_text = sum(1 for s in shots if (s.on_screen_text or "").strip())
    return with_text / len(shots)


def caption_variety(shots, dominant: str) -> float:
    """Fraction of text-bearing shots whose text differs from the dominant overlay (pure).

    Zero when every captioned shot shows the same persistent phrase (DISCIPLINE-style
    single overlay), high when captions change per beat (listicle / talking-head). This
    is the real "can Reforge recreate this?" signal - a persistent short overlay is fine
    (Reforge burns one dominant overlay); varied per-beat captions are not. Empty -> 0.0.
    """
    def _norm(t: str) -> str:
        # strip case + punctuation so OCR noise ("I NEVER LOSE." vs "i never lose")
        # doesn't register as a different caption
        return re.sub(r"[^a-z0-9 ]", "", " ".join((t or "").split()).lower()).strip()

    dom = _norm(dominant)
    texted = [s for s in shots if _norm(getattr(s, "on_screen_text", ""))]
    if not texted:
        return 0.0
    differing = sum(1 for s in texted if _norm(s.on_screen_text) != dom)
    return differing / len(texted)


def is_caption_heavy(variety: float, overlay_word_count: int, *,
                     variety_thresh: float = 0.35, word_thresh: int = 8) -> bool:
    """True if too caption-heavy to recreate as B-roll.

    Trips on VARIED per-beat captions (``variety`` > ``variety_thresh``: listicle /
    talking-head) OR a long dominant overlay (> ``word_thresh`` words: a full sentence).
    A single persistent short overlay ("DISCIPLINE", "I NEVER LOSE") has variety 0 and
    passes - those are exactly the reels Reforge recreates well.
    """
    return variety > variety_thresh or overlay_word_count > word_thresh


# --- DB helpers (thin, over get_conn) ---------------------------------------

def already_seen(shortcode: str) -> bool:
    """True if this shortcode is already in the source_reels ledger (dedup)."""
    with db.get_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT 1 FROM source_reels WHERE shortcode = %s LIMIT 1", (shortcode,))
        return cur.fetchone() is not None


def record_pulled(shortcode: str, url: str, title: str | None = None) -> None:
    """Insert (or refresh) a source reel as 'pulled'. Idempotent on shortcode."""
    sql = """
        INSERT INTO source_reels (shortcode, url, title, status)
        VALUES (%s, %s, %s, 'pulled')
        ON CONFLICT (shortcode) DO UPDATE SET
            url = EXCLUDED.url,
            title = COALESCE(EXCLUDED.title, source_reels.title),
            status = 'pulled',
            skipped_reason = NULL,
            pulled_at = now()
    """
    with db.get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, (shortcode, url, title))
        conn.commit()


def mark_recreated(shortcode: str, reel_id: int, caption_density: float) -> None:
    """Mark a source reel recreated, linking the recreation's reels.id + its density."""
    sql = """
        UPDATE source_reels
        SET status = 'recreated', recreated_reel_id = %s, caption_density = %s,
            skipped_reason = NULL, recreated_at = now()
        WHERE shortcode = %s
    """
    with db.get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, (reel_id, caption_density, shortcode))
        conn.commit()


def mark_skipped(shortcode: str, reason: str, status: str) -> None:
    """Mark a source reel skipped/errored with a reason (status: skipped_* | error)."""
    sql = "UPDATE source_reels SET status = %s, skipped_reason = %s WHERE shortcode = %s"
    with db.get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, (status, reason, shortcode))
        conn.commit()
