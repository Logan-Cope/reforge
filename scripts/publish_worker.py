#!/usr/bin/env python3
"""Local publish worker (NOT deployed): ships due scheduled_posts rows.

Mirrors scripts/worker.py: each cycle claims the oldest due 'scheduled' row
(FOR UPDATE SKIP LOCKED - safe if two workers ever overlap) and hands it to the
configured publish provider. DRY-RUN BY DEFAULT: until PUBLISH_PROVIDER names a
real provider AND PUBLISH_LIVE=1, the dry-run publisher records what WOULD ship
(status 'dry_run_done') and the reel row is never touched. Only a real successful
publish moves reels.status to 'published' (+ ig_media_id / is_trial / posted_at).

Usage:
    PYTHONPATH=src python scripts/publish_worker.py             # loop (every 60s)
    PYTHONPATH=src python scripts/publish_worker.py --once      # one cycle (cron/tests)
    PYTHONPATH=src python scripts/publish_worker.py --interval 300
"""
import argparse
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))  # allow running without PYTHONPATH=src


# --- DB steps (thin I/O; kept simple, untested by design - logic lives in
# reforge.publish.scheduler / providers, which ARE unit-tested) -----------------

def _get_conn():
    from reforge.db.client import get_conn
    return get_conn()


def claim_due(conn):
    """Atomically claim the oldest due scheduled post; return the row or None."""
    row = conn.execute("""
        UPDATE scheduled_posts SET status = 'publishing', started_at = now()
        WHERE id = (SELECT id FROM scheduled_posts
                    WHERE status = 'scheduled' AND scheduled_at <= now()
                    ORDER BY scheduled_at LIMIT 1 FOR UPDATE SKIP LOCKED)
        RETURNING id, reel_id, as_trial, caption
    """).fetchone()
    conn.commit()
    return row


def execute_publish(conn, post_id: int, reel_id: int, as_trial: bool,
                    caption: str | None) -> None:
    """Publish one claimed post via the configured provider; record the outcome."""
    from psycopg.types.json import Json

    from reforge.publish.providers import get_publisher
    from reforge.publish.scheduler import public_video_url, reel_update_for

    row = conn.execute("SELECT output_key FROM reels WHERE id = %s", (reel_id,)).fetchone()
    video_url = public_video_url(row[0] if row else None, os.getenv("MEDIA_BASE_URL", ""))
    publisher = get_publisher()
    print(f"[publish] post {post_id}: reel {reel_id} via {publisher.name} "
          f"(trial={as_trial}) url={video_url}")
    res = publisher.publish(video_url=video_url, caption=caption, as_trial=as_trial)
    if res.ok:
        status = "dry_run_done" if res.dry_run else "published"
        conn.execute("""
            UPDATE scheduled_posts SET status = %s, provider = %s, result = %s,
                   published_at = now() WHERE id = %s
        """, (status, publisher.name, Json(res.payload), post_id))
        upd = reel_update_for(res, as_trial=as_trial)
        if upd:  # ONLY a real successful publish reaches this
            conn.execute("""
                UPDATE reels SET status = %s, ig_media_id = %s, is_trial = %s,
                       posted_at = now() WHERE id = %s
            """, (upd["status"], upd["ig_media_id"], upd["is_trial"], reel_id))
        print(f"[publish] post {post_id}: {status}")
    else:
        conn.execute("""
            UPDATE scheduled_posts SET status = 'error', provider = %s, result = %s,
                   error = %s WHERE id = %s
        """, (publisher.name, Json(res.payload), (res.error or "")[:2000], post_id))
        print(f"[publish] post {post_id}: ERROR {res.error}")
    conn.commit()


def cycle() -> None:
    """One pass: drain every due post, one at a time."""
    conn = _get_conn()
    try:
        while True:
            row = claim_due(conn)
            if row is None:
                print("[publish] nothing due")
                break
            execute_publish(conn, row[0], row[1], bool(row[2]), row[3])
    finally:
        conn.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Local worker for scheduled trial-reel posts.")
    ap.add_argument("--once", action="store_true", help="one cycle then exit (cron/tests)")
    ap.add_argument("--interval", type=int, default=60, help="seconds between cycles")
    args = ap.parse_args()

    while True:
        cycle()
        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
