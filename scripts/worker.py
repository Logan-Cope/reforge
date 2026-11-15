#!/usr/bin/env python3
"""Local run-queue worker (NOT deployed): executes what the serverless app can't.

Each cycle it (1) syncs unsynced ``taste_events`` rows into the local
``taste/taste-log.jsonl`` (the judge's file), then (2) claims the oldest 'queued'
``run_requests`` row (FOR UPDATE SKIP LOCKED - safe if two workers ever overlap) and
runs ``scripts/recreate.py`` as a SUBPROCESS, so this process never imports torch.

Usage:
    PYTHONPATH=src python scripts/worker.py            # loop forever (every 30s)
    PYTHONPATH=src python scripts/worker.py --once     # one cycle (cron / tests)
    PYTHONPATH=src python scripts/worker.py --interval 60
"""
import argparse
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))  # allow running without PYTHONPATH=src


# --- pure helpers (unit-tested; no I/O) ---------------------------------------

def output_path_for(reference_key: str, run_id: int) -> str:
    """Derive a NEW output path per run (never overwrite old renders - house rule)."""
    stem = os.path.splitext(os.path.basename(reference_key))[0]
    return f"output/{stem}_run{run_id}.mp4"


def recreate_cmd(reference_key: str, out: str) -> list[str]:
    """The recreate.py subprocess argv (same interpreter, repo-relative script)."""
    return [sys.executable, os.path.join(REPO, "scripts", "recreate.py"),
            "--reference", reference_key, "--out", out]


def _parse_prefixed_id(key: str | None, prefix: str) -> int | None:
    """The integer id from a ``<prefix><id>`` queue key; None for anything else."""
    if not key or not key.startswith(prefix):
        return None
    rest = key[len(prefix):]
    return int(rest) if rest.isdigit() else None


def parse_rerender_key(key: str | None) -> int | None:
    """Reel id from a ``rerender:<reel_id>`` queue key; None for normal reference keys."""
    return _parse_prefixed_id(key, "rerender:")


def parse_repick_key(key: str | None) -> int | None:
    """Slot id from a ``repick:<slot_id>`` queue key; None for anything else."""
    return _parse_prefixed_id(key, "repick:")


def rerender_cmd(reel_id: int) -> list[str]:
    """The rerender_reel.py subprocess argv (same interpreter, repo-relative script)."""
    return [sys.executable, os.path.join(REPO, "scripts", "rerender_reel.py"),
            "--reel-id", str(reel_id)]


def repick_cmd(slot_id: int) -> list[str]:
    """The repick_slot.py subprocess argv (same interpreter, repo-relative script)."""
    return [sys.executable, os.path.join(REPO, "scripts", "repick_slot.py"),
            "--slot-id", str(slot_id)]


def error_tail(stderr: str | None, limit: int = 2000) -> str:
    """Last ``limit`` chars of stderr for ``run_requests.error`` (stripped; None-safe)."""
    return (stderr or "").strip()[-limit:]


def _upload_media_to_cloud() -> None:
    """Push any new renders + proxies to Supabase Storage so the deployed app can serve
    them. Idempotent via upload_media's manifest (only new files upload). A failure here
    is logged but never fails the render job - the render is done; the sync can be retried.
    """
    try:
        cmd = [sys.executable, os.path.join(REPO, "scripts", "upload_media.py"),
               "--outputs", "--proxies"]
        env = {**os.environ, "PYTHONPATH": os.path.join(REPO, "src")}
        r = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True, timeout=600)
        if r.returncode == 0:
            print("[worker] media synced to Supabase")
        else:
            print(f"[worker] WARN media sync rc={r.returncode}: {error_tail(r.stderr, 300)}")
    except Exception as e:  # never let a sync problem fail the (already-complete) render
        print(f"[worker] WARN media sync failed: {e}")


# --- DB steps (thin I/O; kept simple, untested by design) ----------------------

def _get_conn():
    from reforge.db.client import get_conn
    return get_conn()


def sync_taste_events(conn) -> int:
    """Append each unsynced taste_events row to taste/taste-log.jsonl; mark synced."""
    from reforge.taste import append_taste_log

    rows = conn.execute(
        "SELECT id, entry FROM taste_events WHERE synced = false ORDER BY id").fetchall()
    for event_id, entry in rows:
        append_taste_log(entry)
        conn.execute("UPDATE taste_events SET synced = true WHERE id = %s", (event_id,))
    conn.commit()
    return len(rows)


def backfill_reel_captions(conn) -> None:
    """Fill missing ``reels.caption`` (publish/captions.py, one cheap Haiku call each)
    so the schedule form always has something to pre-fill. A caption problem must
    never fail the render loop - same contract as the media sync."""
    try:
        from reforge.publish.captions import backfill_captions

        n = backfill_captions(conn)
        if n:
            print(f"[worker] generated {n} reel caption(s)")
    except Exception as e:
        print(f"[worker] WARN caption backfill failed: {e}")


def claim_next(conn):
    """Atomically claim the oldest queued run_request; return (id, reference_key) or None."""
    row = conn.execute("""
        UPDATE run_requests SET status = 'running', started_at = now()
        WHERE id = (SELECT id FROM run_requests WHERE status = 'queued'
                    ORDER BY id LIMIT 1 FOR UPDATE SKIP LOCKED)
        RETURNING id, reference_key
    """).fetchone()
    conn.commit()
    return row


def finish(conn, run_id: int, *, ok: bool, error: str = "", reel_id: int | None = None) -> None:
    """Record the outcome. On success, link ``reel_id`` if given (a re-render targets an
    existing reel), else the newest reels row (recreate.py just wrote it)."""
    if ok:
        if reel_id is None:
            reel_id = conn.execute("SELECT max(id) FROM reels").fetchone()[0]
        conn.execute("""
            UPDATE run_requests SET status = 'done', finished_at = now(), reel_id = %s
            WHERE id = %s
        """, (reel_id, run_id))
    else:
        conn.execute("""
            UPDATE run_requests SET status = 'error', finished_at = now(), error = %s
            WHERE id = %s
        """, (error, run_id))
    conn.commit()


def execute_run(conn, run_id: int, reference_key: str) -> None:
    """Run the pipeline for one claimed request as a subprocess and record the outcome.

    A ``rerender:<reel_id>`` key re-renders that existing reel with its swaps applied
    (scripts/rerender_reel.py); a ``repick:<slot_id>`` key re-runs retrieval for one
    flagged beat (scripts/repick_slot.py); anything else is a fresh recreate.py run.
    """
    rerender_id = parse_rerender_key(reference_key)
    repick_id = parse_repick_key(reference_key)
    if rerender_id is not None:
        cmd = rerender_cmd(rerender_id)
        print(f"[worker] run {run_id}: rerender reel {rerender_id} (swaps applied)")
    elif repick_id is not None:
        cmd = repick_cmd(repick_id)
        print(f"[worker] run {run_id}: repick slot {repick_id} (flag as critique)")
    else:
        out = output_path_for(reference_key, run_id)
        cmd = recreate_cmd(reference_key, out)
        print(f"[worker] run {run_id}: recreate {reference_key} -> {out}")
    env = {**os.environ, "PYTHONPATH": os.path.join(REPO, "src")}
    proc = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True)
    if proc.returncode == 0:
        # push the new render to Supabase BEFORE marking done, so the deployed app never
        # reports "done" for a cut whose mp4 isn't in cloud storage yet.
        _upload_media_to_cloud()
        reel_id = rerender_id
        if rerender_id is not None:
            # the fresh cut embodies the current picks - cut freshness restored
            conn.execute("UPDATE reels SET cut_dirty = false WHERE id = %s", (rerender_id,))
        elif repick_id is not None:
            row = conn.execute(
                "SELECT reel_id FROM slots WHERE id = %s", (repick_id,)).fetchone()
            reel_id = row[0] if row else None
        finish(conn, run_id, ok=True, reel_id=reel_id)
        print(f"[worker] run {run_id}: done")
    else:
        finish(conn, run_id, ok=False, error=error_tail(proc.stderr))
        print(f"[worker] run {run_id}: ERROR (exit {proc.returncode})")


def cycle() -> None:
    """One worker pass: sync taste events, then drain the queue one run at a time."""
    conn = _get_conn()
    try:
        synced = sync_taste_events(conn)
        print(f"[worker] synced {synced} taste event(s)")
        backfill_reel_captions(conn)
        while True:
            claimed = claim_next(conn)
            if claimed is None:
                print("[worker] queue empty")
                break
            execute_run(conn, claimed[0], claimed[1])
    finally:
        conn.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Local worker for queued reforge runs.")
    ap.add_argument("--once", action="store_true", help="one cycle then exit (cron/tests)")
    ap.add_argument("--interval", type=int, default=30, help="seconds between cycles")
    args = ap.parse_args()

    while True:
        cycle()
        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
