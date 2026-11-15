"""Logan's taste memory: the rubric (read) + the taste log (append). LLM/DB-free on purpose."""
from __future__ import annotations

import json
import os
from datetime import date

from .config import settings


def load_rubric(path: str | None = None) -> str:
    """Read the taste rubric markdown; return "" if the file is missing."""
    path = path or os.path.join(settings.taste_dir, "rubric.md")
    try:
        with open(path, encoding="utf-8") as fh:
            return fh.read().rstrip()
    except OSError:
        return ""


def append_taste_log(entry: dict, path: str | None = None) -> None:
    """Append one JSON line to the taste log (created if missing); stamp ``ts`` if absent."""
    path = path or os.path.join(settings.taste_dir, "taste-log.jsonl")
    entry = dict(entry)
    entry.setdefault("ts", date.today().isoformat())
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")


def record_taste_event(entry: dict, queries=None) -> None:
    """App-layer taste write: the ``taste_events`` DB table + a best-effort local mirror.

    The deployed app runs on an ephemeral filesystem, so the durable copy is the DB row
    (the local worker later syncs it into ``taste/taste-log.jsonl``). When the taste dir
    IS writable (local dev), also append to the file directly; an OSError (read-only
    serverless fs) is skipped silently. ``queries`` is injectable for tests; it defaults
    to the live app seam via a lazy import so this module stays light for the pipeline.
    """
    if queries is None:
        from reforge.app.queries import Queries  # lazy: no DB import at module load
        queries = Queries()
    queries.insert_taste_event(entry)
    try:
        append_taste_log(entry)
    except OSError:
        pass  # ephemeral/read-only fs - the worker syncs the DB row to the file later
