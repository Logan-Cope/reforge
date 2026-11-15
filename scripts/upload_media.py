#!/usr/bin/env python3
"""Upload preview media (480p proxies + thumbnails) to Supabase Storage.

The deployed app serves clips from a public "media" bucket instead of the 84GB
local originals: thumbnails for review/curation cards, proxies for preview
renders. Idempotent: a local manifest (data/.upload_manifest) records uploaded
keys so reruns only push new files; --force re-pushes everything (x-upsert).

Usage:
    PYTHONPATH=src python scripts/upload_media.py [--thumbs] [--proxies] [--force]
    (no flags = both)
"""
from __future__ import annotations

import argparse
import mimetypes
import os
import sys

import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from reforge.config import settings  # noqa: E402

BUCKET = "media"
MANIFEST = "data/.upload_manifest"


def _headers() -> dict:
    return {"Authorization": f"Bearer {settings.supabase_key}", "apikey": settings.supabase_key}


def ensure_bucket() -> None:
    r = requests.post(f"{settings.supabase_url}/storage/v1/bucket", headers=_headers(),
                      json={"id": BUCKET, "name": BUCKET, "public": True}, timeout=30)
    if r.status_code not in (200, 201) and "already exists" not in r.text.lower():
        raise RuntimeError(f"bucket create failed: {r.status_code} {r.text[:200]}")


def upload_tree(local_root: str, key_prefix: str, done: set[str], force: bool) -> int:
    n = 0
    for dirpath, _dirs, files in os.walk(local_root):
        for name in sorted(files):
            if name.startswith("."):
                continue
            path = os.path.join(dirpath, name)
            key = f"{key_prefix}/{os.path.relpath(path, local_root)}"
            if not force and key in done:
                continue
            ctype = mimetypes.guess_type(name)[0] or "application/octet-stream"
            with open(path, "rb") as fh:
                r = requests.post(
                    f"{settings.supabase_url}/storage/v1/object/{BUCKET}/{key}",
                    headers={**_headers(), "Content-Type": ctype, "x-upsert": "true"},
                    data=fh, timeout=120)
            if r.status_code not in (200, 201):
                print(f"FAIL {key}: {r.status_code} {r.text[:120]}", flush=True)
                continue
            done.add(key)
            n += 1
            if n % 100 == 0:
                print(f"  ...{n} uploaded ({key})", flush=True)
                _save(done)
    return n


def _save(done: set[str]) -> None:
    with open(MANIFEST, "w") as fh:
        fh.write("\n".join(sorted(done)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--thumbs", action="store_true")
    ap.add_argument("--proxies", action="store_true")
    ap.add_argument("--outputs", action="store_true", help="rendered reels (output/*.mp4)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    both = not (args.thumbs or args.proxies or args.outputs)

    if not (settings.supabase_url and settings.supabase_key):
        sys.exit("SUPABASE_URL / SUPABASE_SERVICE_KEY missing from .env")
    ensure_bucket()
    done: set[str] = set()
    if os.path.exists(MANIFEST):
        done = set(open(MANIFEST).read().splitlines())

    total = 0
    if args.thumbs or both:
        total += upload_tree("data/thumbs", "thumbs", done, args.force)
    if args.proxies or both:
        total += upload_tree("data/proxies", "proxies", done, args.force)
    if args.outputs or both:
        total += upload_tree("output", "output", done, args.force)
    _save(done)
    print(f"DONE: {total} uploaded this run, {len(done)} total in manifest")
    print(f"public base: {settings.supabase_url}/storage/v1/object/public/{BUCKET}/")


if __name__ == "__main__":
    main()
