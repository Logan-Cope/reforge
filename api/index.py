"""Vercel serverless entry: the Reforge review app as an ASGI function.

The heavy pipeline (CLIP, ffmpeg, 84GB originals) never runs here - this serves
review/curation/run-queue over Supabase (DB + public media bucket). Local dev
keeps using scripts/serve_app.py.

The pre-rewrite request path is restored by ``RestoreRewrittenPath`` inside
``create_app`` - see the note there.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from reforge.app.main import create_app  # noqa: E402

app = create_app()
