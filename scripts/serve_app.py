#!/usr/bin/env python3
"""Serve the Reforge review/curation app.

Usage:
    python scripts/serve_app.py            # uvicorn on 0.0.0.0:8000
    python scripts/serve_app.py --reload   # dev auto-reload
"""
import argparse
import os
import sys

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")


def main() -> None:
    ap = argparse.ArgumentParser(description="Serve the Reforge review/curation app.")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--reload", action="store_true", help="dev auto-reload")
    args = ap.parse_args()

    # `reforge` lives under src/ (no editable install); cover this process + reload workers.
    if SRC not in sys.path:
        sys.path.insert(0, SRC)
    os.environ["PYTHONPATH"] = SRC + os.pathsep + os.environ.get("PYTHONPATH", "")

    import uvicorn
    uvicorn.run("reforge.app.main:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
