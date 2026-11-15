"""Embedding-model swap via versioned shadow rows + background backfill (Phase 9A).

NO full re-ingest: the decoded frames are cached (`cache_frames`), so a swap re-runs ONLY the
forward pass over those frames and writes `cache_embeddings` under a NEW `embed_version`. Old +
new coexist for A/B; cut over later by pointing the retrieval view at the new version. Because
pgvector columns are fixed-dim, a new version IS the shadow column (old + new side by side).

  python scripts/swap_embeddings.py --to-version clip-shadow-1 --limit 5

Resumable (append-only, skips clips already embedded at the target version). The point being
proven: a swap touches frames-on-disk only - never the video decoder.
"""
from __future__ import annotations

import argparse
import os
import time

from reforge.db.client import (
    cache_put_embedding, cached_content_hashes, get_conn, read_frames,
)
from reforge.ingest import cache


def main() -> None:
    ap = argparse.ArgumentParser(description="Shadow-backfill a new embedding version from cached frames.")
    ap.add_argument("--to-version", required=True, help="new embed_version label (the shadow)")
    ap.add_argument("--limit", type=int, default=None, help="only the first N clips (for a quick proof)")
    args = ap.parse_args()

    from reforge.ingest.embed import clip_embed, clip_model_name

    conn = get_conn()
    ev = cache.EXTRACTOR_VERSION
    todo = sorted(cached_content_hashes("cache_windows", ev, conn=conn))
    done = cached_content_hashes("cache_embeddings", args.to_version, conn=conn)
    todo = [h for h in todo if h not in done]
    if args.limit is not None:
        todo = todo[: args.limit]
    print(f"shadow-embedding {len(todo)} clips -> embed_version={args.to_version} "
          f"(forward pass over CACHED frames; no decode)", flush=True)

    t0 = time.monotonic()
    n_clips = n_emb = n_missing = 0
    for chash in todo:
        frames = read_frames(chash, ev, "f0", conn=conn)
        for widx, fkey in sorted(frames.items()):
            path = fkey if os.path.isabs(fkey) else os.path.abspath(fkey)
            if not os.path.exists(path):
                n_missing += 1
                continue
            cache_put_embedding(chash, args.to_version, widx, clip_embed(path),
                                model=clip_model_name(), conn=conn)
            n_emb += 1
        conn.commit()
        n_clips += 1

    dt = time.monotonic() - t0
    print(f"\nDone: {n_clips} clips, {n_emb} embeddings under embed_version={args.to_version}, "
          f"{n_missing} missing frames. {dt:.1f}s ({n_emb/dt:.1f} emb/s) - NO video decode.", flush=True)
    conn.close()


if __name__ == "__main__":
    main()
