#!/usr/bin/env python3
"""Backfill the Phase 9B caption-text lane: embed every active moment's caption.

For each active (superseded = false) moment, composes its caption text (description +
energy + location + framing notes - `ingest.embed.caption_text`) and embeds it with
OpenAI text-embedding-3-small at dimensions=768 into moment_caption_embeddings.

Idempotent: moments that already have a caption embedding are skipped unless --force.
Moments with an entirely empty caption are skipped (nothing to embed). Full library
(~3.8k moments) costs pennies.

Usage:
    PYTHONPATH=src python scripts/embed_captions.py [--force] [--batch 512]
"""
import argparse

from reforge.config import settings
from reforge.db.client import get_conn
from reforge.ingest.embed import caption_text, openai_text_embed

EMBED_DIM = 768  # must match moment_caption_embeddings (migration 0007)

UPSERT_SQL = """
    insert into moment_caption_embeddings (moment_id, model, embedding, embed_version)
    values (%s, %s, %s, %s)
    on conflict (moment_id) do update set
        model = excluded.model,
        embedding = excluded.embedding,
        embed_version = excluded.embed_version
"""


def main() -> None:
    ap = argparse.ArgumentParser(description="Embed active moments' captions (Phase 9B lane).")
    ap.add_argument("--force", action="store_true",
                    help="re-embed moments that already have a caption embedding")
    ap.add_argument("--batch", type=int, default=512, help="texts per OpenAI request")
    args = ap.parse_args()

    import numpy as np

    model = settings.text_embed_model
    version = f"{model}@{EMBED_DIM}"
    skip = ("" if args.force else
            " and not exists (select 1 from moment_caption_embeddings e where e.moment_id = m.id)")
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(f"""select m.id, m.description, m.energy, m.location, m.framing_notes
                            from moments m
                            where m.superseded = false{skip}
                            order by m.id""")
            rows = cur.fetchall()
        keys = ("id", "description", "energy", "location", "framing_notes")
        todo = [(row[0], caption_text(dict(zip(keys, row)))) for row in rows]
        empty = sum(1 for _, text in todo if not text)
        todo = [(mid, text) for mid, text in todo if text]
        print(f"[embed_captions] {len(todo)} moments to embed "
              f"({empty} skipped: empty caption; --force={args.force})")

        done = 0
        for i in range(0, len(todo), args.batch):
            batch = todo[i:i + args.batch]
            vecs = openai_text_embed([text for _, text in batch], model=model,
                                     dimensions=EMBED_DIM)
            payload = [(mid, model, np.asarray(vec, dtype=np.float32), version)
                       for (mid, _), vec in zip(batch, vecs)]
            with conn.cursor() as cur:
                cur.executemany(UPSERT_SQL, payload)
            conn.commit()
            done += len(batch)
            print(f"[embed_captions] embedded {done}/{len(todo)}")

        with conn.cursor() as cur:
            cur.execute("select count(*) from moment_caption_embeddings")
            total = cur.fetchone()[0]
        print(f"[embed_captions] done: {done} embedded this run; "
              f"{total} total rows in moment_caption_embeddings")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
