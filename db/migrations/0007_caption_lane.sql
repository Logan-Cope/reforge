-- 0007: caption retrieval lane (Phase 9B advanced retriever).
--
-- (a) moment_caption_embeddings -> vector(768). The schema's 1536 was a placeholder for
--     OpenAI text-embedding-3-small's native dim; we request the documented
--     `dimensions=768` truncation instead so the caption lane matches the CLIP lane's
--     dim and storage. The table was empty at migration time, so the type alter is free.
--     `embed_version` records model@dims per row for future swaps (Phase 9A pattern).
-- (b) moments.caption_tsv: generated tsvector over the VLM caption + GIN index, the
--     BM25-style full-text lane fused with the vector lanes in match/retrieve.py.
--
-- Non-destructive: the embeddings table had no rows; caption_tsv is a generated column.

do $$
begin
  if exists (select 1
             from pg_attribute a join pg_class c on c.oid = a.attrelid
             where c.relname = 'moment_caption_embeddings'
               and a.attname = 'embedding' and a.atttypmod <> 768) then
    execute 'alter table moment_caption_embeddings '
            'alter column embedding type vector(768) using embedding::vector(768)';
  end if;
end $$;

alter table moment_caption_embeddings add column if not exists embed_version text;

-- ANN index (recreated after any type change; name matches db/schema.sql).
drop index if exists idx_caption_emb_hnsw;
create index idx_caption_emb_hnsw
  on moment_caption_embeddings using hnsw (embedding vector_cosine_ops);

-- BM25 lane: generated tsvector over the caption + GIN index.
alter table moments add column if not exists caption_tsv tsvector
  generated always as (to_tsvector('english', coalesce(description, ''))) stored;
create index if not exists idx_moments_caption_tsv on moments using gin (caption_tsv);
