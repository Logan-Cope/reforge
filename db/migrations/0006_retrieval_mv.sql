-- 0006: materialized retrieval view (Phase 9A). Retrieval reads THIS instead of joining
-- moments + moment_clip_embeddings + clips live. A scorer promote updates moments.quality_score
-- then REFRESHes the view (seconds) - no re-ingest. Selects exactly the columns `knn` returns,
-- so results are identical to the prior live join (verified before/after).
--
-- Non-destructive: a view over existing tables; drop it and retrieval falls back to the live join.

drop materialized view if exists mv_moment_retrieval;

create materialized view mv_moment_retrieval as
select m.id, m.clip_id, m.start_s, m.end_s, m.description, m.energy, m.location,
       m.is_logan, m.framing_notes, m.thumbnail_key, m.quality_score,
       c.storage_key, e.embedding
from moments m
join moment_clip_embeddings e on e.moment_id = m.id
join clips c on c.id = m.clip_id
where m.superseded = false;

-- Unique index on the moment id -> enables REFRESH MATERIALIZED VIEW CONCURRENTLY.
create unique index if not exists idx_mv_moment_id on mv_moment_retrieval (id);
-- Same ANN index the live embedding table had, now on the view.
create index if not exists idx_mv_emb_hnsw on mv_moment_retrieval using hnsw (embedding vector_cosine_ops);
create index if not exists idx_mv_is_logan on mv_moment_retrieval (is_logan);
create index if not exists idx_mv_location on mv_moment_retrieval (location);
