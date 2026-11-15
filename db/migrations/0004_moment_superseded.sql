-- 0004: soft-supersede moments instead of deleting on re-ingest.
-- Re-segmentation (Phase 3.7) changes the moment set, but prior reels/slots reference old
-- moment ids (preserve those records + their freshness history). So re-ingest marks old
-- moments superseded=true and inserts fresh ones; retrieval (knn) only sees superseded=false.
-- Idempotent.

alter table moments add column if not exists superseded boolean not null default false;
create index if not exists idx_moments_superseded on moments(superseded);
