-- 0001: content-hash dedup for clips.
-- sha256 of the file bytes, computed on ingest; dedups re-downloaded / overlapping
-- batches independent of filename or storage_key. Nullable + unique (Postgres allows
-- many NULLs under a unique index). Idempotent so it is safe to re-apply.
--
-- The base db/schema.sql already declares `content_hash text unique`; this migration
-- brings a live DB that predates that column up to date (ALTER the existing project).

alter table clips add column if not exists content_hash text;
create unique index if not exists clips_content_hash_key on clips (content_hash);
