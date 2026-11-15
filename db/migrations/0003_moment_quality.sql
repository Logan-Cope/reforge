-- 0003: Phase 3.7 per-moment quality. Local CV signals + a one-time VLM pass combine into
-- quality_score; retrieval down-weights low quality, the rerank sees it. Egregious moments
-- (no subject / camera-handling) are hard-dropped at ingest and never inserted. Idempotent.

alter table moments add column if not exists cv_quality     real;
alter table moments add column if not exists vlm_quality    real;
alter table moments add column if not exists quality_score  real;   -- combined 0-1; null = pre-3.7 (neutral)
alter table moments add column if not exists quality_flags  jsonb;  -- e.g. ["dark","blurry","transitional"]
