-- 0002: Phase 3.5 taste layer - persist the reel creative brief + per-slot rerank reason
-- so a run is reproducible and the taste is auditable. Idempotent.

alter table reels add column if not exists brief  jsonb;   -- ReelBrief (vibe/register/energy/tone/theme)
alter table slots add column if not exists reason text;    -- one-line rerank rationale for the pick
