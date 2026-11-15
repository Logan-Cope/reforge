-- 0013: trial-reel publish scheduling.
-- The app's "Schedule trial reel" button inserts here (caption copied from the reel at
-- schedule time); scripts/publish_worker.py claims due rows (FOR UPDATE SKIP LOCKED,
-- same pattern as run_requests) and hands them to the configured publish provider.
-- Dry-run results land in `result` with status 'dry_run_done' and NEVER touch reels;
-- only a real successful publish moves reels.status to 'published'.
-- Idempotent (create table / add column if not exists).

-- schema.sql declares these publish columns on reels, but the live DB predates them
-- (no earlier migration added them) - complete the declared schema here.
alter table reels add column if not exists caption     text;
alter table reels add column if not exists is_trial    boolean default true;
alter table reels add column if not exists ig_media_id text;
alter table reels add column if not exists posted_at   timestamptz;

create table if not exists scheduled_posts (
  id           bigserial primary key,
  reel_id      bigint not null references reels(id) on delete cascade,
  scheduled_at timestamptz not null,
  status       text not null default 'scheduled',
    -- scheduled | publishing | published | dry_run_done | error | canceled
  as_trial     boolean not null default true,
  caption      text,
  provider     text,
  result       jsonb,
  error        text,
  created_at   timestamptz default now(),
  started_at   timestamptz,
  published_at timestamptz
);

create index if not exists idx_scheduled_posts_due
  on scheduled_posts (scheduled_at) where status = 'scheduled';
create index if not exists idx_scheduled_posts_reel on scheduled_posts (reel_id);
