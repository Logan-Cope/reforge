-- 0009: serverless app support (Vercel deploy; ephemeral filesystem).
-- The deployed app can't append to taste/taste-log.jsonl or run the heavy pipeline, so:
--   taste_events - app-layer taste writes land here; the local worker syncs each entry
--                   to taste/taste-log.jsonl (synced=true) so the judge's file stays whole.
--   run_requests - the app's "Run" button queues here; the local worker (scripts/worker.py)
--                   claims queued rows (FOR UPDATE SKIP LOCKED) and executes scripts/recreate.py.
-- Idempotent (create table if not exists).

create table if not exists taste_events (
  id         bigserial primary key,
  created_at timestamptz default now(),
  synced     boolean default false,
  entry      jsonb not null
);

create table if not exists run_requests (
  id            bigserial primary key,
  created_at    timestamptz default now(),
  reference_key text not null,
  status        text not null default 'queued',   -- queued | running | done | error
  started_at    timestamptz,
  finished_at   timestamptz,
  reel_id       bigint,
  error         text
);
