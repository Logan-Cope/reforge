-- 0010: Reforge's own reference-acquisition dedup store.
-- The acquire pipeline (scripts/acquire_and_recreate.py) turns a list of Instagram reel
-- URLs into recreated reels. This table is Reforge's OWN dedup + audit ledger for that
-- front door (it does NOT touch the upstream discovery tool): one row per source reel, keyed
-- by IG shortcode, tracking whether it was pulled, recreated, or skipped (and why).
-- Idempotent (create table if not exists).

create table if not exists source_reels (
  shortcode          text primary key,
  url                text not null,
  title              text,
  source             text default 'vidiq',
  caption_density    real,
  recreated_reel_id  bigint references reels(id),
  status             text not null default 'pulled',   -- pulled | recreated | skipped_caption | skipped_dup | error
  skipped_reason     text,
  pulled_at          timestamptz default now(),
  recreated_at       timestamptz
);
