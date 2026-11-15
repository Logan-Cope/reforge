-- 0008: review/curation app layer.
-- Persists what the phone-first app needs: runner-up plans per slot (the swap UI),
-- the human verdict per beat, reel approval, and a human thumbs label on library
-- moments (curation). `profiles` is the multi-user-lite namespace (PLAN section 3):
-- every visitor's thumbs live under a profile; Logan's judge stays the default.
-- Idempotent (add column / create table if not exists; upsert-safe inserts).

alter table slots add column if not exists candidates jsonb;           -- runner-up SlotPlans: [{moment_id, clip_key, thumbnail_key, score, reason, in_point}]
alter table slots add column if not exists human_verdict text;         -- null | 'keep' | 'swapped'
alter table slots add column if not exists human_note text;
alter table slots add column if not exists swapped_moment_id bigint;

alter table reels add column if not exists approved boolean;
alter table reels add column if not exists approved_at timestamptz;

alter table moments add column if not exists human_label smallint;     -- null = unrated, 1 = thumbs-up, 0 = thumbs-down

create table if not exists profiles (
  id         bigserial primary key,
  handle     text unique not null,
  created_at timestamptz default now()
);

insert into profiles (handle) values ('logan'), ('demo')
on conflict (handle) do nothing;
