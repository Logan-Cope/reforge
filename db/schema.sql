-- Reforge schema (Supabase / Postgres + pgvector)
-- Embedding dims are placeholders: confirm against the chosen models.
--   CLIP ViT-L-14 = 768 ; OpenAI text-embedding-3-small = 1536.

create extension if not exists vector;

-- A source clip (original lives in storage, NOT here; we keep metadata only).
create table if not exists clips (
  id            bigint generated always as identity primary key,
  storage_key   text not null unique,        -- e.g. 'library/C2046.MP4'
  content_hash  text unique,                 -- sha256 of file bytes; dedups re-downloaded/overlapping clips
  filename      text,
  duration_s    real,
  width         int,
  height        int,
  fps           real,
  is_logan      boolean,                      -- null = unknown / mixed
  notes         text,
  added_at      timestamptz default now()
);

-- A good moment inside a clip (the matchable unit).
create table if not exists moments (
  id            bigint generated always as identity primary key,
  clip_id       bigint not null references clips(id) on delete cascade,
  start_s       real not null,
  end_s         real not null,
  description   text,                         -- VLM caption
  energy        text,                         -- 'explosive' | 'slow' | 'ambient' ...
  location      text,                         -- 'gym' | 'dock' | 'street' ...
  is_logan      boolean,
  framing_notes text,
  thumbnail_key text,                         -- proxy/thumb in storage
  created_at    timestamptz default now()
);

-- CLIP frame embedding per moment.
create table if not exists moment_clip_embeddings (
  moment_id  bigint primary key references moments(id) on delete cascade,
  model      text,
  embedding  vector(768)
);

-- Caption-text embedding per moment.
create table if not exists moment_caption_embeddings (
  moment_id  bigint primary key references moments(id) on delete cascade,
  model      text,
  embedding  vector(1536)
);

-- Optional: Twelve Labs Marengo embedding (eval benchmark).
create table if not exists moment_marengo_embeddings (
  moment_id  bigint primary key references moments(id) on delete cascade,
  model      text,
  embedding  vector(1024)
);

-- A recreation run.
create table if not exists reels (
  id                bigint generated always as identity primary key,
  reference_key     text,                      -- reference reel in storage
  reference_url     text,                      -- original IG url
  reference_caption text,                      -- the source reel's own caption (a signal for ours)
  total_duration    real,
  cut_grid          jsonb,                     -- [[start,end], ...]
  caption           text,                      -- our generated one-line caption (Logan's voice, no CTA)
  hashtags          text[],                    -- posted as an auto first-comment
  status            text default 'draft',      -- draft | approved | published
  is_trial          boolean default true,      -- posted as a Trial Reel (non-followers only)
  trial_status      text,                      -- testing | winner | promoted | killed
  trial_metrics     jsonb,                     -- views, non-follower reach, engagement (pulled after 24-72h)
  ig_media_id       text,                      -- Instagram media id once published
  output_key        text,
  posted_at         timestamptz,
  promoted_at       timestamptz,               -- when a winner was promoted to the main page
  created_at        timestamptz default now()
);

-- One slot of a reel (maps a reference shot to a chosen moment + framing).
create table if not exists slots (
  id               bigint generated always as identity primary key,
  reel_id          bigint not null references reels(id) on delete cascade,
  idx              int not null,
  start_s          real,
  end_s            real,
  ref_description  text,
  ref_text_overlay text,
  ref_energy       text,
  chosen_moment_id bigint references moments(id),
  params           jsonb,                      -- {in_point, zoom, xfrac, yfrac, brightness}
  created_at       timestamptz default now()
);

-- Evaluation scores.
create table if not exists eval_scores (
  id         bigint generated always as identity primary key,
  reel_id    bigint references reels(id) on delete cascade,
  metric     text,                            -- structural_match | framing_quality | beat_sync | would_post | ctx_precision_at_k
  value      real,
  detail     jsonb,
  created_at timestamptz default now()
);

-- ANN indexes (HNSW). Build after data is loaded.
create index if not exists idx_clip_emb_hnsw
  on moment_clip_embeddings using hnsw (embedding vector_cosine_ops);
create index if not exists idx_caption_emb_hnsw
  on moment_caption_embeddings using hnsw (embedding vector_cosine_ops);

-- Helpful filters
create index if not exists idx_moments_is_logan on moments(is_logan);
create index if not exists idx_moments_location on moments(location);
