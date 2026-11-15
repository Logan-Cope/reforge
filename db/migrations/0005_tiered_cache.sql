-- 0005: Phase 9A - decouple ingestion into TIER-0 (immutable, expensive, compute-once) and
-- TIER-1 (mutable, cheap, recompute in minutes), content-addressed by (content_hash, version).
--
-- Additive + non-destructive: existing clips/moments/embeddings/reels/slots are untouched.
-- The cache tables are APPEND-ONLY - a new extractor/scorer/model writes rows under a NEW
-- version; old + new coexist for A/B + rollback. Skipping on a (content_hash, version) hit is
-- what turns a re-score from a 2-3h re-ingest into minutes (read Tier-0, recompute Tier-1).
-- Idempotent (create ... if not exists / add column if not exists).

-- ---------------------------------------------------------------------------
-- TIER-0 (immutable, expensive): decode / segment / pose / embed / caption.
-- Keyed by (content_hash, <extractor>_version, window_idx). Never overwritten.
-- ---------------------------------------------------------------------------

-- Shot/segmentation boundaries (deterministic from the motion pass).
create table if not exists cache_windows (
  content_hash      text not null,
  extractor_version text not null,
  window_idx        int  not null,
  start_s           real not null,
  end_s             real not null,
  created_at        timestamptz not null default now(),
  primary key (content_hash, extractor_version, window_idx)
);

-- Extracted frame artifacts (the proxy/thumb in storage). frame_role e.g. 'f0','f1'.
create table if not exists cache_frames (
  content_hash      text not null,
  extractor_version text not null,
  window_idx        int  not null,
  frame_role        text not null,
  storage_key       text not null,
  created_at        timestamptz not null default now(),
  primary key (content_hash, extractor_version, window_idx, frame_role)
);

-- Raw CV signals per window (read at re-score time; never recomputed unless bytes/extractor change).
create table if not exists cache_cv_signals (
  content_hash      text not null,
  extractor_version text not null,
  window_idx        int  not null,
  exposure          real,
  sharpness         real,
  stability         real,
  subject_present   real,
  created_at        timestamptz not null default now(),
  primary key (content_hash, extractor_version, window_idx)
);

-- Raw VLM quality assessment per window (the expensive Haiku call; cache so re-score is free).
create table if not exists cache_vlm_assessments (
  content_hash  text not null,
  vlm_version   text not null,
  window_idx    int  not null,
  usable        real,
  flags         jsonb,
  created_at    timestamptz not null default now(),
  primary key (content_hash, vlm_version, window_idx)
);

-- VLM caption / annotation per clip (one per clip, inherited by every window).
create table if not exists cache_annotations (
  content_hash      text not null,
  annotator_version text not null,
  description       text,
  energy            text,
  location          text,
  is_logan          boolean,
  framing_notes     text,
  created_at        timestamptz not null default now(),
  primary key (content_hash, annotator_version)
);

-- Base visual embedding per window, versioned by model (shadow/dual-write = a new embed_version).
-- 768 = CLIP ViT-L-14 (the current family). A different-dim model gets a sibling table; old +
-- new coexist via embed_version either way.
create table if not exists cache_embeddings (
  content_hash  text not null,
  embed_version text not null,
  window_idx    int  not null,
  model         text,
  embedding     vector(768),
  created_at    timestamptz not null default now(),
  primary key (content_hash, embed_version, window_idx)
);

-- Audio features per clip (structure only for now - Tier-0 slot for 9B beat-snap / energy).
create table if not exists cache_audio_features (
  content_hash      text not null,
  extractor_version text not null,
  features          jsonb,
  created_at        timestamptz not null default now(),
  primary key (content_hash, extractor_version)
);

-- ---------------------------------------------------------------------------
-- TIER-1 (mutable, cheap): derived score per window, keyed by scorer_version.
-- Reads Tier-0 raw signals; recomputing the whole library never touches decode/embed.
-- ---------------------------------------------------------------------------
create table if not exists window_scores (
  content_hash   text not null,
  scorer_version text not null,
  window_idx     int  not null,
  cv_quality     real,
  vlm_quality    real,
  quality_score  real,
  quality_flags  jsonb,
  created_at     timestamptz not null default now(),
  primary key (content_hash, scorer_version, window_idx)
);

create index if not exists idx_window_scores_lookup
  on window_scores (content_hash, scorer_version);

-- ---------------------------------------------------------------------------
-- Link existing moments -> their cached window (content_hash via clips; window_idx here).
-- Backfilled from thumbnail_key (".../NNN.jpg" -> NNN) by the backfill script.
-- ---------------------------------------------------------------------------
alter table moments add column if not exists window_idx int;
create index if not exists idx_moments_window on moments (clip_id, window_idx);
