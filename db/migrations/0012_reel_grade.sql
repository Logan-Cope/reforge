-- 0012: graded approve - one reel-level judgment replaces tapping Keep per beat.
-- grade is the reviewer's confidence dial for the whole reel:
--   post_worthy  -> unflagged beats are strong positives (would post as-is)
--   would_test   -> unflagged beats are okay (would trial it, not the essence)
--   not_it       -> the concept is rejected (feeds the discovery side)
-- A positive grade endorses every beat the reviewer did NOT flag/swap, so the
-- judge learns from all of them without a per-beat tap. `approved` stays as the
-- derived boolean (true for the two positive grades) for the dashboard + counts.
-- Idempotent.

alter table reels add column if not exists grade text
  check (grade in ('post_worthy', 'would_test', 'not_it'));
alter table reels add column if not exists graded_at timestamptz;

-- Backfill: reels approved before grades existed become 'post_worthy' (the old
-- Approve button meant "good to post").
update reels set grade = 'post_worthy'
where approved = true and grade is null;
