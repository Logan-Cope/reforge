-- 0011: cut freshness - does the rendered mp4 still match the reviewer's picks?
-- reels.cut_dirty flips true on any pick change that alters the cut (swap,
-- undo-of-a-swap, repick) and back to false when the worker completes a
-- rerender:<id> job. Backfill: reels with a live swap and no completed rerender
-- are dirty. Reel 14 is explicitly dirty - a swap was rendered into v2 and then
-- undone before this flag existed, so the file still contains the undone swap.
-- Idempotent (add column if not exists; the updates only ever set the flag).

alter table reels add column if not exists cut_dirty boolean not null default false;

update reels r set cut_dirty = true
where exists (select 1 from slots s
              where s.reel_id = r.id and s.human_verdict = 'swapped')
  and not exists (select 1 from run_requests q
                  where q.reference_key = 'rerender:' || r.id and q.status = 'done');

update reels set cut_dirty = true where id = 14;
