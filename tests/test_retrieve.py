"""Unit tests for the pure retrieval logic in match/retrieve.py (no DB / no model)."""
from reforge.llm.base import ClipMoment, ReelBrief, ShotSpec
from reforge.match.retrieve import (
    apply_freshness,
    brief_summary,
    format_candidate_meta,
    freshness_penalty,
    hyde_text,
    merge_candidates,
    quality_adjusted_distance,
    retrieval_filters,
    row_to_moment,
    rrf_fuse,
    select_top_n,
    shot_ask,
)


def _row(mid, dist, key="library/x.mp4", quality=None):
    return {"id": mid, "clip_id": 1, "start_s": 0.0, "end_s": 1.5, "description": "d",
            "energy": "moderate", "location": "gym", "is_logan": True, "framing_notes": "fn",
            "thumbnail_key": "data/thumbs/x/000.jpg", "storage_key": key, "distance": dist,
            "quality_score": quality}


# --- filter concept: "exclude non-Logan person" -----------------------------

def test_retrieval_filters_excludes_non_logan_person_via_is_logan():
    # v1: keep is_logan=true (drops the lone object clip - moot for fitness). The subject
    # enum (logan|object|other|multiple) is the future one-line swap.
    assert retrieval_filters(exclude_non_logan_person=True) == {"is_logan": True}


def test_retrieval_filters_unfiltered_when_disabled():
    assert retrieval_filters(exclude_non_logan_person=False) == {}


# --- merge image + text candidate lists -------------------------------------

def test_merge_dedups_by_moment_id_keeping_nearest():
    image = [_row(10, 0.20), _row(11, 0.35)]
    text = [_row(10, 0.50), _row(12, 0.30)]   # moment 10 appears in both; keep 0.20
    merged = merge_candidates([image, text], k=5)
    ids_dists = [(m["id"], m["distance"]) for m in merged]
    assert ids_dists == [(10, 0.20), (12, 0.30), (11, 0.35)]  # sorted by distance, deduped


def test_merge_respects_top_k():
    rows = [_row(i, i / 10) for i in range(8)]
    assert len(merge_candidates([rows], k=3)) == 3


# --- knn row -> ClipMoment --------------------------------------------------

def test_row_to_moment_maps_fields():
    m = row_to_moment(_row(42, 0.12, key="library/Shorts/Gym/C1402.MP4"))
    assert isinstance(m, ClipMoment)
    assert m.clip_key == "library/Shorts/Gym/C1402.MP4"
    assert m.moment_id == 42
    assert m.is_logan is True
    assert m.thumbnail_key == "data/thumbs/x/000.jpg"
    assert m.distance == 0.12
    assert m.location == "gym"


# --- Phase 3.5 shortlist: pure prompt assembly + top-N selection ------------

def _cand(i, energy="explosive", desc="man does pull-ups", location="gym"):
    return ClipMoment(f"library/k{i}.mp4", 0.0, 1.5, desc, True, "full body", energy,
                      moment_id=i, clip_id=i, location=location)


def test_brief_summary_includes_register_and_theme():
    brief = ReelBrief(vibe="late-night grind", emotional_register="gritty",
                      energy_profile="builds", tone="serious", theme="discipline")
    out = brief_summary(brief)
    assert "gritty" in out and "discipline" in out


def test_shot_ask_includes_role_and_intended_feeling():
    shot = ShotSpec(0, 0.0, 1.0, "a pull-up", "", "explosive", "pull-up",
                    role="payoff", intended_feeling="triumph", intended_energy="explosive")
    out = shot_ask(shot)
    assert "payoff" in out and "triumph" in out


def test_format_candidate_meta_has_energy_location_and_desc():
    line = format_candidate_meta(_cand(0))
    assert "explosive" in line and "gym" in line and "pull-ups" in line


def test_select_top_n_by_score_descending():
    cands = [_cand(i) for i in range(5)]
    picked = select_top_n(cands, [0.1, 0.9, 0.5, 0.95, 0.3], 2)
    assert [c.moment_id for c in picked] == [3, 1]


def test_select_top_n_pads_missing_scores_and_caps():
    cands = [_cand(i) for i in range(3)]
    picked = select_top_n(cands, [0.5], 5)        # only one score; rest default 0
    assert picked[0].moment_id == 0
    assert len(picked) == 3


# --- Phase 3.6 cross-reel freshness -----------------------------------------

def test_freshness_penalty_boosts_never_used():
    assert freshness_penalty(0, 0.0) < 0          # never used -> boost (negative penalty)


def test_freshness_penalty_grows_with_count_and_recency():
    assert freshness_penalty(3, 0.0) > freshness_penalty(1, 0.0)
    assert freshness_penalty(1, 1.0) > freshness_penalty(1, 0.0)


def test_apply_freshness_downweights_used_clip_vs_never_used():
    cands = [_cand(1), _cand(2)]                   # clip_ids 1 and 2
    fit = [0.8, 0.8]                               # tie on taste-fit
    usage = {1: {"count": 3, "recency": 1.0}}      # clip 1 heavily/recently used; clip 2 never
    adjusted = apply_freshness(fit, cands, usage, weight=0.1)
    assert adjusted[1] > adjusted[0]               # fresh clip 2 now ranks above clip 1


def test_apply_freshness_zero_weight_is_noop():
    cands = [_cand(1), _cand(2)]
    assert apply_freshness([0.8, 0.2], cands, {1: {"count": 5, "recency": 1.0}}, weight=0.0) == [0.8, 0.2]


# --- Phase 9B HyDE intent query ----------------------------------------------

def _hyde_shot(**overrides):
    kw = dict(idx=0, start_s=0.0, end_s=1.5, description="does explosive pull-ups at night",
              on_screen_text="", energy="explosive", query="pull-ups outdoor bar night",
              role="payoff", intended_feeling="triumph", intended_energy="explosive")
    kw.update(overrides)
    return ShotSpec(**kw)


def test_hyde_text_reads_like_a_caption_with_shot_tokens():
    out = hyde_text(_hyde_shot())
    assert "Logan" in out                       # library captions describe Logan
    assert "explosive pull-ups at night" in out  # description
    assert "explosive" in out                    # intended energy
    assert "triumph" in out                      # intended feeling
    assert "payoff" in out                       # role in the arc


def test_hyde_text_includes_brief_vibe_and_register():
    brief = ReelBrief(vibe="late-night grind", emotional_register="gritty",
                      energy_profile="builds", tone="serious", theme="discipline")
    out = hyde_text(_hyde_shot(), brief)
    assert "late-night grind" in out and "gritty" in out


def test_hyde_text_works_without_brief_and_falls_back_to_shot_energy():
    out = hyde_text(_hyde_shot(intended_energy="", energy="slow"), None)
    assert "slow" in out


def test_hyde_text_falls_back_to_query_when_no_description():
    out = hyde_text(_hyde_shot(description=""))
    assert "pull-ups outdoor bar night" in out


# --- Phase 9B Reciprocal Rank Fusion ------------------------------------------

def test_rrf_single_list_preserves_its_order():
    rows = [_row(1, 0.9), _row(2, 0.1), _row(3, 0.5)]   # deliberately NOT distance-sorted
    fused = rrf_fuse([rows], k=5)
    assert [r["id"] for r in fused] == [1, 2, 3]         # rank in the lane wins, not distance


def test_rrf_item_in_two_lanes_beats_single_lane_items():
    lane_a = [_row(1, 0.2), _row(2, 0.3)]
    lane_b = [_row(3, 0.1), _row(1, 0.4)]                # moment 1 also ranks 2nd here
    fused = rrf_fuse([lane_a, lane_b], k=5)
    assert fused[0]["id"] == 1                            # 1/(60+1)+1/(60+2) beats any single lane


def test_rrf_dedup_keeps_the_nearest_row_per_moment():
    lane_a = [_row(1, 0.5)]
    lane_b = [_row(1, 0.2)]
    fused = rrf_fuse([lane_a, lane_b], k=5)
    assert len(fused) == 1 and fused[0]["distance"] == 0.2


def test_rrf_k_damps_the_top_rank_advantage():
    # One moment at rank 1 of a single lane vs another at rank 3 of BOTH lanes.
    # Undamped (rrf_k=0): 1/1 > 2/3 -> the single top rank wins.
    # Damped (rrf_k=60): 1/61 < 2/63 -> cross-lane consensus wins.
    lane_a = [_row(1, 0.1), _row(2, 0.2), _row(3, 0.3)]
    lane_b = [_row(4, 0.15), _row(5, 0.2), _row(3, 0.3)]  # m4 ties m1 on rank; m1 nearer
    assert rrf_fuse([lane_a, lane_b], k=5, rrf_k=0)[0]["id"] == 1
    assert rrf_fuse([lane_a, lane_b], k=5, rrf_k=60)[0]["id"] == 3


def test_rrf_quality_downweight_flips_a_close_pair():
    # Adjacent ranks in one lane: without quality the rank-1 junk moment wins;
    # with the down-weight the high-quality rank-2 moment overtakes it.
    rows = [_row(1, 0.2, quality=0.1), _row(2, 0.25, quality=1.0)]
    assert rrf_fuse([rows], k=5, quality_weight=0.0)[0]["id"] == 1
    assert rrf_fuse([rows], k=5, quality_weight=0.5)[0]["id"] == 2


def test_rrf_missing_quality_is_not_penalized():
    rows = [_row(1, 0.2, quality=None), _row(2, 0.25, quality=1.0)]
    fused = rrf_fuse([rows], k=5, quality_weight=0.5)
    assert fused[0]["id"] == 1                            # legacy (None) quality stays neutral


def test_rrf_respects_top_k():
    rows = [_row(i, i / 10) for i in range(8)]
    assert len(rrf_fuse([rows], k=3)) == 3


# --- Phase 3.7 quality down-weighting in retrieval --------------------------

def test_quality_adjusted_distance_penalizes_low_quality():
    # same embedding distance; the higher-quality moment must rank nearer (smaller adj distance)
    hi = quality_adjusted_distance(0.30, quality_score=0.9, weight=0.2)
    lo = quality_adjusted_distance(0.30, quality_score=0.2, weight=0.2)
    assert hi < lo


def test_quality_adjusted_distance_zero_weight_is_noop():
    assert quality_adjusted_distance(0.30, quality_score=0.1, weight=0.0) == 0.30


def test_quality_adjusted_distance_missing_quality_treated_as_neutral():
    # unknown quality (None) should not penalize (legacy moments before the quality pass)
    assert quality_adjusted_distance(0.30, quality_score=None, weight=0.2) == 0.30
