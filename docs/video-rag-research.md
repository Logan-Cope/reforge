# Reforge Research: Getting the Best Shots Into the Final Video

A deep-research synthesis (2025-2026 state of the art) on how to take raw B-roll, keep only the best shots, retrieve the best-fitting ones, and use them well in the final cut. Grounded in a critique of what Reforge does today. Written to be acted on; sources linked throughout.

## The big picture (read this first)

Two findings reframe everything:

1. **You are not far off, and the parts you built on instinct are the parts the research says actually drive quality.** Across the literature, output quality in an assembly system comes from the **selection / rerank stage** and from **selecting for narrative and pacing**, not from a fancier base embedding. "Cross-encoder reranking is the largest single improvement to RAG quality" ([ZeroEntropy](https://zeroentropy.dev/articles/ultimate-guide-to-choosing-the-best-reranking-model-in-2025/)); the best video systems use cheap recall then "reserve the precise but costly cross-encoder for a second-pass reranking on a small candidate set" ([Cascaded Embedding-Reranking](https://arxiv.org/pdf/2512.12935)); and human-inspired editing work finds that "approaches that prioritize narrative and audience experience perform better" ([Human-Inspired Video Editing](https://arxiv.org/html/2507.02790v1)). That is your Opus rerank, your creative brief, your reference cut-structure, and your planned human-review pass. The instincts are right.

2. **The thing that has hurt most (guess, then a 2-3 hour re-ingest) is an architecture problem with a clean fix, and that fix also makes every other improvement cheap to try.** Today scoring, tagging, and embedding are fused into one monolithic pass, so changing the cheapest thing re-runs the most expensive thing. Decoupling that is the keystone move.

So the gap is not "we are doing it wrong." It is three concrete weak links plus an iteration tax. Fix the tax first, then the weak links become afternoons instead of multi-hour gambles.

## What we are doing right (keep it)

- **Retrieve then rerank.** Correct shape. Recall from cheap vector search, precision from a strong reranker. ([ZeroEntropy](https://zeroentropy.dev/articles/ultimate-guide-to-choosing-the-best-reranking-model-in-2025/), [BigData Boutique](https://bigdataboutique.com/blog/rag-reranking-improving-retrieval-quality-with-cross-encoders))
- **Opus rerank with a reel-level creative brief.** An LLM-as-judge over a small candidate set is exactly where the quality is, and listwise reasoning against a brief is something a cheap cross-encoder cannot do.
- **Recreating the reference's cut structure.** Selecting for sequence and pacing is the high-leverage move for video, not marginal per-clip match quality. ([Human-Inspired Video Editing](https://arxiv.org/html/2507.02790v1))
- **Drop nothing, down-weight, human owns removal.** The research validates this hard: generic aesthetic scorers systematically punish dark and unconventional footage, so they must never be a gate (more below). A human shortlist-approval pass is the durable industry pattern, not a crutch.

## The three real gaps (the critique)

1. **Single-frame CLIP is blind to action and energy.** This is a documented failure, not a hunch. CLIP hits only ~25% top-1 accuracy on verb alignment because "the verb signal is weak and easily confounded" ([arxiv 2508.04945](https://arxiv.org/pdf/2508.04945)), and a single frame "may miss subtle or long-range motion cues." So a boring standing shot and an explosive rep in the same gym look the same to it. Two sub-problems: the embedding cannot tell action apart, and we grab a blind midpoint frame instead of the strongest sub-window.

2. **Retrieval returns look-alikes, not best-fits, for two distinct reasons.** CLIP pools a frame into one vector that matches global scene gestalt rather than the specific salient content a beat needs, and it cannot match by verb/intent. These need two different fixes (a caption-text lane and late interaction), not one.

3. **Blind windows + center crop + monolithic ingestion.** We cut mechanical 1.75s windows instead of real shots; we center-crop so off-center subjects on wide/drone clips fall out of frame; and any change triggers the full re-ingest.

## Ranked improvements (own-code first)

Ordered by return on effort. Note the pattern: the highest-quality levers are also the cheapest to iterate **once ingestion is decoupled**, which is why #0 comes first.

### #0 (keystone, do first): Decouple ingestion so re-scoring takes minutes, not hours

Split the pipeline into two tiers, content-addressed, compute-once:

- **Tier 0, immutable and expensive (compute once per clip, never recompute unless the bytes or extractor version change):** decoded frames, pose results, scene/shot boundaries, base visual embeddings, audio features. Key every artifact by `(content_hash, extractor_version)` and skip on cache hit. This is the DVC run-cache idea: a stage "executes only if it has not already been run with the same dependencies and outputs" ([DVC Pipelines](https://doc.dvc.org/user-guide/pipelines)). Content-ingestion systems report ~31x speedups on incremental updates from exactly this ([Unstructured](https://unstructured.io/insights/incremental-data-ingestion-strategies-for-continuous-pipelines)).
- **Tier 1, mutable and cheap (recompute freely):** quality scores, tags, ranking features. These read Tier 0 and produce small rows. Re-running across the whole library should be seconds to minutes because they never touch the decoder or the embedding model.
- **Version every feature column, never overwrite.** New scorer or model = new rows under a new version, so old and new coexist for A/B and rollback. For pgvector: a `model_version` column, or embedding tables partitioned by version ([Postgres Fastware](https://www.postgresql.fastware.com/blog/how-to-store-and-query-embeddings-in-postgresql-without-losing-your-mind)).
- **Embedding-model swaps without a full re-ingest:** shadow/dual-write the new vectors into a separate column, background-backfill, cut over when done ([Qdrant migration](https://qdrant.tech/documentation/tutorials-operations/embedding-model-migration/)). And note the cost reality: extracted embeddings are ~0.6% of the compressed video size ([Deja Vu](https://arxiv.org/pdf/2506.14107)); the expense is decode + forward-pass, which Tier-0 caching removes from every future iteration.
- Serve retrieval from a **materialized view** joining the current embedding + current scores per clip; `REFRESH` on a scorer change instead of re-ingesting.

Net effect: a new quality scorer touches only Tier 1 (minutes). A new embedding model touches one Tier-0 sub-stage via shadow column + backfill (no decode re-run, frames are cached). The 2-3 hour pass happens only when raw clips change. This single change turns the whole "guess then gamble 2-3 hours" loop into "guess then see it in minutes," which de-risks everything below.

### #1: Caption-text embedding lane + hybrid fusion + HyDE queries (highest retrieval ROI, near-zero infra)

Directly attacks the verb/action weakness, all inside current pgvector.

- **Caption-text lane.** For each moment, also store a text embedding of a VLM caption that explicitly describes action/energy/motion ("explosive push-up at the bottom of the rep, high effort"). A caption can express verbs that CLIP cannot, and "using information-rich captions as semantic embeddings enables retrieval of semantically similar scenes that are visually diverse" ([SSE, arxiv 2409.13860](https://arxiv.org/pdf/2409.13860)). We already run a VLM at ingest, so this is one extra embedding call.
- **Hybrid fusion.** Combine the CLIP image vector, the caption-text vector, BM25 over captions, and structured tags via Reciprocal Rank Fusion (k≈60), which needs no score normalization and lifts recall ~15-30% ([fusion guide](https://mbrenndoerfer.com/writing/hybrid-search-bm25-dense-retrieval-fusion)). pgvector runs the lanes as separate queries and fuses in app code. Worth A/B-ing RRF vs weighted fusion (genuinely contested in 2025 benchmarks).
- **HyDE / retrieve by intent.** Instead of retrieving by a reference frame's appearance, have the LLM generate the ideal-shot description for the beat and retrieve against that ([HyDE explainer](https://zilliz.com/learn/improve-rag-and-information-retrieval-with-hyde-hypothetical-document-embeddings)). This is literally how production B-roll tools work: generate a semantic intent, match it against pre-stored keyframe embeddings ([CVPR 2025 IViSE](https://openaccess.thecvf.com/content/CVPR2025W/IViSE/papers/Schoeffmann_AI-based_Video_Content_Understanding_for_Automatic_and_Interactive_Multimedia_Retrieval_CVPRW_2025_paper.pdf)). It pairs perfectly with the caption lane and reuses the creative brief you already build.

### #2: Multi-frame embeddings + a motion-energy signal + best-moment picker (fixes boring-vs-explosive)

- **Move off single-frame.** Cheapest meaningful win: keep pgvector, sample 4-8 frames per window with **SigLIP 2** (Apache-2.0, self-hosted, beats CLIP at every scale, [arxiv 2502.14786](https://arxiv.org/pdf/2502.14786)) and **attention-pool** (not mean-pool) into one vector. In parallel, A/B **X-CLIP** (MIT-ish, 512-dim, true cross-frame attention, ~192ms/clip self-hosted), which in the one clean public benchmark beat both frame-averaged SigLIP and a 40x-larger video model on retrieval ([Mixpeek benchmark](https://mixpeek.com/blog/video-embedding-benchmark-2026)). Caveat: that benchmark is generic text-to-video, not own-B-roll; the decisive test is a one-afternoon eval on your clips using their open harness. Do **not** reach for paid video APIs (Twelve Labs Marengo, Gemini Embedding 2) yet; they top the leaderboard but break self-hosting and the gap is mostly on hard text retrieval you do not have.
- **Add a motion-energy scalar** per moment (optical-flow magnitude / frame-diff / pose velocity) at ingest, stored as a tag. Near-zero cost, directly encodes "energetic," usable as a fusion signal or sort key. This is the cheapest single thing that addresses your exact complaint.
- **Best-moment sub-window picker.** Replace the blind midpoint with the peak-motion sub-window (same cheap signal). Graduate later to V-JEPA 2 motion features or an unsupervised highlight detector if needed. Orthogonal to the embedding choice, and likely the biggest qualitative jump for "explosive rep" reels.

### #3: Real shots + style-aware quality (no more blind windows, no aesthetic gates)

- **Cut real shots.** Replace mechanical windows with **TransNetV2** (MIT, handles gradual transitions and fast cuts) as primary, **PySceneDetect AdaptiveDetector** as the cheap CPU fallback (its rolling-average pass suppresses handheld false cuts). **AutoShot** is the strongest on paper for short social video if you can bake it off on GPU ([AutoShot](https://ar5iv.labs.arxiv.org/html/2304.06116)). Long continuous takes correctly return one shot; that is where the best-moment picker (#2) takes over.
- **Quality = technical defects only, plus a style-aware VLM, percentile-relative.** This is the research-backed fix for the over-flagging that burned us. Generic aesthetic scorers (LAION-Aesthetics, AVA-NIMA) "encode a narrow conventional-pretty-photo gaze" and systematically demote dark/soft/unusual footage; a 2026 audit documents exactly this bias ([Algorithmic Gaze](https://arxiv.org/abs/2601.09896)). DOVER's own authors warn that "a snow scene almost always has low lighting, but such low-lighting does not indicate low technical quality" ([DOVER](https://github.com/VQAssessment/DOVER)). So:
  - Hard-reject only genuine technical defects with **ARNIQA** (Apache-2.0, distortion-focused, [repo](https://github.com/miccunifi/ARNIQA)) plus optional DOVER-technical: real motion blur, hard clipping, fully crushed black. **Explicitly exclude brightness and contrast from reject logic.**
  - Let the **VLM carry aesthetic judgment** with a prompt that names the style as intentional ("this creator shoots intentionally dark, low-key, handheld, sometimes back-to-camera; do not penalize darkness, softness, grain, or facing away; reward physical/emotional peak and motion energy").
  - **Score percentile-relative** (rank a moment within its own shot or the creator's corpus), so moody footage is compared against other moody footage, not a conventional baseline. This structurally neutralizes residual model bias.
  - This is the principled version of "drop nothing, down-weight," and it makes the down-weight signal trustworthy instead of noisy.

### #4: Subject-aware reframe, pinned to no-zoom (fixes the off-center/drone problem)

The locked center-crop is the issue, and the punch-in regression was a tuning failure, not an architecture one. Every credible tool uses the same skeleton: **per-frame subject position, fed through a temporal smoother, panning the crop horizontally with the zoom factor pinned to 1.0 (full source height).**

- Roll YOLO (person/track) + MediaPipe (pose/face) + **EMA or Kalman smoothing** of the crop-center-x, clamp max step per frame, clamp center so the crop never runs off-frame. Pinning zoom to 1.0 structurally prevents any punch-in. Reference implementation to fork: **KazKozDev/auto-vertical-reframe** (MIT, ffmpeg-based, exposes min/max-zoom and per-frame motion limits as flags, [repo](https://github.com/KazKozDev/auto-vertical-reframe)). Note Ultralytics YOLO is AGPL-3.0 (fine self-hosted, viral if distributed).
- **TRACK-vs-LETTERBOX safety switch.** When there is no confident single subject (common on wide drone shots), letterbox the full wide frame instead of guessing a crop, never zoom in. This is the AutoFlip philosophy ([AutoFlip](https://research.google/blog/autoflip-an-open-source-framework-for-intelligent-video-reframing/)) and the practical answer to "just do not use clips where this is a problem", you keep the clip, you just frame it safely.
- Decide TRACK vs LETTERBOX once per shot for stability; pan within a shot only if the subject actually moves.

### #5 (later, if the above is not enough): Late interaction + beat-snap

- **Late-interaction reranking.** ColQwen2 (Apache-2.0, ~2B params, self-hosted) stores one vector per image patch and scores by MaxSim, matching the specific patches a query cares about rather than the whole-frame average, the structural fix for compositional look-alikes. pgvector cannot do MaxSim, but the **VectorChord** Postgres extension added it in 2025, so you can stay in Postgres ([VectorChord](https://github.com/tensorchord/VectorChord)). Deploy it as a **reranker over the top-N**, not the first-stage index, to cap storage/latency. Honest caveat: the large gains are proven on document-page retrieval, not natural-scene B-roll; high confidence it helps, medium on magnitude. Validate on your clips before committing.
- **Beat-snap the cuts.** Use librosa `beat_track` to nudge the reference's shot boundaries to the nearest musical beat for musicality ([librosa](https://librosa.org/doc/0.10.2/generated/librosa.beat.beat_track.html)). Keep clip selection as the taste stage.

## Recommended target architecture

```
INGEST (Tier 0, immutable, content-addressed, compute once)
  raw clip -> shot detection (TransNetV2) -> per shot: best-moment sub-windows (motion-energy)
           -> multi-frame SigLIP2 / X-CLIP embedding (attention-pooled) per moment
           -> VLM caption (action/energy) -> caption-text embedding
           -> pose + motion-energy scalar
  all keyed by (content_hash, extractor_version); skip on cache hit

SCORE/TAG (Tier 1, mutable, recompute in minutes)
  technical-defect screen (ARNIQA, brightness excluded)
  style-aware VLM quality, percentile-relative -> quality_score (down-weight only)
  versioned rows; REFRESH the retrieval materialized view

RETRIEVE (per beat)
  HyDE: brief -> ideal-shot description
  hybrid: image-vec + caption-text-vec + BM25 + tags, fused by RRF, quality-weighted
  (optional) ColQwen2 MaxSim rerank over top-N

SELECT (the quality center)
  Opus rerank against the creative brief + diversity/freshness  [keep]
  -> human shortlist approval (Increment B)                     [keep, it is a real lever]

RENDER
  subject-aware horizontal crop, zoom pinned 1.0, EMA-smoothed, TRACK-or-LETTERBOX
  beat-snapped cut grid; locked recipe otherwise
```

## Cost and self-hostability

Almost everything top-ranked is open-source, self-hostable, and cheap to run. The biggest cost win is #0: you stop paying (in time and API spend) for full re-ingests on every change. Paid video-embedding APIs are a quality ceiling to benchmark against, not something to buy, your self-hosting preference holds. The only new runtime cost is modest: one extra embedding call per moment (caption lane) and a slightly heavier multi-frame embedding at ingest, both one-time and cached.

## Honest confidence notes

- Strongly supported: selection/rerank and narrative-first are where quality lives; generic aesthetic scorers punish dark footage; single-frame CLIP cannot do verbs; the decouple-and-cache ingestion pattern.
- Medium / verify: multi-frame pooling vs true video embedding (benchmark is off-domain, test on your clips); late-interaction magnitude on B-roll; several model licenses (V-JEPA 2, DOVER S-Lab, Q-Align base chain) before any commercial use; vendor benchmark numbers are directional.
- The single highest-value next experiment is cheap once #0 lands: A/B the caption-text lane + motion-energy + multi-frame embedding on your real footage and watch the rerank picks.

## Sources

[ZeroEntropy reranking guide](https://zeroentropy.dev/articles/ultimate-guide-to-choosing-the-best-reranking-model-in-2025/) · [Cascaded embedding-reranking (video)](https://arxiv.org/pdf/2512.12935) · [Human-inspired video editing](https://arxiv.org/html/2507.02790v1) · [CLIP verb weakness](https://arxiv.org/pdf/2508.04945) · [Captions for semantic retrieval](https://arxiv.org/pdf/2409.13860) · [HyDE](https://zilliz.com/learn/improve-rag-and-information-retrieval-with-hyde-hypothetical-document-embeddings) · [Hybrid fusion / RRF](https://mbrenndoerfer.com/writing/hybrid-search-bm25-dense-retrieval-fusion) · [Mixpeek video embedding benchmark](https://mixpeek.com/blog/video-embedding-benchmark-2026) · [SigLIP 2](https://arxiv.org/pdf/2502.14786) · [X-CLIP](https://arxiv.org/abs/2208.02816) · [V-JEPA 2](https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks/) · [VectorChord MaxSim in Postgres](https://github.com/tensorchord/VectorChord) · [Late-interaction overview](https://weaviate.io/blog/late-interaction-overview) · [TransNetV2 / AutoShot](https://ar5iv.labs.arxiv.org/html/2304.06116) · [DOVER](https://github.com/VQAssessment/DOVER) · [ARNIQA](https://github.com/miccunifi/ARNIQA) · [LAION-Aesthetics bias audit](https://arxiv.org/abs/2601.09896) · [KazKozDev auto-vertical-reframe](https://github.com/KazKozDev/auto-vertical-reframe) · [Google AutoFlip](https://research.google/blog/autoflip-an-open-source-framework-for-intelligent-video-reframing/) · [librosa beat tracking](https://librosa.org/doc/0.10.2/generated/librosa.beat.beat_track.html) · [Unstructured incremental ingestion](https://unstructured.io/insights/incremental-data-ingestion-strategies-for-continuous-pipelines) · [DVC pipelines](https://doc.dvc.org/user-guide/pipelines) · [Deja Vu (embedding/frame caching)](https://arxiv.org/pdf/2506.14107) · [pgvector versioned embeddings](https://www.postgresql.fastware.com/blog/how-to-store-and-query-embeddings-in-postgresql-without-losing-your-mind) · [Qdrant embedding migration](https://qdrant.tech/documentation/tutorials-operations/embedding-model-migration/) · [Production B-roll retrieval (CVPR 2025)](https://openaccess.thecvf.com/content/CVPR2025W/IViSE/papers/Schoeffmann_AI-based_Video_Content_Understanding_for_Automatic_and_Interactive_Multimedia_Retrieval_CVPRW_2025_paper.pdf)
