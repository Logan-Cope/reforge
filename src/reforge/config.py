"""Central settings, loaded from environment (.env). Dependency-light on purpose."""
from __future__ import annotations
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class Settings:
    supabase_url: str = os.getenv("SUPABASE_URL", "")
    supabase_key: str = os.getenv("SUPABASE_SERVICE_KEY", "")
    database_url: str = os.getenv("DATABASE_URL", "")
    storage_backend: str = os.getenv("STORAGE_BACKEND", "local")
    local_library_dir: str = os.getenv("LOCAL_LIBRARY_DIR", "./library")
    local_data_dir: str = os.getenv("LOCAL_DATA_DIR", "./data")
    output_dir: str = os.getenv("OUTPUT_DIR", "./output")
    r2_account_id: str = os.getenv("R2_ACCOUNT_ID", "")
    r2_access_key_id: str = os.getenv("R2_ACCESS_KEY_ID", "")
    r2_secret_access_key: str = os.getenv("R2_SECRET_ACCESS_KEY", "")
    r2_bucket: str = os.getenv("R2_BUCKET", "reforge-clips")
    captioner: str = os.getenv("CAPTIONER", "claude")  # claude (primary) | gemini (benchmark)
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    # LLM gateway (Vercel AI Gateway): set AI_GATEWAY_API_KEY to route Claude calls through it;
    # unset = direct Anthropic. Spike-verified lossless (vision, forced tools, usage counts).
    ai_gateway_api_key: str = os.getenv("AI_GATEWAY_API_KEY", "")
    ai_gateway_base_url: str = os.getenv("AI_GATEWAY_BASE_URL", "https://ai-gateway.vercel.sh")
    claude_model: str = os.getenv("CLAUDE_MODEL", "claude-haiku-4-5")
    rerank_model: str = os.getenv("RERANK_MODEL", "claude-opus-4-8")  # taste step (Phase 3.5): Opus
    identity_ref_dir: str = os.getenv("IDENTITY_REF_DIR", "identity_reference")
    taste_dir: str = os.getenv("TASTE_DIR", "taste")
    # Phase 3.6 variety knobs
    diversity_location_cap: int = int(os.getenv("DIVERSITY_LOCATION_CAP", "2"))  # max clips per location/reel
    freshness_weight: float = float(os.getenv("FRESHNESS_WEIGHT", "0.1"))         # cross-reel rotation; 0 = always-best
    # Phase 3.7 quality knobs
    pose_model_path: str = os.getenv("POSE_MODEL_PATH", "models/pose_landmarker_lite.task")  # mediapipe subject detector
    quality_weight: float = float(os.getenv("QUALITY_WEIGHT", "0.15"))            # retrieval down-weight of low quality
    # Phase 3.8 holistic gate: max review-and-repick rounds before accepting the cut
    gate_max_iters: int = int(os.getenv("GATE_MAX_ITERS", "2"))
    # Phase 9B retriever: "hybrid" (CLIP image + caption-text HyDE + BM25, RRF-fused)
    # or "clip" (the original CLIP-only path, kept as the A/B regression baseline).
    retriever: str = os.getenv("RETRIEVER", "hybrid")
    clip_model: str = os.getenv("CLIP_MODEL", "ViT-L-14")
    clip_pretrained: str = os.getenv("CLIP_PRETRAINED", "laion2b_s32b_b82k")
    text_embed_model: str = os.getenv("TEXT_EMBED_MODEL", "text-embedding-3-small")
    twelvelabs_api_key: str = os.getenv("TWELVELABS_API_KEY", "")
    # Trend-context web search (acquire/trends.py); unset = skip enrichment
    tavily_api_key: str = os.getenv("TAVILY_API_KEY", "")
    # Reference acquisition (acquire/download.py): yt-dlp browser cookies for IG at volume;
    # empty = no cookies (works for many public reels).
    ytdlp_cookies_from_browser: str = os.getenv("YTDLP_COOKIES_FROM_BROWSER", "")

settings = Settings()
