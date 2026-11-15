"""Embeddings: CLIP frame vectors (open_clip ViT-L-14, 768-d) per moment, plus the
Phase 9B caption-text lane (OpenAI text-embedding-3-small at dimensions=768).

`l2_normalize` and `caption_text` are the pure helpers (unit-tested). `clip_embed`
lazily loads open_clip and `openai_text_embed` lazily imports the openai SDK, so
this module imports without torch/openai for the pure tests.
"""
from __future__ import annotations

import numpy as np

from ..config import settings

_model = None
_preprocess = None
_tokenizer = None


def l2_normalize(vec) -> np.ndarray:
    """Return ``vec`` scaled to unit L2 norm as float32 (zero vector returned as-is)."""
    arr = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    return arr if norm == 0.0 else (arr / norm).astype(np.float32)


def clip_model_name() -> str:
    """Stable identifier stored alongside each embedding (model + pretrained tag)."""
    return f"{settings.clip_model}/{settings.clip_pretrained}"


def _load_model():
    """Lazily create + cache the open_clip model and its image preprocessor (CPU)."""
    global _model, _preprocess
    if _model is None:
        import open_clip

        _model, _, _preprocess = open_clip.create_model_and_transforms(
            settings.clip_model, pretrained=settings.clip_pretrained
        )
        _model.eval()
    return _model, _preprocess


def clip_embed(frame_path: str) -> np.ndarray:
    """Embed one frame image into a normalized 768-d CLIP vector."""
    import torch
    from PIL import Image

    model, preprocess = _load_model()
    image = preprocess(Image.open(frame_path).convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        features = model.encode_image(image)
    return l2_normalize(features[0].cpu().numpy())


def clip_text_embed(text: str) -> np.ndarray:
    """Embed a text query into the SAME normalized 768-d CLIP space as `clip_embed`.

    CLIP is cross-modal, so a text query can be knn'd directly against the stored frame
    embeddings (used as the complementary retrieval signal in Phase 3; image is primary).
    """
    import torch

    global _tokenizer
    model, _ = _load_model()
    if _tokenizer is None:
        import open_clip

        _tokenizer = open_clip.get_tokenizer(settings.clip_model)
    tokens = _tokenizer([text])
    with torch.no_grad():
        features = model.encode_text(tokens)
    return l2_normalize(features[0].cpu().numpy())


# --- Phase 9B caption-text lane ----------------------------------------------

def caption_text(row: dict) -> str:
    """Compose the text we embed for a moment's caption lane (pure).

    The description carries the verbs (the action signal single-frame CLIP is blind
    to); energy / location / framing notes add the metadata a beat's intent query
    can match on. Missing fields are skipped; an all-empty row yields "".
    """
    parts: list[str] = []
    desc = str(row.get("description") or "").strip()
    if desc:
        parts.append(desc if desc.endswith(".") else desc + ".")
    for label, key in (("Energy", "energy"), ("Location", "location"),
                       ("Framing", "framing_notes")):
        value = str(row.get(key) or "").strip()
        if value:
            parts.append(f"{label}: {value}.")
    return " ".join(parts)


def openai_text_embed(texts: list[str], model: str | None = None,
                      dimensions: int = 768) -> list[list[float]]:
    """Embed a batch of texts with OpenAI, L2-normalized, in input order.

    Default model = ``settings.text_embed_model`` (text-embedding-3-small). We request
    ``dimensions=768`` (a documented, supported truncation of the native 1536) so the
    caption table matches the CLIP lane's dim and stays half the storage - MTEB loss
    at 768 is marginal. Reads OPENAI_API_KEY from the environment (.env via config).
    """
    from openai import OpenAI

    if not texts:
        return []
    resp = OpenAI().embeddings.create(
        model=model or settings.text_embed_model, input=texts, dimensions=dimensions
    )
    ordered = sorted(resp.data, key=lambda d: d.index)
    return [l2_normalize(d.embedding).tolist() for d in ordered]
