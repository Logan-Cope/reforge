# Reforge web app (FastAPI + the retrieval/judge pipeline) for Render.
# ffmpeg: reference deconstruct + preview renders. torch CPU: query-time CLIP embedding.
# Full-res final renders stay local (the 84GB originals never ship); the app serves
# review/curation and preview renders from proxies + thumbnails in cloud storage.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch keeps the image ~4GB smaller than the default CUDA wheels
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir open-clip-torch

COPY pyproject.toml README.md ./
COPY src ./src
COPY scripts ./scripts
COPY db ./db
COPY taste/rubric.md ./taste/rubric.md
RUN pip install --no-cache-dir -e .

# Pre-download CLIP weights at build time so cold starts don't pull ~1.7GB.
# NOTE: ViT-L-14 needs ~2.5-3GB RAM resident - use a Render instance >= 4GB,
# or run demo mode with precomputed reference embeddings (no CLIP load).
ARG PRELOAD_CLIP=1
RUN if [ "$PRELOAD_CLIP" = "1" ]; then \
    python -c "import open_clip; open_clip.create_model_and_transforms('ViT-L-14', pretrained='laion2b_s32b_b82k')"; \
    fi

ENV PYTHONPATH=/app/src PORT=8000
EXPOSE 8000
CMD ["python", "scripts/serve_app.py"]
