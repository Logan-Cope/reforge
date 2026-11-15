"""One-line IG captions for finished reels, generated from the deconstruct brief.

Runs LOCALLY (backfill script + a guarded hook in the render worker) - the deployed
app never calls an LLM; it just reads ``reels.caption``. The schedule form pre-fills
from that column and Logan edits or deletes freely.

House style (docs/PLAN.md ground rules): Logan's voice, short, no CTA, no hashtags,
no em dashes in published text.
"""
from __future__ import annotations

import os

MAX_LEN = 220  # IG allows 2200; a good caption here is one short line


def caption_prompt(brief: dict | None, reference_key: str) -> str:
    """The Haiku prompt for one caption (pure; unit-tested)."""
    b = brief or {}
    ref = os.path.splitext(os.path.basename(reference_key or ""))[0]
    context = "\n".join(f"- {k}: {v}" for k, v in b.items()
                        if k in ("tone", "vibe", "theme", "emotional_register") and v)
    return f"""Write ONE Instagram caption for a fitness reel.

Reel context (from the reference "{ref}"):
{context or '- (no brief available; it is a fitness B-roll reel)'}

Rules:
- one line, first person, under 120 characters
- confident and natural, like a fit Christian engineer in his 20s talking, not a brand
- no hashtags, no call to action, no emojis, no em dashes
- do not mention "reel", "video", or the camera

Reply with the caption text only."""


def parse_caption(text: str | None) -> str:
    """Clean a model reply to the caption itself (pure; unit-tested).

    First line only, surrounding quotes and a leading "Caption:" label stripped,
    em dashes removed (house rule), length capped.
    """
    if not text:
        return ""
    line = text.strip().splitlines()[0].strip()
    if line.lower().startswith("caption:"):
        line = line[len("caption:"):].strip()
    line = line.strip('"“”').strip()
    line = line.replace(" - ", "-").replace(" - ", " ")  # no em dashes, ever
    return line[:MAX_LEN]


def generate_caption(brief: dict | None, reference_key: str) -> str:
    """One Haiku call -> a cleaned caption ('' on any failure; callers treat empty
    as 'leave the column null and move on' - never block a render on a caption)."""
    try:
        import anthropic

        from reforge.config import settings

        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        msg = client.messages.create(
            model=settings.claude_model, max_tokens=100,
            messages=[{"role": "user",
                       "content": caption_prompt(brief, reference_key)}])
        return parse_caption(msg.content[0].text)
    except Exception:
        return ""


def backfill_captions(conn, limit: int = 50) -> int:
    """Fill ``reels.caption`` where missing (oldest first); returns how many were set.

    Safe to call from the worker loop: each reel is one cheap Haiku call, failures
    leave the row null for the next pass.
    """
    rows = conn.execute("""
        SELECT id, brief, reference_key FROM reels
        WHERE caption IS NULL ORDER BY id LIMIT %s
    """, (limit,)).fetchall()
    done = 0
    for reel_id, brief, reference_key in rows:
        cap = generate_caption(brief, reference_key or "")
        if cap:
            conn.execute("UPDATE reels SET caption = %s WHERE id = %s", (cap, reel_id))
            done += 1
    conn.commit()
    return done
