"""End-to-end LLM-as-judge gate: "would Logan post this render as-is?"

The judge is deliberately INDEPENDENT of the pipeline's VisionModel - it builds its
own `anthropic` client (via `_client_config`, so AI Gateway routing is honored) so a
pipeline regression can't silently bias its own grader. It sees GOLD reference reels
(the accepted quality bar) then the candidate's frames in time order, and records a
strict verdict through a forced tool call.

`_would_post_instruction` and `_parse_would_post` are pure (unit-tested); frame
extraction and the anthropic SDK live behind `contact_frames` / `judge_render`.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass, field

DEFAULT_JUDGE_MODEL = "claude-opus-4-8"


@dataclass
class WouldPostVerdict:
    """The judge's read on one rendered reel."""

    score: int                                  # 0-100 vs the gold bar
    would_post: bool                            # would Logan post this as-is?
    weak_beats: list[int] = field(default_factory=list)  # 0-based frame/beat indices
    notes: str = ""


_WOULD_POST_TOOL = {
    "name": "record_would_post",
    "description": "Record the would-post verdict for the candidate reel.",
    "input_schema": {
        "type": "object",
        "properties": {
            "score": {"type": "integer", "description": "0-100, closeness to the gold bar"},
            "would_post": {"type": "boolean", "description": "would Logan post this as-is"},
            "weak_beats": {"type": "array", "items": {"type": "integer"},
                           "description": "0-based indices of candidate frames failing the rubric"},
            "notes": {"type": "string", "description": "two sentences max"},
        },
        "required": ["score", "would_post", "weak_beats", "notes"],
        "additionalProperties": False,
    },
}


def _would_post_instruction(rubric: str) -> str:
    """Judge instruction text (pure); prepends Logan's taste rubric when present."""
    base = (
        "You are a HARSH short-form video curator applying Logan's taste standards. You will "
        "see GOLD REFERENCE reels first - these define the accepted quality bar - then a "
        "CANDIDATE reel's frames in time order. Record ONE verdict via record_would_post:\n"
        "- score: 0-100, how close the candidate is to the gold bar.\n"
        "- would_post: would Logan post this reel as-is?\n"
        "- weak_beats: 0-based indices (in frame order) of candidate frames/beats that fail "
        "the rubric - unreadable subject (cannot tell who it is or what he is doing at a "
        "glance on a phone), subject out of frame, setting-up-not-training, grainy/dirty "
        "footage, or near-duplicate looks.\n"
        "- notes: two sentences max.\n"
        "STYLE, not defects: this creator shoots intentionally dark, low-key, moody, "
        "handheld, sometimes back-to-camera - the gold reels show exactly this look. "
        "Darkness, night settings, silhouettes, and moody grading are the STYLE and must "
        "NOT be penalized; a dark beat fails ONLY if the subject is genuinely unreadable. "
        "Judge each beat the way the gold reels set the bar, not against conventional "
        "bright-and-sharp standards.\n"
        "Each candidate frame is ONE beat of the reel (its mid-beat frame). Be strict but "
        "proportionate: the gold reels themselves score 85-100; a reel where more than a "
        "fifth of its beats are weak cannot exceed 70."
    )
    if not rubric:
        return base
    return "LOGAN'S TASTE RUBRIC - apply every test below:\n" + rubric + "\n\n" + base


def _parse_would_post(result: dict) -> WouldPostVerdict:
    """Tolerant parse of the tool-call payload into a verdict (pure).

    score is clamped to 0-100; a missing would_post is inferred as score >= 80;
    non-integer weak_beats entries are coerced when possible and dropped otherwise.
    """
    try:
        score = int(result.get("score", 0))
    except (TypeError, ValueError):
        score = 0
    score = min(100, max(0, score))

    would_post = result.get("would_post")
    if not isinstance(would_post, bool):
        would_post = score >= 80

    weak_beats: list[int] = []
    for beat in result.get("weak_beats") or []:
        if isinstance(beat, bool):
            continue
        if isinstance(beat, int) or (isinstance(beat, float) and beat.is_integer()):
            weak_beats.append(int(beat))

    return WouldPostVerdict(score=score, would_post=would_post,
                            weak_beats=weak_beats, notes=str(result.get("notes") or "").strip())


def contact_frames(video_path: str, n: int, tmp_dir: str) -> list[str]:
    """One JPEG per detected beat (mid-beat), capped at ``n``; even sampling as fallback.

    Mid-beat frames are what a viewer actually reads: evenly-spaced timestamps land on
    cuts and mid-motion blur, and oversample long beats into phantom "near-duplicates" -
    both of which made the judge mis-score real reels during calibration.
    """
    from ..deconstruct.reference import cut_grid
    from ..ingest.frames import extract_frame, sample_timestamps
    from ..ingest.probe import probe_media

    try:
        grid = cut_grid(video_path)
    except Exception:
        grid = []
    if len(grid) >= 2:
        times = [(start + end) / 2 for start, end in grid][:n]
    else:
        times = sample_timestamps(probe_media(video_path)["duration_s"], n=n)
    return [extract_frame(video_path, t, os.path.join(tmp_dir, f"{i:02d}.jpg"))
            for i, t in enumerate(times)]


def _frame_block(path: str) -> dict:
    from ..llm.claude import _frame_block as claude_frame_block

    return claude_frame_block(path)  # downscaled to the API's 1568px cap (payload size)


def judge_render(video_path: str, gold_paths: list[str], *, n_frames: int = 16,
                 gold_frames_each: int = 6, model: str | None = None) -> WouldPostVerdict:
    """Judge one rendered reel against the gold references; return the verdict.

    Builds its own anthropic client (gateway-aware via ``_client_config``) with model
    ``EVAL_JUDGE_MODEL`` (default Opus) - independent of the pipeline's VisionModel.
    At most 2 gold reels are sent (token cost); frames go to a temp dir, cleaned up.
    """
    import anthropic

    from ..config import settings
    from ..llm.claude import _client_config, _maybe_trace
    from ..taste import load_rubric

    api_key, base_url = _client_config(
        settings.anthropic_api_key, settings.ai_gateway_api_key, settings.ai_gateway_base_url)
    client = _maybe_trace(
        anthropic.Anthropic(api_key=api_key, base_url=base_url, timeout=180.0, max_retries=3))
    model = model or os.getenv("EVAL_JUDGE_MODEL", DEFAULT_JUDGE_MODEL)

    tmp_dir = tempfile.mkdtemp(prefix="reforge_wouldpost_")
    try:
        content: list[dict] = [{"type": "text", "text": _would_post_instruction(load_rubric())}]
        for gi, gold in enumerate(gold_paths[:2]):
            gold_dir = os.path.join(tmp_dir, f"gold{gi}")
            os.makedirs(gold_dir, exist_ok=True)
            content.append({"type": "text", "text": f"GOLD REFERENCE {gi}:"})
            content += [_frame_block(p) for p in contact_frames(gold, gold_frames_each, gold_dir)]
        cand_dir = os.path.join(tmp_dir, "candidate")
        os.makedirs(cand_dir, exist_ok=True)
        content.append({"type": "text", "text": "CANDIDATE (frames in time order):"})
        content += [_frame_block(p) for p in contact_frames(video_path, n_frames, cand_dir)]

        response = client.messages.create(
            model=model,
            max_tokens=1024,
            tools=[_WOULD_POST_TOOL],
            tool_choice={"type": "tool", "name": "record_would_post"},
            messages=[{"role": "user", "content": content}],
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    result = next(b.input for b in response.content if b.type == "tool_use")
    return _parse_would_post(result)
