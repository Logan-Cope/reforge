"""Taste-log dataset: parse Logan's per-beat judgments into labeled examples.

``taste/taste-log.jsonl`` records human review of clip picks per reel beat. Only
per-beat lines with resolvable clip references become ``LabeledBeat``s; summary,
multi-beat note, and note-only lines carry no per-clip data and are skipped.
Parsing is pure; ``load_taste_log`` is the thin file-reading seam, and mapping
fragments to DB ``storage_key``s stays in the pure ``resolve_fragment``.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

# "clip244 Jogging/dji_184442 (drone run)" -> optional clipNNN prefix, fragment
# (may contain spaces), optional trailing " (...)" parenthetical.
_CLIP_REF = re.compile(r"^\s*(?:clip\d+\s+)?(?P<frag>.+?)(?:\s+\(.*\))?\s*$")


@dataclass
class LabeledBeat:
    """One beat's human judgment: which clip fragments were good/bad picks."""

    run: str
    beat: int
    good: list[str] = field(default_factory=list)   # path fragments judged good
    bad: list[str] = field(default_factory=list)    # path fragments judged bad
    note: str = ""


def parse_clip_ref(text: str) -> str | None:
    """Extract the path fragment from a clip reference string.

    "clip244 Jogging/dji_184442 (drone run)" -> "Jogging/dji_184442". Handles refs
    without a parenthetical; returns None when there is no path-like token ("/").
    """
    m = _CLIP_REF.match(text or "")
    if not m:
        return None
    frag = m.group("frag")
    return frag if "/" in frag else None


def parse_taste_log(lines: list[str]) -> list[LabeledBeat]:
    """Parse taste-log JSONL lines into LabeledBeats (pure; never raises).

    Per-beat swap lines yield bad=[judge_pick], good=[swapped_to]; "keep" verdicts
    yield good=[judge_pick]; rejection lines yield bad=[judge_pick]. Lines without
    per-clip data (review_summary, multi-beat notes, note-only) and malformed JSON
    are skipped.
    """
    beats: list[LabeledBeat] = []
    for line in lines:
        try:
            entry = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(entry, dict) or not isinstance(entry.get("beat"), int):
            continue  # review_summary, multi-beat "beats" lines, or junk
        pick = parse_clip_ref(entry.get("judge_pick", ""))
        swapped = parse_clip_ref(entry.get("swapped_to", ""))
        good = [swapped] if swapped else []
        bad: list[str] = []
        if pick:
            (good if entry.get("verdict") == "keep" else bad).append(pick)
        if not good and not bad:
            continue  # note-only line without resolvable clips
        beats.append(LabeledBeat(run=entry.get("run", ""), beat=entry["beat"],
                                 good=good, bad=bad, note=entry.get("logan_note", "")))
    return beats


def load_taste_log(path: str | None = None) -> list[LabeledBeat]:
    """Read and parse the taste log (I/O seam around ``parse_taste_log``)."""
    if path is None:
        from ..config import settings

        path = os.path.join(getattr(settings, "taste_dir", "taste"), "taste-log.jsonl")
    with open(path, encoding="utf-8") as f:
        return parse_taste_log(f.readlines())


def resolve_fragment(fragment: str, storage_keys: list[str]) -> str | None:
    """The storage_key matching ``fragment`` (case-insensitive), pure.

    Whole-substring match first; when that fails, fall back to requiring every token of
    the fragment (split on /, _, spaces; len >= 3) to appear in the key - the taste log
    abbreviates long filenames (e.g. "dji_184442" for "dji_mimo_20251007_184442_...").
    None when nothing matches. When several match, "most specific" isn't definable, so
    return the shortest key (ties broken lexicographically) for determinism.
    """
    needle = fragment.lower()
    matches = [k for k in storage_keys if needle in k.lower()]
    if not matches:
        tokens = [t for t in re.split(r"[/_\s]+", needle) if len(t) >= 3]
        if tokens:
            matches = [k for k in storage_keys if all(t in k.lower() for t in tokens)]
    return min(matches, key=lambda k: (len(k), k)) if matches else None
