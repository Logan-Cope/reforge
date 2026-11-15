"""Retrieval eval harness: score `match.retrieve` against the labeled taste log.

Each `LabeledBeat` names a reference run; we deconstruct that reference once, take the
beat's shot + representative frame, retrieve candidates, and score them against the
storage keys Logan judged good (hit@k / precision@k / recall@k / mrr).

CAVEAT: taste-log beat indices come from the hand-run chat judge, whose cut grid may
differ slightly from a fresh deconstruct's grid - rows are DIRECTIONAL, not exact.

`reference_for_run` and `retrieval_rows` are pure (dependencies injected, unit-tested);
`run_retrieval_eval` is the I/O wrapper (DB, vision model, ffmpeg deconstruct).
"""
from __future__ import annotations

import datetime
import json
import os
import re

from .dataset import LabeledBeat, resolve_fragment
from .metrics import format_results_table, hit_at_k, mrr, precision_at_k, recall_at_k

# Taste-log run prefixes -> the reference reel they were judged against (files in reference/).
REFERENCE_MAP: dict[str, str] = {
    "DISCIPLINE": "reference/DISCIPLINE.mp4",
    "grace": "reference/grace.mp4",
    "Bible-gym": "reference/Bible-gym.mp4",
}


def reference_for_run(run: str, mapping: dict[str, str] | None = None) -> str | None:
    """The reference video for a taste-log run name (pure).

    Longest case-insensitive prefix match over ``mapping`` (default REFERENCE_MAP);
    None when no prefix matches.
    """
    mapping = REFERENCE_MAP if mapping is None else mapping
    run_l = (run or "").lower()
    matches = [p for p in mapping if p and run_l.startswith(p.lower())]
    return mapping[max(matches, key=len)] if matches else None


def retrieval_rows(beats: list[LabeledBeat], decon_by_ref: dict, retrieve_fn,
                   list_keys_fn, k_values: tuple[int, ...] = (5, 10, 20)) -> list[dict]:
    """Score retrieval for each labeled beat (pure-testable core; deps injected).

    ``decon_by_ref`` maps reference paths (see ``reference_for_run``) to Deconstructions;
    beats without one are skipped. Out-of-range beat indices emit a row with only a
    ``warning`` field. ``good_not_retrieved`` notes resolved good keys entirely absent
    from the retrieved list.
    """
    storage_keys = list_keys_fn()
    rows: list[dict] = []
    for beat in beats:
        ref = reference_for_run(beat.run)
        decon = decon_by_ref.get(ref) if ref else None
        if decon is None:
            continue
        if beat.beat >= len(decon.shots):
            rows.append({"run": beat.run, "beat": beat.beat,
                         "warning": f"beat {beat.beat} out of range "
                                    f"({len(decon.shots)} shots in fresh deconstruct)"})
            continue

        shot = decon.shots[beat.beat]
        ref_frame = decon.frames_per_shot[beat.beat][0]
        candidates = retrieve_fn(shot, ref_frame, k=max(k_values))
        retrieved = [c.clip_key for c in candidates]

        resolved_all = [resolve_fragment(f, storage_keys) for f in beat.good + beat.bad]
        relevant = {resolve_fragment(f, storage_keys) for f in beat.good} - {None}

        row: dict = {"run": beat.run, "beat": beat.beat}
        for k in k_values:
            row[f"hit@{k}"] = hit_at_k(retrieved, relevant, k)
            row[f"precision@{k}"] = precision_at_k(retrieved, relevant, k)
            row[f"recall@{k}"] = recall_at_k(retrieved, relevant, k)
        row["mrr"] = mrr(retrieved, relevant)
        row["resolved_fragments"] = sum(1 for key in resolved_all if key is not None)
        row["unresolved_fragments"] = sum(1 for key in resolved_all if key is None)
        missing = sorted(relevant - set(retrieved))
        if missing:
            row["good_not_retrieved"] = missing  # good key never surfaced at any rank
        rows.append(row)
    return rows


def _list_storage_keys() -> list[str]:
    """All clip storage keys, read once from the DB (I/O seam)."""
    from ..db.client import get_conn

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("select storage_key from clips")
            return [row[0] for row in cur.fetchall()]
    finally:
        conn.close()


def run_retrieval_eval(work_dir: str = "data/eval_work",
                       retriever: str | None = None) -> list[dict]:
    """Deconstruct each referenced reel once, then score retrieval per labeled beat.

    I/O wrapper around ``retrieval_rows``: loads the taste log, deconstructs each
    needed reference with the configured captioner, reads storage keys from the DB,
    and retrieves with the real ``match.retrieve``. ``retriever`` ("clip" | "hybrid",
    Phase 9B A/B) is passed through; None = the RETRIEVER env default. See module
    caveat: taste-log beat indices may not line up exactly with a fresh deconstruct's
    cut grid.
    """
    from ..deconstruct.reference import deconstruct
    from ..llm import get_captioner
    from ..match.retrieve import retrieve
    from .dataset import load_taste_log

    beats = load_taste_log()
    refs = sorted({r for r in (reference_for_run(b.run) for b in beats) if r})
    vision = get_captioner()
    decon_by_ref = {}
    for ref in refs:
        name = re.sub(r"\W+", "_", os.path.splitext(os.path.basename(ref))[0])
        decon_by_ref[ref] = deconstruct(ref, vision, os.path.join(work_dir, name))

    storage_keys = _list_storage_keys()
    return retrieval_rows(beats, decon_by_ref,
                          lambda shot, ref_frame, k: retrieve(shot, ref_frame, k=k,
                                                              retriever=retriever),
                          lambda: storage_keys)


# --- synthetic labeled set (eval/synthetic.py -> labeled-beats.json) ---------

def shot_from_record(record: dict) -> "object":
    """Rebuild the ShotSpec a labeled beat was judged against from its ``query_meta`` (pure).

    Timing fields are irrelevant to retrieval and set to 0.
    """
    from ..llm.base import ShotSpec

    meta = record.get("query_meta") or {}
    return ShotSpec(
        idx=record.get("beat", 0), start_s=0.0, end_s=0.0,
        description=meta.get("description", ""),
        on_screen_text="",
        energy=meta.get("energy", ""),
        query=meta.get("query", ""),
        role=meta.get("role", ""),
        intended_feeling=meta.get("intended_feeling", ""),
        intended_energy=meta.get("intended_energy", ""),
    )


def brief_from_record(record: dict) -> "object | None":
    """Rebuild the ReelBrief stored with a labeled beat, or None when absent (pure)."""
    from ..llm.base import ReelBrief

    b = record.get("brief")
    if not b:
        return None
    return ReelBrief(vibe=b.get("vibe", ""), emotional_register=b.get("emotional_register", ""),
                     energy_profile=b.get("energy_profile", ""), tone=b.get("tone", ""),
                     theme=b.get("theme", ""))


def labeled_rows(labels: list[dict], decon_by_ref: dict, retrieve_fn,
                 k_values: tuple[int, ...] = (5, 10, 20)) -> list[dict]:
    """Score retrieval for each synthetic labeled beat (pure-testable core; deps injected).

    ``labels`` are ``labeled-beats.json`` records; relevance is compared on
    ``ClipMoment.moment_id`` against ``relevant_moment_ids`` (explicit ids - no
    fragment resolution). ``retrieve_fn(shot, ref_frame, brief, k)`` returns candidate
    ClipMoments. Beats whose reference has no deconstruction are skipped; beats past
    the fresh cut grid emit a warning-only row.
    """
    rows: list[dict] = []
    for record in labels:
        ref = reference_for_run(record.get("run", ""))
        decon = decon_by_ref.get(ref) if ref else None
        if decon is None:
            continue
        beat = record["beat"]
        if beat >= len(decon.frames_per_shot):
            rows.append({"run": record["run"], "beat": beat,
                         "warning": f"beat {beat} out of range "
                                    f"({len(decon.frames_per_shot)} shots in fresh deconstruct)"})
            continue

        shot = shot_from_record(record)
        ref_frame = decon.frames_per_shot[beat][0]
        candidates = retrieve_fn(shot, ref_frame, brief_from_record(record), max(k_values))
        retrieved = [c.moment_id for c in candidates]
        relevant = set(record.get("relevant_moment_ids") or [])

        row: dict = {"run": record["run"], "beat": beat}
        for k in k_values:
            row[f"hit@{k}"] = hit_at_k(retrieved, relevant, k)
            row[f"precision@{k}"] = precision_at_k(retrieved, relevant, k)
            row[f"recall@{k}"] = recall_at_k(retrieved, relevant, k)
        row["mrr"] = mrr(retrieved, relevant)
        row["n_relevant"] = len(relevant)
        rows.append(row)
    return rows


def summary_row(rows: list[dict], k_values: tuple[int, ...] = (5, 10, 20)) -> dict:
    """Mean of each metric over the scored rows (warning-only rows excluded), pure.

    ``beat`` carries the scored-row count; bool hits average to a hit RATE.
    """
    metric_keys = [f"{m}@{k}" for k in k_values for m in ("hit", "precision", "recall")] + ["mrr"]
    scored = [r for r in rows if "mrr" in r]
    summary: dict = {"run": "MEAN", "beat": len(scored)}
    for key in metric_keys:
        values = [float(r[key]) for r in scored if key in r]
        summary[key] = sum(values) / len(values) if values else 0.0
    return summary


def _frames_only_decon(ref_path: str, work_dir: str):
    """Cut grid + frames for a reference, NO vision call (I/O: ffmpeg only).

    The labeled set stores each beat's query_meta + brief, so the from-labels eval only
    needs the deterministic parts of deconstruct: the grid (beat alignment) and the
    representative frames (the CLIP image lane).
    """
    from types import SimpleNamespace

    from ..deconstruct.reference import cut_grid, extract_frames

    name = re.sub(r"\W+", "_", os.path.splitext(os.path.basename(ref_path))[0])
    grid = cut_grid(ref_path)
    frames = extract_frames(ref_path, grid, os.path.join(work_dir, name, "frames"))
    return SimpleNamespace(frames_per_shot=frames)


def run_retrieval_eval_from_labels(labels_path: str, retriever: str | None = None,
                                   k_values: tuple[int, ...] = (5, 10, 20),
                                   work_dir: str = "data/eval_work",
                                   decon_by_ref: dict | None = None) -> list[dict]:
    """Score one retriever against the synthetic labeled set + append the MEAN summary row.

    I/O wrapper around ``labeled_rows``: reads ``labels_path`` (see
    ``synthetic.labeled_beat_record``), re-derives each referenced reel's frames once
    (deterministic ffmpeg; queries come from the stored query_meta/brief so both A/B
    arms score against the EXACT asks that were labeled), and retrieves with the real
    ``match.retrieve``. Pass a shared ``decon_by_ref`` dict to reuse frames across
    retriever runs.
    """
    from ..match.retrieve import retrieve

    with open(labels_path, encoding="utf-8") as fh:
        labels = json.load(fh)

    decon_by_ref = {} if decon_by_ref is None else decon_by_ref
    refs = sorted({r for r in (reference_for_run(rec.get("run", "")) for rec in labels) if r})
    for ref in refs:
        if ref not in decon_by_ref:
            decon_by_ref[ref] = _frames_only_decon(ref, work_dir)

    rows = labeled_rows(
        labels, decon_by_ref,
        lambda shot, ref_frame, brief, k: retrieve(shot, ref_frame, k=k,
                                                   retriever=retriever, brief=brief),
        k_values)
    rows.append(summary_row(rows, k_values))
    return rows


def _table_columns(rows: list[dict]) -> list[str]:
    """Identifier columns first, then every key holding a numeric/bool value in any row."""
    columns = [c for c in ("run", "beat") if any(c in r for r in rows)]
    for row in rows:
        for key, value in row.items():
            if key not in columns and isinstance(value, (int, float, bool)):
                columns.append(key)
    return columns


def save_results(rows: list[dict], name: str, out_dir: str = "eval_results") -> tuple[str, str]:
    """Write ``<out_dir>/<YYYY-MM-DD>-<name>.json`` (raw rows) + ``.md`` (metrics table)."""
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, f"{datetime.date.today().isoformat()}-{name}")
    json_path, md_path = base + ".json", base + ".md"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2)
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(format_results_table(rows, _table_columns(rows)) + "\n")
    return json_path, md_path
