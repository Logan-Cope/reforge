"""Evaluation metrics (pure).

Retrieval (RAGAS-style, the part of RAGAS that maps to this pipeline):
  - precision_at_k / recall_at_k / mrr / hit_at_k: did we surface the right
    candidate clips for a labeled beat?
Assembly (video-specific) metrics - structural_match, framing_quality (encodes the
v1/v2 fully-in-frame lesson), beat_sync, would_post - plus RAGAS faithfulness for
text sub-steps belong to the LLM-judge harness (later task).
"""
from __future__ import annotations


def precision_at_k(retrieved: list, relevant: set, k: int) -> float:
    """Fraction of the top-k retrieved items that are relevant (denominator k).

    0.0 when k <= 0 or nothing was retrieved.
    """
    if k <= 0 or not retrieved:
        return 0.0
    return sum(1 for item in retrieved[:k] if item in relevant) / k


def recall_at_k(retrieved: list, relevant: set, k: int) -> float:
    """Fraction of relevant items found in the top-k.

    Convention: 0.0 when ``relevant`` is empty (nothing to recall counts as a miss,
    keeping aggregates conservative).
    """
    if not relevant:
        return 0.0
    return sum(1 for item in retrieved[:k] if item in relevant) / len(relevant)


def mrr(retrieved: list, relevant: set) -> float:
    """Reciprocal rank of the FIRST relevant retrieved item; 0.0 if none."""
    for i, item in enumerate(retrieved):
        if item in relevant:
            return 1.0 / (i + 1)
    return 0.0


def hit_at_k(retrieved: list, relevant: set, k: int) -> bool:
    """Whether any relevant item appears in the top-k."""
    return any(item in relevant for item in retrieved[:k])


def format_results_table(rows: list[dict], columns: list[str]) -> str:
    """Plain markdown table of result rows; floats formatted %.3f, missing blank."""
    def cell(value) -> str:
        if value is None:
            return ""
        if isinstance(value, float):
            return f"{value:.3f}"
        return str(value)

    lines = ["| " + " | ".join(columns) + " |",
             "| " + " | ".join("---" for _ in columns) + " |"]
    lines += ["| " + " | ".join(cell(row.get(c)) for c in columns) + " |" for row in rows]
    return "\n".join(lines)
