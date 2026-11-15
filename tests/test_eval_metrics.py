"""Unit tests for the pure retrieval metrics in eval/metrics.py."""
from reforge.eval.metrics import (
    format_results_table,
    hit_at_k,
    mrr,
    precision_at_k,
    recall_at_k,
)

RETRIEVED = ["a", "b", "c", "d"]
RELEVANT = {"b", "d", "x"}


# --- precision@k -------------------------------------------------------------

def test_precision_at_k_counts_relevant_in_top_k():
    assert precision_at_k(RETRIEVED, RELEVANT, k=2) == 0.5      # top-2 = a,b -> 1 relevant
    assert precision_at_k(RETRIEVED, RELEVANT, k=4) == 0.5      # b,d of 4


def test_precision_at_k_zero_when_k_nonpositive_or_empty():
    assert precision_at_k(RETRIEVED, RELEVANT, k=0) == 0.0
    assert precision_at_k(RETRIEVED, RELEVANT, k=-1) == 0.0
    assert precision_at_k([], RELEVANT, k=3) == 0.0


# --- recall@k ----------------------------------------------------------------

def test_recall_at_k_fraction_of_relevant_found():
    assert recall_at_k(RETRIEVED, RELEVANT, k=2) == 1 / 3       # only b found
    assert recall_at_k(RETRIEVED, RELEVANT, k=4) == 2 / 3       # b and d


def test_recall_at_k_zero_when_relevant_empty():
    assert recall_at_k(RETRIEVED, set(), k=4) == 0.0


# --- mrr ---------------------------------------------------------------------

def test_mrr_reciprocal_rank_of_first_relevant():
    assert mrr(RETRIEVED, RELEVANT) == 0.5                      # first relevant is b at rank 2
    assert mrr(["b", "a"], RELEVANT) == 1.0


def test_mrr_zero_when_no_relevant_retrieved():
    assert mrr(["a", "c"], RELEVANT) == 0.0
    assert mrr([], RELEVANT) == 0.0


# --- hit@k -------------------------------------------------------------------

def test_hit_at_k():
    assert hit_at_k(RETRIEVED, RELEVANT, k=2) is True
    assert hit_at_k(RETRIEVED, RELEVANT, k=1) is False
    assert hit_at_k([], RELEVANT, k=3) is False


# --- results table -----------------------------------------------------------

def test_format_results_table_markdown():
    rows = [{"metric": "p@5", "clip": 0.5, "n": 3}]
    out = format_results_table(rows, ["metric", "clip", "n"])
    lines = out.splitlines()
    assert lines[0] == "| metric | clip | n |"
    assert lines[1] == "| --- | --- | --- |"
    assert lines[2] == "| p@5 | 0.500 | 3 |"     # floats %.3f, ints untouched


def test_format_results_table_missing_value_blank():
    out = format_results_table([{"a": 1.0}], ["a", "b"])
    assert out.splitlines()[2] == "| 1.000 |  |"
