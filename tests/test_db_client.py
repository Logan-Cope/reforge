"""Unit tests for the pure query-building logic in db/client.py.

These touch no database: only the kind->table/dim mapping and the KNN filter
clause builder, which are factored out so they can be tested in isolation.
"""
import pytest

from reforge.db.client import bm25_rank_to_distance, build_filter_clause, table_for_kind


# --- kind -> (table, dim) mapping -------------------------------------------

def test_table_for_kind_maps_each_kind_to_its_table_and_dim():
    assert table_for_kind("clip") == ("moment_clip_embeddings", 768)
    # Phase 9B: caption lane stores OpenAI text-embedding-3-small at dimensions=768
    # (migration 0007 altered the table from the schema's placeholder 1536).
    assert table_for_kind("caption") == ("moment_caption_embeddings", 768)
    assert table_for_kind("marengo") == ("moment_marengo_embeddings", 1024)


# --- Phase 9B: BM25 rank -> distance-like score ------------------------------

def test_bm25_rank_to_distance_inverts_rank_ordering():
    # ts_rank is DESC (bigger = better); distance must be ASC (smaller = better).
    assert bm25_rank_to_distance(2.0) < bm25_rank_to_distance(0.5)


def test_bm25_rank_to_distance_stays_in_unit_range():
    assert bm25_rank_to_distance(0.0) == 1.0        # no match strength -> farthest
    assert 0.0 < bm25_rank_to_distance(100.0) < 1.0  # never negative or zero


def test_table_for_kind_rejects_unknown_kind():
    with pytest.raises(ValueError):
        table_for_kind("bogus")


# --- KNN filter clause builder ----------------------------------------------

def test_filter_clause_empty_when_no_filters():
    assert build_filter_clause(None) == ("", {})
    assert build_filter_clause({}) == ("", {})


def test_filter_clause_is_logan():
    sql, params = build_filter_clause({"is_logan": True})
    assert sql == " AND m.is_logan = %(f_is_logan)s"
    assert params == {"f_is_logan": True}


def test_filter_clause_location():
    sql, params = build_filter_clause({"location": "gym"})
    assert sql == " AND m.location = %(f_location)s"
    assert params == {"f_location": "gym"}


def test_filter_clause_combines_filters():
    sql, params = build_filter_clause({"is_logan": True, "location": "gym"})
    assert "m.is_logan = %(f_is_logan)s" in sql
    assert "m.location = %(f_location)s" in sql
    assert params == {"f_is_logan": True, "f_location": "gym"}


def test_filter_clause_skips_none_values():
    # A sparse filter dict should drop keys whose value is None.
    sql, params = build_filter_clause({"is_logan": None, "location": "dock"})
    assert "is_logan" not in sql
    assert params == {"f_location": "dock"}


def test_filter_clause_keeps_is_logan_false():
    # False is a meaningful filter value and must not be treated like None.
    sql, params = build_filter_clause({"is_logan": False})
    assert sql == " AND m.is_logan = %(f_is_logan)s"
    assert params == {"f_is_logan": False}


def test_filter_clause_ignores_unknown_columns():
    # Only known moment columns are honored; stray keys are not injected.
    sql, params = build_filter_clause({"unknown_col": "x"})
    assert sql == ""
    assert params == {}
