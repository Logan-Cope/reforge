"""Queries connection reuse + single-round-trip verdict tests (no live DB).

The app is single-user/local, so ``Queries`` keeps ONE cached autocommit connection
(reconnect-once on connection errors) instead of a pool. These tests fake ``get_conn``
to prove: lazy construction, reuse across calls, one reconnect+retry on
OperationalError/InterfaceError, no retry for ordinary query errors, and that each
verdict write (keep/swap/flag) is a single round trip returning the joined slot row.
"""
import threading

import psycopg
import pytest

import reforge.app.queries as q_mod
from reforge.app.queries import _SLOT_KEYS, Queries


class FakeCursor:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows


class FakeConn:
    """Connection double: records executes, optionally fails the first N of them."""

    def __init__(self, rows=(), fail=0, exc=psycopg.OperationalError):
        self.rows = list(rows)
        self.fail = fail
        self.exc = exc
        self.executed = []
        self.autocommit = False
        self.closed = False

    def execute(self, sql, params=()):
        self.executed.append((sql, params))
        if self.fail > 0:
            self.fail -= 1
            raise self.exc("connection dropped")
        return FakeCursor(self.rows)

    def close(self):
        self.closed = True


@pytest.fixture
def conns(monkeypatch):
    """Patch get_conn to hand out FakeConns from a queue; return the created list."""
    created = []
    pending = []

    def fake_get_conn():
        conn = pending.pop(0) if pending else FakeConn(rows=[(1,)])
        created.append(conn)
        return conn

    monkeypatch.setattr(q_mod, "get_conn", fake_get_conn)
    created_pending = (created, pending)
    return created_pending


def test_constructor_is_lazy(monkeypatch):
    def boom():
        raise AssertionError("get_conn called at construction time")

    monkeypatch.setattr(q_mod, "get_conn", boom)
    Queries()  # must not connect


def test_rows_reuses_one_cached_autocommit_connection(conns):
    created, _ = conns
    q = Queries()
    assert q._rows("SELECT 1") == [(1,)]
    assert q._rows("SELECT 1") == [(1,)]
    assert len(created) == 1                       # one conn for both calls
    assert created[0].autocommit is True           # no explicit commit round trip
    assert len(created[0].executed) == 2


def test_rows_reconnects_once_and_retries_on_operational_error(conns):
    created, pending = conns
    pending.append(FakeConn(fail=1))               # first conn dies on execute
    pending.append(FakeConn(rows=[(42,)]))
    q = Queries()
    assert q._rows("SELECT 42") == [(42,)]
    assert len(created) == 2                       # exactly one reconnect
    assert created[0].closed is True               # dead conn discarded


def test_rows_reconnects_on_interface_error(conns):
    created, pending = conns
    pending.append(FakeConn(fail=1, exc=psycopg.InterfaceError))
    pending.append(FakeConn(rows=[(7,)]))
    q = Queries()
    assert q._rows("SELECT 7") == [(7,)]
    assert len(created) == 2


def test_rows_gives_up_after_one_retry(conns):
    created, pending = conns
    pending.append(FakeConn(fail=1))
    pending.append(FakeConn(fail=1))               # retry conn also dies
    q = Queries()
    with pytest.raises(psycopg.OperationalError):
        q._rows("SELECT 1")
    assert len(created) == 2                       # no third attempt


def test_rows_does_not_retry_query_errors(conns):
    created, pending = conns
    pending.append(FakeConn(fail=1, exc=psycopg.ProgrammingError))
    q = Queries()
    with pytest.raises(psycopg.ProgrammingError):
        q._rows("SELECT nope")
    assert len(created) == 1                       # a bad query is not a dead conn


def test_rows_guards_execute_with_a_lock(conns):
    """uvicorn may run sync handlers in a threadpool; the shared conn needs a lock."""
    q = Queries()
    assert isinstance(q._lock, type(threading.Lock()))


# --- verdict writes are ONE round trip returning the joined slot row -----------

# every verdict write's SELECT carries the reel bookkeeping after the slot columns
_EXTRA_KEYS = ["total", "decided", "cut_dirty"]
_EXTRAS = (8, 3, True)


def _fake_slot_row(swapped_moment_id=None, candidates=(), params=None,
                   moment_start=2.0, moment_end=6.0, extras=_EXTRAS):
    """Positional row matching _SLOT_KEYS (+ the verdict-write extras appended);
    json-ish columns get real dict/list values."""
    row = list(range(len(_SLOT_KEYS)))
    overrides = {"swapped_moment_id": swapped_moment_id, "candidates": list(candidates),
                 "params": params if params is not None else {"in_point": 4.2},
                 "human_note": None, "moment_start": moment_start, "moment_end": moment_end}
    for key, val in overrides.items():
        row[_SLOT_KEYS.index(key)] = val
    return tuple(row) + tuple(extras)


def _capture_rows(q):
    calls = []

    def fake_rows(sql, params=()):
        calls.append((sql, params))
        return [_fake_slot_row()]

    q._rows = fake_rows
    return calls


_ALL_KEYS = list(_SLOT_KEYS) + ["effective_in"] + _EXTRA_KEYS


def test_keep_slot_is_a_single_round_trip():
    q = Queries()
    calls = _capture_rows(q)
    slot = q.keep_slot(71, "solid pick")
    assert len(calls) == 1
    sql = calls[0][0].lower()
    assert "update slots" in sql and "left join moments" in sql
    assert list(slot.keys()) == _ALL_KEYS


def test_weak_keep_sets_weak_verdict_without_dirtying_the_cut():
    q = Queries()
    calls = _capture_rows(q)
    slot = q.weak_keep_slot(71, "ok not great")
    assert len(calls) == 1
    sql = calls[0][0].lower()
    assert "human_verdict = 'weak'" in sql
    assert "update reels set cut_dirty = true" not in sql   # still in the reel, cut unchanged
    assert list(slot.keys()) == _ALL_KEYS


def test_swap_slot_is_a_single_round_trip():
    q = Queries()
    calls = _capture_rows(q)
    slot = q.swap_slot(71, 901, "too dark")
    assert len(calls) == 1
    assert "update slots" in calls[0][0].lower()
    assert slot["id"] == 0                          # first _SLOT_KEYS column mapped


def test_flag_slot_is_a_single_round_trip():
    q = Queries()
    calls = _capture_rows(q)
    slot = q.flag_slot(71, "too dark: blurry")
    assert len(calls) == 1
    assert "update slots" in calls[0][0].lower()
    assert list(slot.keys()) == _ALL_KEYS


def test_undo_slot_is_a_single_round_trip_clearing_the_verdict():
    q = Queries()
    calls = _capture_rows(q)
    slot = q.undo_slot(71)
    assert len(calls) == 1
    sql = calls[0][0].lower()
    assert "update slots" in sql
    assert "human_verdict = null" in sql
    assert "swapped_moment_id = null" in sql
    assert calls[0][1] == (71, 71)                  # before-CTE + UPDATE target
    assert list(slot.keys()) == _ALL_KEYS


def test_undo_slot_preserves_the_note():
    """A note is user data - undoing a verdict must NOT wipe it (live data-loss bug)."""
    q = Queries()
    calls = _capture_rows(q)
    q.undo_slot(71)
    assert "human_note = null" not in calls[0][0].lower()


# --- cut freshness rides on the same verdict statement ---------------------------

def test_swap_slot_marks_the_reel_cut_dirty_in_the_same_statement():
    q = Queries()
    calls = _capture_rows(q)
    slot = q.swap_slot(71, 901)
    sql = calls[0][0].lower()
    assert len(calls) == 1                           # still ONE round trip
    assert "update reels set cut_dirty = true" in sql
    assert slot["cut_dirty"] is True and slot["total"] == 8 and slot["decided"] == 3


def test_undo_slot_marks_dirty_only_when_a_swap_was_undone():
    q = Queries()
    calls = _capture_rows(q)
    q.undo_slot(71)
    sql = calls[0][0].lower()
    assert "update reels set cut_dirty = true" in sql
    assert "old_verdict = 'swapped'" in sql          # conditional: only undo-of-a-swap


def test_keep_and_flag_do_not_touch_the_reel_cut():
    q = Queries()
    calls = _capture_rows(q)
    q.keep_slot(71)
    q.flag_slot(71, "too dark")
    for sql, _params in calls:
        assert "update reels" not in sql.lower()     # the cut did not change


def test_verdict_writes_carry_decided_counts_for_the_live_header():
    q = Queries()
    calls = _capture_rows(q)
    slot = q.keep_slot(71)
    sql = calls[0][0].lower()
    assert "human_verdict is not null" in sql        # decided computed in-statement
    assert slot["decided"] == 3 and slot["total"] == 8


def test_verdict_writes_return_none_for_missing_slot():
    q = Queries()
    q._rows = lambda sql, params=(): []
    assert q.keep_slot(999) is None
    assert q.swap_slot(999, 1) is None
    assert q.flag_slot(999, "x") is None
    assert q.undo_slot(999) is None


# --- effective_in rides along on every slot read/write --------------------------

def test_slot_rows_carry_effective_in_unswapped():
    q = Queries()
    q._rows = lambda sql, params=(): [_fake_slot_row()]
    assert q.keep_slot(71)["effective_in"] == 4.2            # params.in_point


def test_slot_rows_carry_effective_in_swapped_candidate():
    cands = [{"moment_id": 901, "in_point": 1.2}]
    q = Queries()
    q._rows = lambda sql, params=(): [_fake_slot_row(swapped_moment_id=901, candidates=cands)]
    assert q.swap_slot(71, 901)["effective_in"] == 1.2       # the candidate's in_point


def test_slot_rows_carry_effective_in_swapped_midpoint_fallback():
    q = Queries()
    q._rows = lambda sql, params=(): [_fake_slot_row(swapped_moment_id=999,
                                                     moment_start=2.0, moment_end=6.0)]
    assert q._slot(71)["effective_in"] == 4.0                # swapped moment midpoint


def test_reel_detail_slots_carry_effective_in(monkeypatch):
    q = Queries()

    def fake_rows(sql, params=()):
        if "FROM reels" in sql:
            return [(5, "reference/DISCIPLINE.mp4", None, False, None, False, None, None,
                        False, None, True)]
        return [_fake_slot_row()]

    q._rows = fake_rows
    detail = q.reel_detail(5)
    assert detail["slots"][0]["effective_in"] == 4.2
    assert detail["reel"]["cut_dirty"] is False      # freshness reaches the template


# --- rerender area helpers -------------------------------------------------------

def test_decided_counts_maps_row():
    q = Queries()
    calls = []

    def fake_rows(sql, params=()):
        calls.append((sql, params))
        return [(3, 8)]

    q._rows = fake_rows
    assert q.decided_counts(5) == {"decided": 3, "total": 8}
    assert calls[0][1] == (5,)


def test_reel_cut_dirty_reads_the_flag():
    q = Queries()
    q._rows = lambda sql, params=(): [(True,)]
    assert q.reel_cut_dirty(5) is True
    q._rows = lambda sql, params=(): []
    assert q.reel_cut_dirty(999) is False


def test_latest_rerender_maps_row_and_key():
    q = Queries()
    calls = []

    def fake_rows(sql, params=()):
        calls.append((sql, params))
        return [("done", None, "output/reel5_v2.mp4")]

    q._rows = fake_rows
    assert q.latest_rerender(5) == {"status": "done", "error": None,
                                    "output_key": "output/reel5_v2.mp4"}
    assert "rerender:5" in calls[0][1]                        # keyed to this reel


def test_latest_rerender_none_when_never_rerendered():
    q = Queries()
    q._rows = lambda sql, params=(): []
    assert q.latest_rerender(5) is None


# --- repick queue helpers ----------------------------------------------------------

def test_latest_repick_maps_row_and_key():
    q = Queries()
    calls = []

    def fake_rows(sql, params=()):
        calls.append((sql, params))
        return [("running", None)]

    q._rows = fake_rows
    assert q.latest_repick(71) == {"status": "running", "error": None}
    assert "repick:71" in calls[0][1]                 # keyed to this slot


def test_latest_repick_none_when_never_repicked():
    q = Queries()
    q._rows = lambda sql, params=(): []
    assert q.latest_repick(71) is None


def test_slot_detail_returns_the_joined_row():
    q = Queries()
    q._rows = lambda sql, params=(): [_fake_slot_row()]
    assert q.slot_detail(71)["effective_in"] == 4.2


def test_grade_reel_endorses_every_unflagged_beat():
    # A positive grade turns "watch, flag the misses, grade" into per-beat learning:
    # unflagged beats (0 and 2) are accepted at the exact moment they played.
    q = Queries()
    slot_rows = [(0, "keep", 801), (1, "flagged", 802), (2, None, 803)]
    q._rows = lambda sql, params=(): slot_rows
    res = q.grade_reel(7, "post_worthy")
    assert res["approved"] is True and res["grade"] == "post_worthy"
    assert res["slots"] == 3 and res["flagged"] == 1 and res["kept"] == 1
    assert res["accepted"] == [{"beat": 0, "moment_id": 801, "verdict": "keep"},
                               {"beat": 2, "moment_id": 803, "verdict": None}]


def test_grade_reel_not_it_endorses_nothing():
    q = Queries()
    q._rows = lambda sql, params=(): [(0, "keep", 801)]
    res = q.grade_reel(7, "not_it")
    assert res["approved"] is False and res["accepted"] == []


def test_grade_reel_rejects_unknown_grade():
    q = Queries()
    with pytest.raises(ValueError):
        q.grade_reel(7, "meh")
