import json
from types import SimpleNamespace

import reforge.taste as taste_mod
from reforge.llm.claude import _rerank_instruction
from reforge.taste import load_rubric


def test_load_rubric_reads_explicit_path(tmp_path):
    path = tmp_path / "rubric.md"
    path.write_text("# Rubric\n- whole body in frame\n")
    assert load_rubric(str(path)) == "# Rubric\n- whole body in frame"


def test_load_rubric_missing_path_returns_empty(tmp_path):
    assert load_rubric(str(tmp_path / "nope.md")) == ""


def test_load_rubric_default_path_uses_settings_taste_dir(tmp_path, monkeypatch):
    (tmp_path / "rubric.md").write_text("- default dir rule\n")
    monkeypatch.setattr(taste_mod, "settings", SimpleNamespace(taste_dir=str(tmp_path)))
    assert load_rubric() == "- default dir rule"


def test_rerank_instruction_without_rubric_is_base():
    base = _rerank_instruction("")
    assert base.startswith("Rank EVERY candidate via record_rerank")
    assert "RUBRIC" not in base


def test_rerank_instruction_prepends_rubric():
    text = _rerank_instruction("- rule one")
    assert "rule one" in text
    assert "TASTE RUBRIC" in text
    assert _rerank_instruction("") in text


def test_rerank_instruction_appends_critique():
    text = _rerank_instruction("- rule one", critique="subject unreadable, too dark")
    assert "rule one" in text                                  # rubric still present
    assert "subject unreadable, too dark" in text              # critique included
    assert "rejected" in text.lower()                          # labeled as gate feedback
    assert _rerank_instruction("") in text                     # base instruction intact


def test_rerank_instruction_no_critique_block_when_empty():
    assert "rejected" not in _rerank_instruction("- rule one").lower()


# --- record_taste_event (app-layer write: DB table + best-effort local file) ----

class _FakeQueries:
    def __init__(self):
        self.events = []

    def insert_taste_event(self, entry):
        self.events.append(entry)


def test_record_taste_event_writes_db_and_local_file(tmp_path, monkeypatch):
    monkeypatch.setattr(taste_mod, "settings", SimpleNamespace(taste_dir=str(tmp_path)))
    q = _FakeQueries()
    taste_mod.record_taste_event({"event": "human_keep", "run": 5}, queries=q)
    assert q.events == [{"event": "human_keep", "run": 5}]
    (line,) = (tmp_path / "taste-log.jsonl").read_text().splitlines()
    entry = json.loads(line)
    assert entry["event"] == "human_keep" and entry["run"] == 5
    assert "ts" in entry                                       # append_taste_log stamps ts


def test_record_taste_event_skips_file_when_unwritable(monkeypatch):
    def boom(entry, path=None):
        raise OSError("read-only filesystem")

    monkeypatch.setattr(taste_mod, "append_taste_log", boom)
    q = _FakeQueries()
    taste_mod.record_taste_event({"event": "clip_label", "label": 1}, queries=q)  # must not raise
    assert q.events == [{"event": "clip_label", "label": 1}]   # DB write still happened
