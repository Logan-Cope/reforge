"""Tavily trend-context tool: pure query/digest helpers + no-key/no-network fallback."""
from types import SimpleNamespace

from reforge.acquire.trends import _digest, _trend_query, trend_context
from reforge.llm.base import ReelBrief
from reforge.match.retrieve import brief_summary


def _brief(**overrides):
    kw = dict(vibe="late-night grind", emotional_register="gritty",
              energy_profile="builds", tone="serious", theme="discipline")
    kw.update(overrides)
    return ReelBrief(**kw)


# --- _trend_query: pure composition ------------------------------------------

def test_trend_query_includes_theme_overlay_and_ref_name():
    q = _trend_query(_brief(), overlay_text="DISCIPLINE", ref_name="DISCIPLINE.mp4")
    assert "discipline" in q.lower()
    assert '"DISCIPLINE"' in q
    assert "instagram reel" in q.lower()


def test_trend_query_stays_short():
    q = _trend_query(_brief(vibe="a very long moody late night city grind session vibe"),
                     overlay_text="NO DAYS OFF EVER AGAIN", ref_name="ref.mp4")
    assert len(q.split()) <= 15


def test_trend_query_works_without_overlay_or_ref():
    q = _trend_query(_brief(), overlay_text="", ref_name="")
    assert "discipline" in q.lower()
    assert '""' not in q


# --- _digest: pure join + truncation ------------------------------------------

def test_digest_joins_titles_and_content():
    results = [
        {"title": "Why moody gym reels work", "content": "Dark b-roll taps aspiration."},
        {"title": "Discipline trend explained", "content": "Overlay text carries theme."},
    ]
    out = _digest(results)
    assert "Why moody gym reels work" in out
    assert "Overlay text carries theme." in out


def test_digest_truncates_at_word_boundary():
    results = [{"title": "T", "content": "word " * 500}]
    out = _digest(results, max_chars=50)
    assert len(out) <= 50
    assert not out.rstrip("…").endswith(" ")   # no dangling half-word / trailing space
    assert "wor…" not in out or out.endswith("…")


def test_digest_empty_results_is_empty_string():
    assert _digest([]) == ""
    assert _digest([{"title": "", "content": ""}]) == ""


# --- trend_context: never a blocker --------------------------------------------

def test_trend_context_returns_empty_without_api_key(monkeypatch):
    import reforge.acquire.trends as trends
    monkeypatch.setattr(trends, "settings", SimpleNamespace(tavily_api_key=""))
    assert trend_context(_brief(), overlay_text="DISCIPLINE") == ""


def test_trend_context_returns_empty_on_request_failure(monkeypatch):
    import reforge.acquire.trends as trends
    monkeypatch.setattr(trends, "settings", SimpleNamespace(tavily_api_key="tvly-test"))

    def boom(*a, **kw):
        raise RuntimeError("network down")
    monkeypatch.setattr(trends.requests, "post", boom)
    assert trend_context(_brief(), overlay_text="DISCIPLINE") == ""


# --- brief_summary surfaces trend context to the judge --------------------------

def test_brief_summary_appends_trend_context_when_set():
    b = _brief()
    b.trend_context = "This trend pairs stark overlays with moody gym b-roll."
    out = brief_summary(b)
    assert "TREND CONTEXT: This trend pairs stark overlays" in out


def test_brief_summary_unchanged_when_trend_context_empty():
    out = brief_summary(_brief())
    assert "TREND CONTEXT" not in out
    assert out.endswith("THEME: discipline.")
