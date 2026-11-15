"""_maybe_trace: LangSmith wrapping is strictly opt-in via LANGSMITH_API_KEY.

No test here makes a network call - clients are constructed with fake keys and never used.
"""
import anthropic
import pytest

from reforge.llm.claude import _maybe_trace


def _client() -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key="test-key-not-real")


def test_no_op_without_langsmith_key(monkeypatch):
    """Without LANGSMITH_API_KEY the client must come back untouched (same object)."""
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    client = _client()
    assert _maybe_trace(client) is client


def test_no_op_does_not_set_langsmith_env(monkeypatch):
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.delenv("LANGSMITH_PROJECT", raising=False)
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    import os

    _maybe_trace(_client())
    assert "LANGSMITH_PROJECT" not in os.environ
    assert "LANGSMITH_TRACING" not in os.environ


def test_wraps_when_key_present(monkeypatch):
    """With the key set (and langsmith installed), the returned client is still functional
    (usable messages API) and tracing env defaults are applied."""
    pytest.importorskip("langsmith")
    import os

    monkeypatch.setenv("LANGSMITH_API_KEY", "ls-test-key-not-real")
    monkeypatch.delenv("LANGSMITH_PROJECT", raising=False)
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)

    wrapped = _maybe_trace(_client())
    # wrap_anthropic patches the client (may return the same object) - it must still
    # expose the messages surface _create relies on.
    assert hasattr(wrapped, "messages") and hasattr(wrapped.messages, "create")
    assert callable(wrapped.messages.create)
    assert os.environ["LANGSMITH_PROJECT"] == "reforge"
    assert os.environ["LANGSMITH_TRACING"] == "true"


def test_explicit_project_env_wins(monkeypatch):
    """setdefault semantics: an explicitly configured project is not overwritten."""
    pytest.importorskip("langsmith")
    import os

    monkeypatch.setenv("LANGSMITH_API_KEY", "ls-test-key-not-real")
    monkeypatch.setenv("LANGSMITH_PROJECT", "custom-project")
    _maybe_trace(_client())
    assert os.environ["LANGSMITH_PROJECT"] == "custom-project"
