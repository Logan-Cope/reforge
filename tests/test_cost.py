"""Unit tests for the pure cost estimator in llm/cost.py."""
import pytest

from reforge.llm.cost import estimate_cost


def test_estimate_cost_per_model_and_total():
    usage = {
        "claude-haiku-4-5": {"input": 1_000_000, "output": 200_000, "calls": 17},
        "claude-opus-4-8": {"input": 500_000, "output": 100_000, "calls": 16},
    }
    out = estimate_cost(usage)
    # haiku $1/$5 per 1M -> 1.0 + 1.0 = 2.0 ; opus $5/$25 -> 2.5 + 2.5 = 5.0
    assert out["per_model"]["claude-haiku-4-5"]["usd"] == pytest.approx(2.0)
    assert out["per_model"]["claude-opus-4-8"]["usd"] == pytest.approx(5.0)
    assert out["total_usd"] == pytest.approx(7.0)
    assert out["total_calls"] == 33


def test_estimate_cost_unknown_model_unpriced_but_counted():
    out = estimate_cost({"mystery": {"input": 1_000_000, "output": 0, "calls": 1}})
    assert out["per_model"]["mystery"]["usd"] == 0.0
    assert out["total_usd"] == 0.0
    assert out["total_calls"] == 1


def test_estimate_cost_empty():
    out = estimate_cost({})
    assert out["total_usd"] == 0.0 and out["total_calls"] == 0 and out["per_model"] == {}
