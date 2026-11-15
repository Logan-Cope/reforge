"""Per-run LLM cost estimation (real per-reel cost, not a guess).

`ClaudeVision` accumulates token usage by model across a run; `estimate_cost` prices it.
Pricing is $ per 1M tokens (input, output). Pure + unit-tested.
"""
from __future__ import annotations

# ($ per 1M input tokens, $ per 1M output tokens)
MODEL_PRICING: dict[str, tuple[float, float]] = {
    "claude-fable-5": (10.0, 50.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-opus-4-7": (5.0, 25.0),
    # standard price; intro pricing ($2/$10) runs through 2026-08-31 - we price conservatively
    "claude-sonnet-5": (3.0, 15.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
}


def estimate_cost(usage_by_model: dict[str, dict]) -> dict:
    """Price accumulated usage. ``usage_by_model[model] = {input, output, calls}``.

    Returns ``{per_model: {model: {input, output, calls, usd}}, total_usd, total_calls}``.
    Unknown models are counted but priced at $0 (flagged by a 0 cost).
    """
    per_model: dict[str, dict] = {}
    total_usd = 0.0
    total_calls = 0
    for model, u in usage_by_model.items():
        inp = u.get("input", 0)
        out = u.get("output", 0)
        calls = u.get("calls", 0)
        price_in, price_out = MODEL_PRICING.get(model, (0.0, 0.0))
        usd = inp / 1_000_000 * price_in + out / 1_000_000 * price_out
        per_model[model] = {"input": inp, "output": out, "calls": calls, "usd": usd}
        total_usd += usd
        total_calls += calls
    return {"per_model": per_model, "total_usd": total_usd, "total_calls": total_calls}
