"""Cost planning.

Total calls ~= n_items * n_metrics * n_repeats * n_conditions * calls_per_metric. Prices
come from a user-supplied table (YAML), never hardcoded (CLAUDE.md rule 6). The CLI prints
a plan and asks for confirmation above a configured budget before any live run.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from causeval.core.errors import ConfigError


@dataclass(frozen=True)
class ModelPricing:
    """USD per 1M tokens. Values come from user config, not from causeval."""

    input_usd_per_1m: float
    output_usd_per_1m: float


@dataclass(frozen=True)
class TokensPerCall:
    prompt: int
    completion: int


PriceTable = dict[str, ModelPricing]


def load_price_table(path: str | Path) -> PriceTable:
    """Load a price table from YAML.

    Expected shape::

        models:
          gpt-4o-mini:
            input_usd_per_1m: 0.15
            output_usd_per_1m: 0.60
    """
    raw: Any = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict) or "models" not in raw:
        raise ConfigError(f"price table {path!s} must have a top-level 'models' mapping")
    table: PriceTable = {}
    for model, entry in raw["models"].items():
        try:
            table[model] = ModelPricing(
                input_usd_per_1m=float(entry["input_usd_per_1m"]),
                output_usd_per_1m=float(entry["output_usd_per_1m"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigError(f"invalid pricing for model {model!r}: {exc}") from exc
    return table


def cost_per_call(model: str, tokens: TokensPerCall, prices: PriceTable) -> float:
    if model not in prices:
        raise ConfigError(
            f"no price table entry for model {model!r}; add it to your config "
            f"(known models: {sorted(prices)})"
        )
    p = prices[model]
    return (
        tokens.prompt / 1_000_000 * p.input_usd_per_1m
        + tokens.completion / 1_000_000 * p.output_usd_per_1m
    )


class CostPlan(BaseModel):
    """Estimated cost of a run, shown before any live call."""

    model: str
    n_items: int
    n_metrics: int
    n_repeats: int
    n_conditions: int
    calls_per_metric: int
    total_calls: int
    prompt_tokens_per_call: int
    completion_tokens_per_call: int
    total_tokens: int
    est_cost_usd: float


def plan(
    *,
    n_items: int,
    n_metrics: int,
    n_repeats: int,
    n_conditions: int,
    calls_per_metric: int,
    tokens_per_call: TokensPerCall,
    model: str,
    price_table: PriceTable,
) -> CostPlan:
    """Estimate the cost of a run. Raises :class:`ConfigError` if ``model`` is unpriced."""
    total_calls = n_items * n_metrics * n_repeats * n_conditions * calls_per_metric
    per_call = cost_per_call(model, tokens_per_call, price_table)
    total_tokens = total_calls * (tokens_per_call.prompt + tokens_per_call.completion)
    return CostPlan(
        model=model,
        n_items=n_items,
        n_metrics=n_metrics,
        n_repeats=n_repeats,
        n_conditions=n_conditions,
        calls_per_metric=calls_per_metric,
        total_calls=total_calls,
        prompt_tokens_per_call=tokens_per_call.prompt,
        completion_tokens_per_call=tokens_per_call.completion,
        total_tokens=total_tokens,
        est_cost_usd=round(total_calls * per_call, 6),
    )
