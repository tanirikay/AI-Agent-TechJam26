"""Deterministic price policies over an already relevant candidate pool."""

from __future__ import annotations

from collections.abc import Mapping

from src.vocab import BudgetConstraint, parse_price


def _candidate_price(asin: str, products: Mapping[str, dict]) -> float | None:
    product = products.get(asin)
    return parse_price(product.get("price")) if product is not None else None


def rerank_by_maximum_budget(
    pool_asins: list[str],
    products: Mapping[str, dict],
    amount: float,
    tolerance: float = 1.15,
) -> list[str]:
    """Stably demote only candidates confirmed above the tolerated maximum."""
    threshold = amount * tolerance
    non_violations: list[str] = []
    violations: list[str] = []
    for asin in pool_asins:
        price = _candidate_price(asin, products)
        if price is not None and price > threshold:
            violations.append(asin)
        else:
            # Unknown is neutral, not confirmed affordable. Keeping it in the
            # incumbent tier preserves its original relevance order.
            non_violations.append(asin)
    return non_violations + violations


def rerank_by_target_price(
    pool_asins: list[str],
    products: Mapping[str, dict],
    amount: float,
    *,
    min_coverage: float = 0.10,
    window: int = 40,
    band: float = 0.15,
    distance_sort: bool = False,
) -> list[str]:
    """Optionally tier a bounded prefix by proximity to an explicit target."""
    if not pool_asins or window <= 0:
        return pool_asins

    prefix = pool_asins[:window]
    suffix = pool_asins[window:]
    prices = {asin: _candidate_price(asin, products) for asin in prefix}
    known_count = sum(price is not None for price in prices.values())
    if known_count / len(prefix) < min_coverage:
        return pool_asins

    lower = amount * (1.0 - band)
    upper = amount * (1.0 + band)
    within: list[str] = []
    unknown: list[str] = []
    outside: list[str] = []
    for asin in prefix:
        price = prices[asin]
        if price is None:
            unknown.append(asin)
        elif lower <= price <= upper:
            within.append(asin)
        else:
            outside.append(asin)

    if distance_sort:
        within.sort(key=lambda asin: abs(float(prices[asin]) - amount))
        outside.sort(key=lambda asin: abs(float(prices[asin]) - amount))

    return within + unknown + outside + suffix


def rerank_by_price_constraint(
    pool_asins: list[str],
    products: Mapping[str, dict],
    constraint: BudgetConstraint | None,
    *,
    maximum_tolerance: float = 1.15,
    target_enabled: bool = False,
    target_min_coverage: float = 0.10,
    target_window: int = 40,
    target_band: float = 0.15,
    target_distance_sort: bool = False,
) -> list[str]:
    """Apply the policy selected by an explicit, mode-aware constraint."""
    if constraint is None:
        return pool_asins
    if constraint.mode == "maximum":
        return rerank_by_maximum_budget(
            pool_asins, products, constraint.amount, maximum_tolerance
        )
    if not target_enabled:
        return pool_asins
    return rerank_by_target_price(
        pool_asins,
        products,
        constraint.amount,
        min_coverage=target_min_coverage,
        window=target_window,
        band=target_band,
        distance_sort=target_distance_sort,
    )
