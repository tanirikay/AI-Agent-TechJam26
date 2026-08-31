"""Deterministic synthetic diagnostics for explicit budget/price policies.

These fixtures are not official public metrics and do not use public labels.
They isolate price coverage, missing prices, cheap decoys, and target position
inside an already relevant 20-candidate pool.
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path

from src.price_rerank import rerank_by_maximum_budget, rerank_by_price_constraint
from src.vocab import BudgetConstraint, parse_price


POOL_SIZE = 20
TOP_K = 10
MISS_TURN = 11
COVERAGE_BUCKETS = (0, 5, 10, 25, 100)


@dataclass(frozen=True)
class SyntheticCase:
    name: str
    request: BudgetConstraint
    target: str
    pool: list[str]
    products: dict[str, dict]
    tags: tuple[str, ...]

    @property
    def coverage_percent(self) -> int:
        known = sum(
            parse_price(self.products[asin].get("price")) is not None
            for asin in self.pool
        )
        return round(100 * known / len(self.pool))


def _case(
    name: str,
    mode: str,
    target_price: float | None,
    coverage_percent: int,
    *,
    target_rank: int = 12,
    tags: tuple[str, ...] = (),
    leading_prices: tuple[float | None, ...] = (),
) -> SyntheticCase:
    target = f"{name}_target"
    decoys = [f"{name}_decoy_{i:02d}" for i in range(POOL_SIZE - 1)]
    pool = decoys[: target_rank - 1] + [target] + decoys[target_rank - 1 :]
    products = {
        asin: {"parent_asin": asin, "price": None}
        for asin in pool
    }
    products[target]["price"] = target_price

    desired_known = coverage_percent * POOL_SIZE // 100
    known_so_far = int(target_price is not None)
    if known_so_far > desired_known:
        raise ValueError(f"{name}: target price exceeds requested coverage")

    price_cycle = (20.0, 90.0, 50.0, 35.0, 75.0, 45.0, 110.0)
    decoy_index = 0
    for index, price in enumerate(leading_prices):
        if index >= len(pool):
            break
        asin = pool[index]
        if asin == target or price is None or known_so_far >= desired_known:
            continue
        products[asin]["price"] = price
        known_so_far += 1

    while known_so_far < desired_known:
        asin = decoys[decoy_index]
        decoy_index += 1
        if products[asin]["price"] is not None:
            continue
        products[asin]["price"] = price_cycle[(decoy_index - 1) % len(price_cycle)]
        known_so_far += 1

    return SyntheticCase(
        name=name,
        request=BudgetConstraint(50.0, mode),
        target=target,
        pool=pool,
        products=products,
        tags=tags,
    )


def build_cases() -> list[SyntheticCase]:
    cases = [
        # Required price-coverage buckets for target requests. At 0% and 5%
        # the relevant target is deliberately unknown-priced.
        _case("target_near_cov_0", "target", None, 0, tags=("target price", "target price null")),
        _case("target_near_cov_5", "target", None, 5, tags=("target price", "low coverage")),
        _case("target_near_cov_10", "target", 50.0, 10, tags=("target price", "known relevant")),
        _case("target_near_cov_25", "target", 50.0, 25, tags=("target price", "known relevant")),
        _case("target_near_cov_100", "target", 50.0, 100, tags=("target price", "known relevant")),
        # Maximum-budget behavior for below/near/above/null targets.
        _case("maximum_target_below", "maximum", 40.0, 25, tags=("maximum budget", "target below budget")),
        _case("maximum_target_near", "maximum", 50.0, 25, tags=("maximum budget", "target near requested price")),
        _case("maximum_target_above", "maximum", 90.0, 25, tags=("maximum budget", "target above budget")),
        _case("maximum_target_null", "maximum", None, 25, tags=("maximum budget", "target price null", "relevant unknown")),
        _case("target_price_null", "target", None, 25, tags=("target price", "target price null", "relevant unknown")),
        # Known-over-budget candidates ahead of a relevant known/unknown item
        # exercise safe demotion at partial coverage.
        _case(
            "maximum_known_relevant_after_overages",
            "maximum",
            40.0,
            100,
            tags=("maximum budget", "relevant known", "target below budget"),
            leading_prices=(90.0,) * 8,
        ),
        _case(
            "maximum_unknown_relevant_after_overages",
            "maximum",
            None,
            25,
            tags=("maximum budget", "relevant unknown"),
            leading_prices=(90.0,) * 5,
        ),
        # Cheap products are explicitly irrelevant decoys. Maximum mode must
        # not promote them merely for being cheap; target tiering may move the
        # genuinely near-price target ahead of them when coverage permits.
        _case(
            "target_near_after_irrelevant_cheap_decoys",
            "target",
            50.0,
            100,
            tags=("target price", "irrelevant cheap decoys", "relevant known"),
            leading_prices=(20.0,) * 10,
        ),
        _case(
            "maximum_unknown_after_irrelevant_cheap_decoys",
            "maximum",
            None,
            100 - (100 // POOL_SIZE),
            tags=("maximum budget", "irrelevant cheap decoys", "relevant unknown"),
            leading_prices=(20.0,) * 10,
        ),
        _case("target_outside_band", "target", 20.0, 100, tags=("target price", "target below budget")),
    ]
    return cases


def _rank(case: SyntheticCase, ranking: list[str]) -> int | None:
    return ranking.index(case.target) + 1 if case.target in ranking else None


def _metrics(rows: list[dict]) -> dict:
    if not rows:
        return {
            "sample_count": 0,
            "hit_rate_at_10": 0.0,
            "mrr": 0.0,
            "mttc": None,
            "technical_score": 0.0,
        }
    hit_rate = statistics.fmean(row["hit"] for row in rows)
    mrr = statistics.fmean(row["reciprocal_rank"] for row in rows)
    mttc = statistics.fmean(1 if row["hit"] else MISS_TURN for row in rows)
    efficiency = max(0.0, min(1.0, (11.0 - mttc) / 10.0))
    technical_score = 0.50 * hit_rate + 0.30 * mrr + 0.20 * efficiency
    return {
        "sample_count": len(rows),
        "hit_rate_at_10": round(hit_rate, 6),
        "mrr": round(mrr, 6),
        "mttc": round(mttc, 6),
        "technical_score": round(technical_score, 6),
    }


def _apply(config: str, case: SyntheticCase) -> list[str]:
    if config == "control_current_policy":
        # Before mode awareness every recognized number was treated as a
        # maximum, including legacy "around $50" phrases.
        return rerank_by_maximum_budget(
            case.pool, case.products, case.request.amount, tolerance=1.15
        )
    if config == "mode_aware_default":
        return rerank_by_price_constraint(case.pool, case.products, case.request)
    if config.startswith("target_tiering_min_"):
        coverage = float(config.rsplit("_", 1)[1])
        return rerank_by_price_constraint(
            case.pool,
            case.products,
            case.request,
            target_enabled=True,
            target_min_coverage=coverage,
            target_window=40,
        )
    if config == "target_distance_min_0.10":
        return rerank_by_price_constraint(
            case.pool,
            case.products,
            case.request,
            target_enabled=True,
            target_min_coverage=0.10,
            target_window=40,
            target_distance_sort=True,
        )
    raise ValueError(config)


def run_diagnostic() -> dict:
    cases = build_cases()
    configs = (
        "control_current_policy",
        "mode_aware_default",
        "target_tiering_min_0.00",
        "target_tiering_min_0.05",
        "target_tiering_min_0.10",
        "target_tiering_min_0.25",
        "target_distance_min_0.10",
    )
    rows_by_config: dict[str, list[dict]] = {}
    for config in configs:
        rows: list[dict] = []
        for case in cases:
            ranking = _apply(config, case)
            rank = _rank(case, ranking)
            hit = rank is not None and rank <= TOP_K
            rows.append({
                "case": case.name,
                "coverage_percent": case.coverage_percent,
                "mode": case.request.mode,
                "tags": list(case.tags),
                "rank": rank,
                "hit": hit,
                "reciprocal_rank": 0.0 if not hit else 1.0 / rank,
            })
        rows_by_config[config] = rows

    control_by_case = {
        row["case"]: row for row in rows_by_config["control_current_policy"]
    }
    config_reports: dict[str, dict] = {}
    for config, rows in rows_by_config.items():
        gained: list[str] = []
        lost: list[str] = []
        rank_better: list[str] = []
        rank_worse: list[str] = []
        hit_gained: list[str] = []
        hit_lost: list[str] = []
        for row in rows:
            control = control_by_case[row["case"]]
            if row["reciprocal_rank"] > control["reciprocal_rank"]:
                gained.append(row["case"])
            elif row["reciprocal_rank"] < control["reciprocal_rank"]:
                lost.append(row["case"])
            if row["rank"] < control["rank"]:
                rank_better.append(row["case"])
            elif row["rank"] > control["rank"]:
                rank_worse.append(row["case"])
            if row["hit"] and not control["hit"]:
                hit_gained.append(row["case"])
            elif control["hit"] and not row["hit"]:
                hit_lost.append(row["case"])

        by_coverage = {
            f"{bucket}%": _metrics([
                row for row in rows if row["coverage_percent"] == bucket
            ])
            for bucket in COVERAGE_BUCKETS
        }
        config_reports[config] = {
            "metrics": _metrics(rows),
            "paired_vs_control": {
                "gained_cases": gained,
                "lost_cases": lost,
                "hit_gained_cases": hit_gained,
                "hit_lost_cases": hit_lost,
                "rank_better_cases": rank_better,
                "rank_worse_cases": rank_worse,
            },
            "by_price_coverage": by_coverage,
        }

    return {
        "diagnostic_label": "synthetic robustness diagnostic; not official public metrics",
        "case_count": len(cases),
        "coverage_buckets_percent": list(COVERAGE_BUCKETS),
        "paired_definition": (
            "gained/lost compares per-case reciprocal rank; hit and raw-rank "
            "changes are reported separately"
        ),
        "configs": config_reports,
        "cases": [
            {
                "name": case.name,
                "mode": case.request.mode,
                "amount": case.request.amount,
                "coverage_percent": case.coverage_percent,
                "target_initial_rank": _rank(case, case.pool),
                "target_catalog_price": case.products[case.target]["price"],
                "tags": list(case.tags),
                "ranks": {
                    config: next(
                        row["rank"] for row in rows_by_config[config]
                        if row["case"] == case.name
                    )
                    for config in configs
                },
            }
            for case in cases
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="runs/budget_robustness.json")
    args = parser.parse_args()
    result = run_diagnostic()
    payload = json.dumps(result, indent=2) + "\n"
    Path(args.output).write_text(payload, encoding="utf-8")
    print(json.dumps({
        "diagnostic_label": result["diagnostic_label"],
        "case_count": result["case_count"],
        "metrics": {
            name: report["metrics"] for name, report in result["configs"].items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
