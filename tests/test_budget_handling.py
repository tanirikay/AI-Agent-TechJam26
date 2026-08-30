from __future__ import annotations

import unittest

from agent import memory
from agent.price_rerank import (
    rerank_by_maximum_budget,
    rerank_by_price_constraint,
    rerank_by_target_price,
)
from agent.vocab import (
    BudgetConstraint,
    Vocabulary,
    extract_budget,
    extract_budget_constraint,
    parse_price,
)


def _products(prices: dict[str, object]) -> dict[str, dict]:
    return {
        asin: {"parent_asin": asin, "price": price}
        for asin, price in prices.items()
    }


class BudgetParsingTest(unittest.TestCase):
    def test_maximum_budget_phrases(self) -> None:
        phrases = (
            "under $50",
            "below $50",
            "less than $50",
            "no more than $50",
            "at most $50",
            "maximum $50",
            "budget of $50",
        )
        for phrase in phrases:
            with self.subTest(phrase=phrase):
                self.assertEqual(
                    extract_budget_constraint(phrase),
                    BudgetConstraint(50.0, "maximum"),
                )

    def test_target_price_phrases(self) -> None:
        phrases = (
            "around $50",
            "about $50",
            "approximately $50",
            "roughly $50",
            "close to $50",
        )
        for phrase in phrases:
            with self.subTest(phrase=phrase):
                self.assertEqual(
                    extract_budget_constraint(phrase),
                    BudgetConstraint(50.0, "target"),
                )

    def test_compatibility_wrapper_still_returns_float(self) -> None:
        self.assertEqual(extract_budget("around $42.50"), 42.5)
        self.assertEqual(extract_budget("around 42.50"), 42.5)

    def test_vague_price_words_do_not_create_numeric_budget(self) -> None:
        for word in ("cheap", "affordable", "premium", "expensive"):
            with self.subTest(word=word):
                self.assertIsNone(extract_budget_constraint(f"I want something {word}"))

    def test_non_currency_measurement_does_not_create_target_price(self) -> None:
        self.assertIsNone(extract_budget_constraint("It is about 50 inches long"))

    def test_memory_preserves_mode_while_budget_slot_remains_float(self) -> None:
        state = memory.reset("s", {})
        memory.update_state(state, "I want shoes around $50", 1, Vocabulary())
        self.assertEqual(memory.get_filled_slots(state)["budget"], 50.0)
        self.assertEqual(state.budget_constraint, BudgetConstraint(50.0, "target"))


class PriceParsingTest(unittest.TestCase):
    def test_missing_prices_remain_none(self) -> None:
        for value in (None, "", "unknown", "not available"):
            with self.subTest(value=value):
                self.assertIsNone(parse_price(value))


class MaximumBudgetRerankTest(unittest.TestCase):
    def test_no_budget_is_exactly_inert(self) -> None:
        pool = ["a", "b", "c"]
        result = rerank_by_price_constraint(pool, _products({}), None)
        self.assertIs(result, pool)

    def test_all_null_prices_leave_ranking_unchanged(self) -> None:
        pool = ["a", "b", "c"]
        products = _products({asin: None for asin in pool})
        self.assertEqual(rerank_by_maximum_budget(pool, products, 50.0), pool)

    def test_known_affordable_candidate_is_not_promoted_over_unknown(self) -> None:
        pool = ["unknown", "affordable"]
        products = _products({"unknown": None, "affordable": 20.0})
        self.assertEqual(rerank_by_maximum_budget(pool, products, 50.0), pool)

    def test_known_over_budget_product_is_demoted(self) -> None:
        pool = ["over", "unknown", "affordable"]
        products = _products({"over": 80.0, "unknown": None, "affordable": 40.0})
        self.assertEqual(
            rerank_by_maximum_budget(pool, products, 50.0),
            ["unknown", "affordable", "over"],
        )

    def test_unknown_price_target_is_neutral(self) -> None:
        pool = ["affordable", "target", "over"]
        products = _products({"affordable": 40.0, "target": None, "over": 80.0})
        self.assertEqual(rerank_by_maximum_budget(pool, products, 50.0), pool)

    def test_relative_order_is_stable_inside_maximum_tiers(self) -> None:
        pool = ["over1", "unknown1", "affordable1", "over2", "unknown2", "affordable2"]
        products = _products({
            "over1": 100.0,
            "unknown1": None,
            "affordable1": 40.0,
            "over2": 90.0,
            "unknown2": None,
            "affordable2": 30.0,
        })
        self.assertEqual(
            rerank_by_maximum_budget(pool, products, 50.0),
            ["unknown1", "affordable1", "unknown2", "affordable2", "over1", "over2"],
        )

    def test_maximum_demotion_applies_even_at_low_price_coverage(self) -> None:
        pool = [f"p{i}" for i in range(20)]
        products = _products({asin: None for asin in pool})
        products["p0"]["price"] = 100.0  # 5% coverage, confirmed violation
        result = rerank_by_maximum_budget(pool, products, 50.0)
        self.assertEqual(result[-1], "p0")
        self.assertEqual(result[:-1], pool[1:])


class TargetPriceRerankTest(unittest.TestCase):
    def test_target_mode_is_disabled_by_default(self) -> None:
        pool = ["far", "near"]
        products = _products({"far": 100.0, "near": 50.0})
        result = rerank_by_price_constraint(
            pool, products, BudgetConstraint(50.0, "target")
        )
        self.assertIs(result, pool)

    def test_target_falls_back_when_coverage_is_insufficient(self) -> None:
        pool = [f"p{i}" for i in range(20)]
        products = _products({asin: None for asin in pool})
        products["p19"]["price"] = 50.0  # 5% < 10%
        result = rerank_by_target_price(
            pool, products, 50.0, min_coverage=0.10, window=20
        )
        self.assertIs(result, pool)

    def test_target_activates_at_exact_minimum_coverage(self) -> None:
        pool = [f"p{i}" for i in range(20)]
        products = _products({asin: None for asin in pool})
        products["p0"]["price"] = 100.0
        products["p19"]["price"] = 50.0  # 2 / 20 = 10%
        result = rerank_by_target_price(
            pool, products, 50.0, min_coverage=0.10, window=20
        )
        self.assertEqual(result[0], "p19")
        self.assertEqual(result[-1], "p0")

    def test_high_coverage_target_tiers_near_unknown_then_outside(self) -> None:
        pool = ["outside1", "near1", "unknown1", "near2", "outside2", "unknown2"]
        products = _products({
            "outside1": 100.0,
            "near1": 48.0,
            "unknown1": None,
            "near2": 55.0,
            "outside2": 20.0,
            "unknown2": None,
        })
        self.assertEqual(
            rerank_by_target_price(pool, products, 50.0, min_coverage=0.50, window=6),
            ["near1", "near2", "unknown1", "unknown2", "outside1", "outside2"],
        )

    def test_target_tiers_preserve_relative_order(self) -> None:
        pool = ["outside1", "near1", "near2", "unknown1", "outside2", "unknown2"]
        products = _products({
            "outside1": 80.0,
            "near1": 55.0,
            "near2": 45.0,
            "unknown1": None,
            "outside2": 20.0,
            "unknown2": None,
        })
        self.assertEqual(
            rerank_by_target_price(pool, products, 50.0, min_coverage=0.50, window=6),
            ["near1", "near2", "unknown1", "unknown2", "outside1", "outside2"],
        )

    def test_only_configured_target_window_is_reordered(self) -> None:
        pool = ["far", "near", "suffix_near"]
        products = _products({"far": 100.0, "near": 50.0, "suffix_near": 50.0})
        self.assertEqual(
            rerank_by_target_price(pool, products, 50.0, min_coverage=1.0, window=2),
            ["near", "far", "suffix_near"],
        )

    def test_repeated_output_is_deterministic(self) -> None:
        pool = ["far1", "near1", "unknown", "near2", "far2"]
        products = _products({
            "far1": 90.0,
            "near1": 45.0,
            "unknown": None,
            "near2": 55.0,
            "far2": 10.0,
        })
        first = rerank_by_target_price(
            pool, products, 50.0, min_coverage=0.50, window=5
        )
        for _ in range(20):
            self.assertEqual(
                rerank_by_target_price(
                    pool, products, 50.0, min_coverage=0.50, window=5
                ),
                first,
            )


if __name__ == "__main__":
    unittest.main()
