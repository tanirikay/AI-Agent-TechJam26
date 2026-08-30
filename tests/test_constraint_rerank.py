from __future__ import annotations

import unittest
from types import SimpleNamespace

from agent.agent import Agent
from agent.constraint_rerank import (
    build_normalized_product_evidence,
    find_unmatched_constraints,
    normalize_constraint_text,
    rerank_by_exact_constraints,
)
from agent.memory import RawConstraint, Source, get_active_raw_constraints, reset, update_state
from agent.vocab import Vocabulary, extract_negated_values


class ConstraintNormalizationTest(unittest.TestCase):
    def test_normalization_handles_punctuation_dictionary_separators_and_whitespace(self) -> None:
        product = {
            "parent_asin": "A",
            "title": "Example",
            "details": {"Care Instructions": "Hand-wash   only"},
        }
        evidence = build_normalized_product_evidence(product)
        self.assertEqual(
            normalize_constraint_text("  Care Instructions: Hand wash only. "),
            "care instructions hand wash only",
        )
        self.assertIn("care instructions hand wash only", evidence)


class RawConstraintMemoryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.vocabulary = Vocabulary()

    def test_arbitrary_out_of_vocabulary_key_requirement_is_retained(self) -> None:
        state = reset("s", {})
        update_state(
            state,
            "I'm looking for Walking Shoes. A key requirement is: Rubber sole.",
            1,
            self.vocabulary,
        )
        active = [item.normalized_text for item in get_active_raw_constraints(state)]
        self.assertEqual(active, ["walking shoes", "rubber sole"])

    def test_semicolon_clarification_creates_multiple_constraints(self) -> None:
        state = reset("s", {})
        state.pending_ask = "feature"
        state.slots["feature"].asked = True
        update_state(
            state,
            "For that, what matters is: Pull On closure; Hand wash only.",
            2,
            self.vocabulary,
        )
        self.assertEqual(
            [item.normalized_text for item in get_active_raw_constraints(state)],
            ["pull on closure", "hand wash only"],
        )
        self.assertTrue(all(
            item.source_attribute == "feature" for item in state.raw_constraints
        ))

    def test_no_preference_reply_adds_no_raw_constraint(self) -> None:
        for message in (
            "I don't have a preference for material; please use your judgment.",
            "I don't have an additional preference for material.",
        ):
            with self.subTest(message=message):
                state = reset("s", {})
                state.pending_ask = "material"
                state.slots["material"].asked = True
                update_state(state, message, 2, self.vocabulary)
                self.assertEqual(state.raw_constraints, [])

    def test_override_retires_only_identified_opening_preference(self) -> None:
        state = reset("s", {})
        update_state(
            state,
            "I'm looking for Women's Shoes. Hand wash only.",
            1,
            self.vocabulary,
        )
        state.pending_ask = "material"
        state.slots["material"].asked = True
        update_state(
            state,
            "For that, what matters is: Leather lining.",
            2,
            self.vocabulary,
        )
        update_state(
            state,
            "Actually, ignore my earlier preference. What I need is: cotton.",
            3,
            self.vocabulary,
        )

        by_text = {item.normalized_text: item for item in state.raw_constraints}
        self.assertTrue(by_text["women s shoes"].active)
        self.assertFalse(by_text["hand wash only"].active)
        self.assertEqual(by_text["hand wash only"].replaced_turn, 3)
        self.assertTrue(by_text["leather lining"].active)
        self.assertTrue(by_text["cotton"].active)

    def test_historical_im_to_size_m_behavior_is_unchanged(self) -> None:
        state = reset("s", {})
        update_state(state, "I'm looking for walking shoes.", 1, self.vocabulary)
        self.assertEqual(state.slots["size"].value, "m")

    def test_clarification_exhaustion_remains_intact(self) -> None:
        state = reset("s", {})
        state.pending_ask = "material"
        state.slots["material"].asked = True
        update_state(
            state,
            "I don't have an additional preference for material.",
            2,
            self.vocabulary,
        )
        self.assertTrue(state.slots["material"].answered)
        self.assertEqual(state.slots["material"].source, Source.NO_PREFERENCE)
        self.assertIsNone(state.pending_ask)

    def test_brand_matching_still_respects_word_boundaries(self) -> None:
        vocabulary = Vocabulary(brand_terms={"outdoorx", "king"})
        extracted = vocabulary.extract("I'm looking for outdoor shoes, not outdoorxylophones.")
        self.assertNotIn("brand", extracted)


class ExactConstraintRerankTest(unittest.TestCase):
    def test_ties_preserve_original_bm25_order(self) -> None:
        constraints = [RawConstraint("Rubber sole", "rubber sole", 1, "preference")]
        pool = ["A", "B", "C"]
        evidence = {"A": "mesh upper", "B": "leather upper", "C": "canvas upper"}
        self.assertEqual(
            rerank_by_exact_constraints(pool, evidence, constraints, 3), pool
        )

    def test_matching_product_moves_ahead_of_nonmatching_product(self) -> None:
        constraints = [RawConstraint("Rubber sole", "rubber sole", 1, "preference")]
        self.assertEqual(
            rerank_by_exact_constraints(
                ["nonmatch", "match"],
                {"nonmatch": "mesh upper", "match": "mesh upper rubber sole"},
                constraints,
                2,
            ),
            ["match", "nonmatch"],
        )

    def test_output_is_deterministic(self) -> None:
        constraints = [
            RawConstraint("Rubber sole", "rubber sole", 1, "preference"),
            RawConstraint("Hand wash", "hand wash", 2, "preference"),
        ]
        pool = ["A", "B", "C", "D"]
        evidence = {
            "A": "rubber sole",
            "B": "hand wash rubber sole",
            "C": "hand wash",
            "D": "unmatched",
        }
        first = rerank_by_exact_constraints(pool, evidence, constraints, 4)
        for _ in range(20):
            self.assertEqual(
                rerank_by_exact_constraints(pool, evidence, constraints, 4), first
            )


class UnmatchedConstraintTest(unittest.TestCase):
    """Pure-logic tests for the semantic-fallback trigger condition. No ML
    dependency -- these must always run, even where sentence-transformers
    isn't installed."""

    def test_no_unmatched_when_every_clause_appears_somewhere_in_pool(self) -> None:
        constraints = [RawConstraint("Rubber sole", "rubber sole", 1, "preference")]
        pool = ["A", "B"]
        evidence = {"A": "mesh upper", "B": "canvas upper rubber sole"}
        self.assertEqual(find_unmatched_constraints(pool, evidence, constraints), [])

    def test_clause_missing_from_entire_pool_is_reported(self) -> None:
        rubber = RawConstraint("Rubber sole", "rubber sole", 1, "preference")
        waterproof = RawConstraint("keeps you dry in the rain", "keeps you dry in the rain", 2, "preference")
        pool = ["A", "B"]
        evidence = {"A": "mesh upper rubber sole", "B": "canvas upper rubber sole waterproof"}
        self.assertEqual(find_unmatched_constraints(pool, evidence, [rubber, waterproof]), [waterproof])

    def test_checks_whole_pool_not_just_a_truncated_window(self) -> None:
        # A clause absent from the first candidates but present further down
        # the pool must NOT be reported as unmatched -- that's a ranking
        # question rerank_by_exact_constraints already answers, not a gap.
        constraint = RawConstraint("Rubber sole", "rubber sole", 1, "preference")
        pool = ["A", "B", "C"]
        evidence = {"A": "mesh upper", "B": "canvas upper", "C": "rubber sole"}
        self.assertEqual(find_unmatched_constraints(pool, evidence, [constraint]), [])

    def test_inactive_and_empty_constraints_are_ignored(self) -> None:
        inactive = RawConstraint("Rubber sole", "rubber sole", 1, "preference", active=False)
        empty = RawConstraint("", "", 1, "preference")
        pool = ["A"]
        evidence = {"A": "mesh upper"}
        self.assertEqual(find_unmatched_constraints(pool, evidence, [inactive, empty]), [])


class SemanticFallbackBonusTest(unittest.TestCase):
    """Tests for rerank_by_exact_constraints' optional `bonus` tiebreak."""

    def test_no_bonus_reproduces_pre_fallback_behaviour_exactly(self) -> None:
        constraints = [RawConstraint("Rubber sole", "rubber sole", 1, "preference")]
        pool = ["nonmatch", "match"]
        evidence = {"nonmatch": "mesh upper", "match": "mesh upper rubber sole"}
        without_bonus_param = rerank_by_exact_constraints(pool, evidence, constraints, 2)
        with_none_bonus = rerank_by_exact_constraints(pool, evidence, constraints, 2, bonus=None)
        self.assertEqual(without_bonus_param, with_none_bonus)
        self.assertEqual(without_bonus_param, ["match", "nonmatch"])

    def test_bonus_breaks_ties_among_equal_exact_match_counts(self) -> None:
        # Neither candidate matches the clause (0 exact matches each) -- the
        # bonus is the only signal, so it decides the order.
        constraint = RawConstraint("keeps you dry", "keeps you dry", 1, "preference")
        pool = ["low_sim", "high_sim"]
        evidence = {"low_sim": "cotton shirt", "high_sim": "waterproof jacket"}
        result = rerank_by_exact_constraints(
            pool, evidence, [constraint], 2,
            bonus={"low_sim": 0.01, "high_sim": 0.04},
        )
        self.assertEqual(result, ["high_sim", "low_sim"])

    def test_bonus_at_the_documented_epsilon_cannot_outrank_a_real_exact_match(self) -> None:
        # This is the actual safety property the whole mechanism depends on:
        # a candidate with fewer exact matches must never beat one with more,
        # no matter how confident the semantic guess is, as long as callers
        # respect the documented epsilon bound (agent.py's
        # SEMANTIC_FALLBACK_EPSILON = 0.05, safe for up to ~10 unmatched
        # clauses per session: 10 * 0.05 = 0.5 < 1.0).
        exact_match = RawConstraint("rubber sole", "rubber sole", 1, "preference")
        pool = ["fewer_exact_high_semantic", "more_exact_zero_semantic"]
        evidence = {
            "fewer_exact_high_semantic": "mesh upper",
            "more_exact_zero_semantic": "mesh upper rubber sole",
        }
        max_plausible_unmatched_clauses = 10
        epsilon = 0.05
        worst_case_bonus = epsilon * max_plausible_unmatched_clauses
        self.assertLess(worst_case_bonus, 1.0)
        result = rerank_by_exact_constraints(
            pool, evidence, [exact_match], 2,
            bonus={"fewer_exact_high_semantic": worst_case_bonus, "more_exact_zero_semantic": 0.0},
        )
        self.assertEqual(result, ["more_exact_zero_semantic", "fewer_exact_high_semantic"])


class NegationExclusionTest(unittest.TestCase):
    """Pure-logic tests for extract_negated_values -- no ML dependency,
    always run. See tests/test_semantic_fallback_integration.py for the
    full pipeline test (negation exclusion + real embedding similarity)."""

    def setUp(self) -> None:
        self.vocabulary = Vocabulary()

    def test_no_negation_cue_returns_no_exclusions(self) -> None:
        self.assertEqual(
            extract_negated_values("soft and breathable, everyday wear", self.vocabulary), {}
        )

    def test_literal_material_negation_is_excluded(self) -> None:
        result = extract_negated_values("not black, i prefer something brighter", self.vocabulary)
        self.assertEqual(result, {"color": {"black"}})

    def test_multiple_literal_values_in_one_negated_span_are_all_captured(self) -> None:
        # _match_vocab (single-value-per-slot, used by extract()) would stop
        # at the first match -- this is exactly the bug found and fixed
        # while building this feature.
        result = extract_negated_values("a lightweight jacket, unlike leather or suede", self.vocabulary)
        self.assertEqual(result, {"material": {"leather", "suede"}})

    def test_category_alias_resolves_terms_outside_the_plain_vocabulary(self) -> None:
        # "animal fiber" is not a MATERIAL_VOCAB term at all -- only
        # MATERIAL_CATEGORY_ALIASES can resolve this.
        result = extract_negated_values(
            "made of a plant-based natural material, not an animal fiber", self.vocabulary
        )
        self.assertEqual(result, {"material": {"wool", "silk", "cashmere", "leather", "suede"}})

    def test_negated_span_stops_at_clause_boundary(self) -> None:
        # "not wool" should not also capture "soft" as excluded.
        result = extract_negated_values("not wool, but soft is nice", self.vocabulary)
        self.assertEqual(result, {"material": {"wool"}})


def _stub_agent(products: dict[str, dict]) -> Agent:
    """Agent instance with just enough set up to call
    _apply_negation_exclusion/_matches_excluded -- no real catalog file or
    FTS5 index needed, since those methods only touch self.index."""
    agent = Agent.__new__(Agent)
    agent.index = SimpleNamespace(products=products, vocabulary=Vocabulary())
    return agent


class AlwaysOnNegationExclusionTest(unittest.TestCase):
    """Tests for Agent._apply_negation_exclusion -- the always-on pipeline
    stage added in iteration 8, distinct from the still-opt-in
    semantic_fallback_window. No ML dependency; always runs."""

    def _products(self) -> dict[str, dict]:
        return {
            "sweater": {"parent_asin": "sweater", "title": "Cozy Wool Sweater", "features": ["100% Wool"]},
            "shirt": {"parent_asin": "shirt", "title": "Classic Cotton T-Shirt", "features": ["100% Cotton"]},
            "boot": {"parent_asin": "boot", "title": "Trailblazer Hiking Boot", "features": ["100% Waterproof upper"]},
        }

    def test_no_active_negation_leaves_pool_order_untouched(self) -> None:
        agent = _stub_agent(self._products())
        state = reset("s", {})
        update_state(state, "I'm looking for a soft cotton shirt.", 1, agent.index.vocabulary)
        pool = ["boot", "sweater", "shirt"]
        self.assertEqual(agent._apply_negation_exclusion(pool, state), pool)

    def test_negated_material_from_a_clarification_reply_demotes_it(self) -> None:
        # Realistic shape: a "what matters is:" reply following a pending
        # ask, same pattern the rest of this file already uses -- a bare
        # sentence with no such structure never becomes a RawConstraint at
        # all (see _extract_raw_constraints), regardless of negation.
        agent = _stub_agent(self._products())
        state = reset("s", {})
        state.pending_ask = "material"
        state.slots["material"].asked = True
        update_state(state, "For that, what matters is: without wool.", 1, agent.index.vocabulary)
        pool = ["sweater", "shirt", "boot"]
        # Confirmed-excluded (sweater) sinks; relative order of the rest is preserved.
        self.assertEqual(agent._apply_negation_exclusion(pool, state), ["shirt", "boot", "sweater"])

    def test_category_alias_negation_also_demotes_via_the_always_on_path(self) -> None:
        agent = _stub_agent(self._products())
        state = reset("s", {})
        state.pending_ask = "material"
        state.slots["material"].asked = True
        update_state(
            state,
            "For that, what matters is: made of a plant-based natural material, not an animal fiber.",
            1, agent.index.vocabulary,
        )
        pool = ["sweater", "boot", "shirt"]
        self.assertEqual(agent._apply_negation_exclusion(pool, state), ["boot", "shirt", "sweater"])

    def test_negation_from_turn_one_still_applies_on_a_later_turn(self) -> None:
        # get_active_raw_constraints accumulates across the whole session --
        # a negation stated early should keep demoting candidates later,
        # not just on the turn it was said.
        agent = _stub_agent(self._products())
        state = reset("s", {})
        state.pending_ask = "material"
        state.slots["material"].asked = True
        update_state(state, "For that, what matters is: without wool.", 1, agent.index.vocabulary)
        state.pending_ask = "color"
        state.slots["color"].asked = True
        update_state(state, "I don't have an additional preference for color.", 2, agent.index.vocabulary)
        pool = ["sweater", "shirt", "boot"]
        self.assertEqual(agent._apply_negation_exclusion(pool, state), ["shirt", "boot", "sweater"])


if __name__ == "__main__":
    unittest.main()
