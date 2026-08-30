from __future__ import annotations

import unittest

from agent.constraint_rerank import (
    build_normalized_product_evidence,
    normalize_constraint_text,
    rerank_by_exact_constraints,
)
from agent.memory import RawConstraint, Source, get_active_raw_constraints, reset, update_state
from agent.vocab import Vocabulary


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


if __name__ == "__main__":
    unittest.main()
