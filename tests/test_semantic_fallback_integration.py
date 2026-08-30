"""Integration test for the constraint-rerank semantic fallback (agent.py's
`semantic_fallback_window`), using REAL sentence-transformer embeddings
against a small hand-built synthetic catalog.

Why this exists as a separate file: this is the one mechanism in this
codebase that cannot be validated against the public 200 sessions, because
every real disclosure there is a literal catalog quote (see CLAUDE.md) --
there is no local example of genuine paraphrasing to test against. This
file manufactures paraphrase scenarios deliberately, matching the risk
docs/competition_specification.md:40 leaves open for the private 800
("If natural-language paraphrasing is added by the organizer...").

Skips cleanly if sentence-transformers/torch aren't installed, matching
agent/embed_rerank.py's own lazy-import discipline -- this mechanism is
opt-in and must not force the ML dependency on anyone running the default
test suite.
"""

from __future__ import annotations

import unittest

try:
    import sentence_transformers  # noqa: F401
    HAVE_EMBEDDINGS = True
except ImportError:
    HAVE_EMBEDDINGS = False

from agent.constraint_rerank import (
    build_normalized_product_evidence,
    find_unmatched_constraints,
    rerank_by_exact_constraints,
)
from agent.memory import RawConstraint, get_attribute_value
from agent.vocab import Vocabulary, extract_negated_values

# A small synthetic catalog spanning three unrelated product types, so a
# semantic mismatch (e.g. matching the wool sweater instead of the boot)
# would be an obvious, unambiguous failure -- not a close call.
_CATALOG = {
    "boot": {
        "parent_asin": "boot",
        "title": "Trailblazer Hiking Boot",
        "features": ["100% Waterproof upper", "Rubber sole", "Reinforced toe"],
        "details": {},
        "store": "TrailCo",
        "description": [],
        "categories": ["Clothing", "Shoes", "Hiking Boots"],
    },
    "shirt": {
        "parent_asin": "shirt",
        "title": "Classic Cotton T-Shirt",
        "features": ["100% Cotton", "Machine washable", "Crew neck"],
        "details": {},
        "store": "BasicWear",
        "description": [],
        "categories": ["Clothing", "Tops", "T-Shirts"],
    },
    "sweater": {
        "parent_asin": "sweater",
        "title": "Cozy Wool Sweater",
        "features": ["100% Wool", "Hand wash only", "Ribbed cuffs"],
        "details": {},
        "store": "WarmKnits",
        "description": [],
        "categories": ["Clothing", "Sweaters"],
    },
}

_NORMALIZED_EVIDENCE = {asin: build_normalized_product_evidence(p) for asin, p in _CATALOG.items()}


@unittest.skipUnless(HAVE_EMBEDDINGS, "sentence-transformers not installed -- semantic fallback is opt-in")
class SemanticFallbackParaphraseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from agent.embed_rerank import EmbeddingReranker

        cls.reranker = EmbeddingReranker(cache_dir="C:/Hackathons/AI-Agent-TechJam26/.embed_cache_test")
        cls.reranker.build_or_load(_CATALOG)

    def _raw_bonus_for(self, clause_text: str, epsilon: float = 0.05) -> dict[str, float]:
        """The embedding fallback WITHOUT the negation-exclusion layer --
        used only to document why the exclusion layer is necessary."""
        sims = self.reranker.similarity(list(_CATALOG), clause_text)
        return {asin: epsilon * max(0.0, sim) for asin, sim in sims.items()}

    def _bonus_for(self, clause_text: str, epsilon: float = 0.05) -> dict[str, float]:
        """Mirrors Agent.respond()'s actual fallback path: deterministic
        negation exclusion first, THEN embedding similarity only among
        survivors."""
        vocabulary = Vocabulary()
        excluded = extract_negated_values(clause_text, vocabulary)
        candidates = list(_CATALOG)
        if excluded:
            candidates = [
                asin for asin in _CATALOG
                if not any(
                    get_attribute_value(_CATALOG[asin], slot, vocabulary) in values
                    for slot, values in excluded.items()
                )
            ]
        sims = self.reranker.similarity(candidates, clause_text)
        return {asin: epsilon * max(0.0, sim) for asin, sim in sims.items()}

    def test_clause_is_correctly_flagged_as_unmatched_by_exact_search(self) -> None:
        # "keeps your feet dry when it's raining" shares zero literal
        # vocabulary with any product's evidence text -- exact match must
        # find nothing, which is exactly the trigger condition for the
        # fallback to even engage.
        clause = RawConstraint(
            "keeps your feet dry when it's raining",
            "keeps your feet dry when it s raining",
            1, "preference",
        )
        unmatched = find_unmatched_constraints(list(_CATALOG), _NORMALIZED_EVIDENCE, [clause])
        self.assertEqual(unmatched, [clause])

    def test_paraphrased_waterproof_clause_promotes_the_boot_not_a_distractor(self) -> None:
        bonus = self._bonus_for("keeps your feet dry when it's raining")
        pool = ["sweater", "shirt", "boot"]  # deliberately worst-case order
        result = rerank_by_exact_constraints(pool, _NORMALIZED_EVIDENCE, [], 3, bonus=bonus)
        self.assertEqual(result[0], "boot", f"expected boot first, got order {result} (bonus={bonus})")

    def test_paraphrased_cotton_clause_promotes_the_shirt_not_a_distractor(self) -> None:
        # Positive-phrasing paraphrase (no comparison to another material) --
        # this is the case the mechanism handles correctly.
        bonus = self._bonus_for("soft and breathable, great for everyday casual wear")
        pool = ["sweater", "boot", "shirt"]
        result = rerank_by_exact_constraints(pool, _NORMALIZED_EVIDENCE, [], 3, bonus=bonus)
        self.assertEqual(result[0], "shirt", f"expected shirt first, got order {result} (bonus={bonus})")

    def test_RAW_EMBEDDING_negation_weakness_without_the_exclusion_layer(self) -> None:
        """Documents WHY the negation-exclusion layer exists, using the raw
        embedding path (_raw_bonus_for) with no exclusion applied.

        "made from a soft breathable natural fabric, not wool or synthetic"
        is a natural way a paraphrasing customer might describe cotton --
        and MiniLM ranks the WOOL sweater (0.476) above the cotton shirt
        (0.367) for it, because pooled sentence embeddings are well known to
        struggle with negation: the literal word "wool" in the clause pulls
        the embedding toward wool-adjacent products regardless of "not."
        This is not a wording fluke -- reproduced across multiple phrasings.

        This is the RAW mechanism only. Agent.respond() never calls the
        embedding this way in practice -- see the tests below for the
        actual fallback path, which runs negation exclusion first and
        gets this case right. Kept as a permanent regression/documentation
        guard: if this ever starts passing on its own (i.e. MiniLM itself
        improves at negation), that's independently interesting and worth
        noting, but the exclusion layer should stay regardless -- it's
        deterministic and free, not a stopgap for a model limitation alone.
        """
        bonus = self._raw_bonus_for("made from a soft breathable natural fabric, not wool or synthetic")
        pool = ["shirt", "boot", "sweater"]
        result = rerank_by_exact_constraints(pool, _NORMALIZED_EVIDENCE, [], 3, bonus=bonus)
        self.assertEqual(result[0], "sweater")

    def test_negation_exclusion_fixes_the_literal_material_case(self) -> None:
        # Same clause as above, through the ACTUAL fallback path (exclusion
        # then embedding among survivors) -- this is what Agent.respond()
        # does. The sweater is hard-excluded before the embedding runs, so
        # the ambiguous comparison never happens.
        bonus = self._bonus_for("made from a soft breathable natural fabric, not wool or synthetic")
        pool = ["shirt", "boot", "sweater"]
        result = rerank_by_exact_constraints(pool, _NORMALIZED_EVIDENCE, [], 3, bonus=bonus)
        self.assertEqual(result[0], "shirt", f"expected shirt first, got order {result} (bonus={bonus})")

    def test_negation_exclusion_fixes_the_category_alias_case(self) -> None:
        # The harder case: "animal fiber" never appears in MATERIAL_VOCAB at
        # all -- this only works because of MATERIAL_CATEGORY_ALIASES, not
        # literal vocabulary matching. Confirmed to still fail on the raw
        # embedding path in CLAUDE.md's iteration 7 notes (0.217 vs 0.168,
        # sweater still wins) even without saying "wool" at all.
        bonus = self._bonus_for("made of a plant-based natural material, not an animal fiber")
        pool = ["shirt", "boot", "sweater"]
        result = rerank_by_exact_constraints(pool, _NORMALIZED_EVIDENCE, [], 3, bonus=bonus)
        self.assertEqual(result[0], "shirt", f"expected shirt first, got order {result} (bonus={bonus})")

    def test_exact_match_still_wins_over_semantic_bonus_when_both_present(self) -> None:
        # The safety property: a literal match for the SWEATER combined with
        # a semantic bonus that (mis)favors the boot must still rank the
        # sweater first -- exact evidence is never allowed to lose to a
        # semantic guess, by construction of the epsilon bound.
        exact_clause = RawConstraint("100% Wool", "100 wool", 1, "preference")
        bonus = self._bonus_for("keeps your feet dry when it's raining")  # favors "boot"
        pool = ["boot", "shirt", "sweater"]
        result = rerank_by_exact_constraints(pool, _NORMALIZED_EVIDENCE, [exact_clause], 3, bonus=bonus)
        self.assertEqual(result[0], "sweater")


if __name__ == "__main__":
    unittest.main()
