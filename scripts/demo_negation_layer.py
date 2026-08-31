"""Live demo: the always-on negation-exclusion layer (agent.py's
_apply_negation_exclusion), against a small hand-built catalog.

This mechanism never fires on the public 200 sessions (every real
disclosure there is a literal catalog quote with no negation), so this
demo exists specifically to prove it works, using self-produced test data
-- same catalog used in tests/test_semantic_fallback_integration.py.

Usage:
    python -m scripts.demo_negation_layer
    python -m scripts.demo_negation_layer --auto
"""

from __future__ import annotations

import argparse
import sqlite3

from agent import memory
from agent.agent import Agent
from agent.constraint_rerank import build_normalized_product_evidence
from agent.retrieval import CatalogIndex
from agent.vocab import Vocabulary, extract_negated_values

CATALOG = {
    "boot": {
        "parent_asin": "boot", "title": "Trailblazer Hiking Boot",
        "features": ["100% Waterproof upper", "Rubber sole", "Reinforced toe"],
        "details": {}, "store": "TrailCo", "description": [],
        "categories": ["Clothing", "Shoes", "Hiking Boots"], "price": 89.0,
    },
    "shirt": {
        "parent_asin": "shirt", "title": "Classic Cotton T-Shirt",
        "features": ["100% Cotton", "Machine washable", "Crew neck"],
        "details": {}, "store": "BasicWear", "description": [],
        "categories": ["Clothing", "Tops", "T-Shirts"], "price": 19.0,
    },
    "sweater": {
        "parent_asin": "sweater", "title": "Cozy Wool Sweater",
        "features": ["100% Wool", "Hand wash only", "Ribbed cuffs"],
        "details": {}, "store": "WarmKnits", "description": [],
        "categories": ["Clothing", "Sweaters"], "price": 55.0,
    },
}


def build_demo_index() -> CatalogIndex:
    conn = sqlite3.connect(":memory:")
    cur = conn.cursor()
    cur.execute(
        "CREATE VIRTUAL TABLE products USING fts5("
        "parent_asin UNINDEXED, title, categories, features, details, store, description, "
        "tokenize='unicode61 remove_diacritics 2')"
    )
    for asin, p in CATALOG.items():
        cur.execute(
            "INSERT INTO products VALUES (?,?,?,?,?,?,?)",
            (asin, p["title"], " ".join(p["categories"]), " ".join(p["features"]), "", p["store"], ""),
        )
    conn.commit()
    normalized_evidence = {a: build_normalized_product_evidence(p) for a, p in CATALOG.items()}
    return CatalogIndex(connection=conn, products=CATALOG, vocabulary=Vocabulary(), normalized_evidence=normalized_evidence)


def pause(auto: bool) -> None:
    if not auto:
        input("\n[press Enter to continue]\n")
    else:
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Live negation-exclusion demo")
    parser.add_argument("--auto", action="store_true")
    args = parser.parse_args()

    print("=" * 78)
    print("DEMO CATALOG (self-produced -- this scenario doesn't occur in the")
    print("public 200 sessions, so we built it to prove the mechanism directly)")
    print("=" * 78)
    for asin, p in CATALOG.items():
        print(f"  [{asin}] {p['title']}  -  {p['features'][0]}")
    pause(args.auto)

    index = build_demo_index()
    agent = Agent(index=index)
    sid = "negation_demo"
    agent.reset(sid, {})

    print("-" * 78)
    print("TURN 1")
    print("-" * 78)
    msg1 = "I'm looking for a top."
    print(f'CUSTOMER: "{msg1}"')
    agent.respond(sid, msg1, 1, 10)
    state = agent._sessions[sid]
    # Simulate having just asked about material, so the reply below is
    # captured as a clarification-shaped raw constraint.
    state.pending_ask = "material"
    state.slots["material"].asked = True
    pause(args.auto)

    print("-" * 78)
    print("TURN 2")
    print("-" * 78)
    msg2 = "For that, what matters is: not wool or synthetic."
    print(f'CUSTOMER: "{msg2}"')
    memory.update_state(state, msg2, 2, agent.index.vocabulary)

    excluded = extract_negated_values(msg2, agent.index.vocabulary)
    print(f"\nDetected exclusion (deterministic, no ML): {excluded}")

    pool_before = ["sweater", "boot", "shirt"]
    pool_after = agent._apply_negation_exclusion(pool_before, state)
    print(f"\nCandidate pool BEFORE negation exclusion: {pool_before}")
    print(f"Candidate pool AFTER negation exclusion:  {pool_after}")
    print("\n-> the wool sweater is not deleted, it's demoted: sunk to the bottom,")
    print("   stable order preserved among everything else, because its own")
    print("   material (wool) matches what the customer explicitly ruled out.")
    print("=" * 78)


if __name__ == "__main__":
    main()
