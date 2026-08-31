"""Split-half robustness check for policy-constant candidates.

The full-200 policy sweep is fit and reported on the same 200 sessions, so a
narrow win there can be overfitting. This checks a candidate on each
alternating half independently (same ~40/40/15/5 scenario mix per half). A
setting is only worth adopting if it beats the incumbent on BOTH halves.

Currently checks preference_tag_bonus (the only policy constant the sweep
moved -- pool_threshold and budget_tolerance are structurally inert on this
evaluator, see CLAUDE.md).
"""

from __future__ import annotations

import sys

from agent_dev.agent import Agent
from src.retrieval import build_catalog_index
from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl

CANDIDATES = [float(a) for a in sys.argv[1:]] or [0.2, 0.35, 0.5]


def row(tag, r):
    return (f"{tag}  tech={r['recommended_technical_score']:.4f}  hit={r['hit_rate_at_10']:.4f}  "
            f"mrr={r['mrr']:.4f}  mttc={r['mttc']:.3f}")


def main() -> None:
    samples = load_jsonl("data/public_set.jsonl")
    catalog_ids, categories, products = catalog_index("data/catalog.jsonl")
    shared_index = build_catalog_index("data/catalog.jsonl")
    half_a, half_b = samples[0::2], samples[1::2]

    for bonus in CANDIDATES:
        print(f"\n=== preference_tag_bonus = {bonus} ===")
        for tag, subset in (("full ", samples), ("halfA", half_a), ("halfB", half_b)):
            agent = Agent("data/catalog.jsonl", index=shared_index, preference_tag_bonus=bonus)
            r = evaluate(agent, subset, catalog_ids, categories, products)
            print(row(tag, r))


if __name__ == "__main__":
    main()
