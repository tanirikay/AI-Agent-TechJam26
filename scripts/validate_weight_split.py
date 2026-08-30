"""Split-half robustness check for the field-weight sweep's winner.

The weight sweep picked "flatten weights" by evaluating on the same 200
public sessions we also report scores on -- real overfitting risk. This
splits those 200 into two halves (alternating sample_id, so each half keeps
the same ~40/40/15/5% scenario mix) and checks whether "flatten" still beats
the starter's untuned defaults on EACH half independently, not just on
average. If it wins both halves, that's real signal, not sweep noise.
"""

from __future__ import annotations

from agent.agent import Agent
from agent.retrieval import build_catalog_index
from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl

CANDIDATES = {
    "starter defaults": (6.0, 4.0, 2.5, 2.5, 1.5, 1.0),
    "flatten (winner)": (4.0, 4.0, 3.0, 3.0, 3.0, 2.0),
}


def main() -> None:
    samples = load_jsonl("data/public_set.jsonl")
    catalog_ids, categories, products = catalog_index("data/catalog.jsonl")
    shared_index = build_catalog_index("data/catalog.jsonl")

    half_a = samples[0::2]
    half_b = samples[1::2]
    print(f"half_a n={len(half_a)}  half_b n={len(half_b)}")

    for name, weights in CANDIDATES.items():
        agent_a = Agent("data/catalog.jsonl", index=shared_index, field_weights=weights)
        r_a = evaluate(agent_a, half_a, catalog_ids, categories, products)
        agent_b = Agent("data/catalog.jsonl", index=shared_index, field_weights=weights)
        r_b = evaluate(agent_b, half_b, catalog_ids, categories, products)
        print(f"{name:20s} half_a: tech={r_a['recommended_technical_score']:.4f} hit={r_a['hit_rate_at_10']:.4f} "
              f"mrr={r_a['mrr']:.4f}   half_b: tech={r_b['recommended_technical_score']:.4f} "
              f"hit={r_b['hit_rate_at_10']:.4f} mrr={r_b['mrr']:.4f}")


if __name__ == "__main__":
    main()
