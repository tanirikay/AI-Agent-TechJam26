"""Task 1 (iter 2): the policy sweep found technical_score rising monotonically
with preference_tag_bonus across the whole swept range (0.0..0.5), pinned at
the edge. This extends the sweep upward to find the actual peak, then
split-half validates the best candidate against the incumbent (0.2).

preference_tag_bonus is the additive bonus in
memory.select_clarification_attribute: score += (#user preference_tags that
map to this slot) * bonus. Higher = clarification order follows the profile
more aggressively vs. entropy*coverage.
"""

from __future__ import annotations

import json
from pathlib import Path

from agent_dev.agent import Agent
from src.retrieval import build_catalog_index
from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl

FULL_CANDIDATES = [0.2, 0.5, 0.6, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]
SPLIT_CANDIDATES = [0.2, 0.5, 0.75, 1.0]


def s(r):
    return (r["hit_rate_at_10"], round(r["mrr"], 5), r["mttc"], r["recommended_technical_score"],
            {k: (v["hit_rate_at_10"], round(v["mrr"], 4), v["mttc"]) for k, v in r["scenario_metrics"].items()})


def main() -> None:
    samples = load_jsonl("data/public_set.jsonl")
    catalog_ids, categories, products = catalog_index("data/catalog.jsonl")
    idx = build_catalog_index("data/catalog.jsonl")
    half_a, half_b = samples[0::2], samples[1::2]

    print("=== FULL 200 ===")
    rows = []
    for b in FULL_CANDIDATES:
        r = evaluate(Agent("data/catalog.jsonl", index=idx, preference_tag_bonus=b), samples,
                     catalog_ids, categories, products)
        hit, mrr, mttc, tech, by = s(r)
        rows.append({"bonus": b, "hit": hit, "mrr": mrr, "mttc": mttc, "tech": tech, "by": by})
        print(f"bonus={b:<4}  tech={tech:.6f}  hit={hit:.4f}  mrr={mrr:.5f}  mttc={mttc:.3f}")
        print(f"           {by}")

    print("\n=== SPLIT-HALF ===")
    for b in SPLIT_CANDIDATES:
        ra = evaluate(Agent("data/catalog.jsonl", index=idx, preference_tag_bonus=b), half_a,
                      catalog_ids, categories, products)
        rb = evaluate(Agent("data/catalog.jsonl", index=idx, preference_tag_bonus=b), half_b,
                      catalog_ids, categories, products)
        print(f"bonus={b:<4}  A: tech={ra['recommended_technical_score']:.4f} hit={ra['hit_rate_at_10']:.4f} "
              f"mrr={ra['mrr']:.4f} mttc={ra['mttc']:.3f}   "
              f"B: tech={rb['recommended_technical_score']:.4f} hit={rb['hit_rate_at_10']:.4f} "
              f"mrr={rb['mrr']:.4f} mttc={rb['mttc']:.3f}")

    Path("runs").mkdir(exist_ok=True)
    Path("runs/pref_bonus.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
