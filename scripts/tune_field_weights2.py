"""Task 1 (iter 2): finer field-weight search around the sweep winner
"boost store + title, cut description" (7,4,2.5,2,4,0.5), plus a split-half
robustness check of that winner vs the incumbent (5,6,2.5,2,3,1).

Column order: title, categories, features, details, store, description.
"""

from __future__ import annotations

import json
from pathlib import Path

from agent_dev.agent import Agent
from src.retrieval import build_catalog_index
from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl

INCUMBENT = (5.0, 6.0, 2.5, 2.0, 3.0, 1.0)
SWEEP_WINNER = (7.0, 4.0, 2.5, 2.0, 4.0, 0.5)

CANDIDATES: dict[str, tuple] = {
    "incumbent (5,6,2.5,2,3,1)": INCUMBENT,
    "sweep winner (7,4,2.5,2,4,0.5)": SWEEP_WINNER,
    "winner +cat (7,5,2.5,2,4,0.5)": (7.0, 5.0, 2.5, 2.0, 4.0, 0.5),
    "winner cat=6 (7,6,2.5,2,4,0.5)": (7.0, 6.0, 2.5, 2.0, 4.0, 0.5),
    "winner store=5 (7,4,2.5,2,5,0.5)": (7.0, 4.0, 2.5, 2.0, 5.0, 0.5),
    "winner store=3 (7,4,2.5,2,3,0.5)": (7.0, 4.0, 2.5, 2.0, 3.0, 0.5),
    "winner title=6 (6,4,2.5,2,4,0.5)": (6.0, 4.0, 2.5, 2.0, 4.0, 0.5),
    "winner title=8 (8,4,2.5,2,4,0.5)": (8.0, 4.0, 2.5, 2.0, 4.0, 0.5),
    "winner desc=1 (7,4,2.5,2,4,1)": (7.0, 4.0, 2.5, 2.0, 4.0, 1.0),
    "winner feat=3 (7,4,3,2,4,0.5)": (7.0, 4.0, 3.0, 2.0, 4.0, 0.5),
    "winner det=1 (7,4,2.5,1,4,0.5)": (7.0, 4.0, 2.5, 1.0, 4.0, 0.5),
    "hi title+store (7,4,2,2,4,0.5)": (7.0, 4.0, 2.0, 2.0, 4.0, 0.5),
    "cat+store (5,5,2.5,2,4,0.75)": (5.0, 5.0, 2.5, 2.0, 4.0, 0.75),
}


def summ(result):
    return dict(
        hit=result["hit_rate_at_10"],
        mrr=round(result["mrr"], 5),
        mttc=result["mttc"],
        tech=result["recommended_technical_score"],
        by={k: (v["hit_rate_at_10"], round(v["mrr"], 4)) for k, v in result["scenario_metrics"].items()},
    )


def main() -> None:
    samples = load_jsonl("data/public_set.jsonl")
    catalog_ids, categories, products = catalog_index("data/catalog.jsonl")
    shared_index = build_catalog_index("data/catalog.jsonl")

    half_a = samples[0::2]
    half_b = samples[1::2]

    print("=== FULL 200 ===")
    full = []
    for name, w in CANDIDATES.items():
        r = evaluate(Agent("data/catalog.jsonl", index=shared_index, field_weights=w), samples,
                     catalog_ids, categories, products)
        full.append((name, w, summ(r)))
        s = summ(r)
        print(f"{s['tech']:.6f}  hit={s['hit']:.4f}  mrr={s['mrr']:.5f}  mttc={s['mttc']:.3f}  {name}")
        print(f"          scenarios={s['by']}")

    print("\n=== SPLIT-HALF (incumbent vs sweep winner) ===")
    for name, w in [("incumbent", INCUMBENT), ("sweep winner", SWEEP_WINNER)]:
        ra = evaluate(Agent("data/catalog.jsonl", index=shared_index, field_weights=w), half_a,
                      catalog_ids, categories, products)
        rb = evaluate(Agent("data/catalog.jsonl", index=shared_index, field_weights=w), half_b,
                      catalog_ids, categories, products)
        print(f"{name:14s} half_a tech={ra['recommended_technical_score']:.4f} hit={ra['hit_rate_at_10']:.4f} "
              f"mrr={ra['mrr']:.4f}  |  half_b tech={rb['recommended_technical_score']:.4f} "
              f"hit={rb['hit_rate_at_10']:.4f} mrr={rb['mrr']:.4f}")

    Path("runs").mkdir(exist_ok=True)
    Path("runs/field_weights2.json").write_text(
        json.dumps([(n, list(w), s) for n, w, s in full], indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
