"""Sweep BM25 per-field weights over the 200 public sessions.

Builds the ~50k-row FTS5 index once and reuses it across candidate weight
vectors (each candidate still needs a fresh Agent so its session state
starts clean, but the expensive index build only happens once).

Column order: title, categories, features, details, store, description.
"""

from __future__ import annotations

import json
from pathlib import Path

from agent_dev.agent import Agent
from src.retrieval import build_catalog_index
from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl

# name -> (title, categories, features, details, store, description)
CANDIDATES: dict[str, tuple[float, float, float, float, float, float]] = {
    "baseline (starter defaults)": (6.0, 4.0, 2.5, 2.5, 1.5, 1.0),
    "boost store": (6.0, 4.0, 2.5, 2.5, 4.0, 1.0),
    "boost store + title, cut description": (7.0, 4.0, 2.5, 2.0, 4.0, 0.5),
    "flatten weights": (4.0, 4.0, 3.0, 3.0, 3.0, 2.0),
    "title-dominant": (8.0, 3.0, 2.0, 2.0, 2.0, 0.5),
    "categories-dominant": (5.0, 6.0, 2.5, 2.0, 3.0, 1.0),
    "store-dominant": (5.0, 3.0, 2.0, 2.0, 6.0, 1.0),
}


def main() -> None:
    catalog_path = "data/catalog.jsonl"
    dataset_path = "data/public_set.jsonl"

    samples = load_jsonl(dataset_path)
    catalog_ids, categories, products = catalog_index(catalog_path)

    print("Building shared catalog index (once)...")
    shared_index = build_catalog_index(catalog_path)

    results = []
    for name, weights in CANDIDATES.items():
        print(f"\n=== {name} {weights} ===")
        agent = Agent(catalog_path, index=shared_index, field_weights=weights)
        result = evaluate(agent, samples, catalog_ids, categories, products)
        summary = {
            "name": name,
            "weights": weights,
            "hit_rate_at_10": result["hit_rate_at_10"],
            "mrr": result["mrr"],
            "mttc": result["mttc"],
            "technical_score": result["recommended_technical_score"],
            "scenario_metrics": result["scenario_metrics"],
        }
        results.append(summary)
        print(json.dumps({k: v for k, v in summary.items() if k != "scenario_metrics"}, indent=2))

    results.sort(key=lambda r: r["technical_score"], reverse=True)
    print("\n=== RANKED BY technical_score ===")
    for r in results:
        print(f"{r['technical_score']:.6f}  hit={r['hit_rate_at_10']:.4f}  mrr={r['mrr']:.4f}  "
              f"mttc={r['mttc']:.4f}  {r['name']} {r['weights']}")

    Path("results_weight_sweep.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
