"""Coordinate-wise sweep of the policy constants that were never tuned:
pool_threshold (ask-vs-answer cutoff), budget_tolerance (how strictly a
stated budget is enforced), preference_tag_bonus (how much user_profile
nudges clarification choice).

Greedy coordinate search rather than a full grid (4 x 5 x 4 = 80 runs would
take well over an hour): sweep one constant at a time, keep the winner, move
to the next. Reuses one prebuilt catalog index across all runs.
"""

from __future__ import annotations

import json
from pathlib import Path

from agent.agent import Agent
from agent.retrieval import build_catalog_index
from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl

POOL_THRESHOLD_CANDIDATES = [8, 12, 15, 20, 30]
BUDGET_TOLERANCE_CANDIDATES = [1.0, 1.1, 1.15, 1.25, 1.4]
# preference_tag_bonus re-tuned to 1.5 (see scripts/tune_pref_bonus.py and the
# CLAUDE.md "iter 2" note). pool_threshold / budget_tolerance remain
# structurally inert on this evaluator.
PREFERENCE_TAG_BONUS_CANDIDATES = [0.5, 1.0, 1.25, 1.5, 2.0]

DEFAULTS = {"pool_threshold": 15, "budget_tolerance": 1.15, "preference_tag_bonus": 1.5}


def run(shared_index, samples, catalog_ids, categories, products, **kwargs) -> dict:
    params = {**DEFAULTS, **kwargs}
    agent = Agent("data/catalog.jsonl", index=shared_index, **params)
    result = evaluate(agent, samples, catalog_ids, categories, products)
    return {
        "params": params,
        "hit_rate_at_10": result["hit_rate_at_10"],
        "mrr": result["mrr"],
        "mttc": result["mttc"],
        "technical_score": result["recommended_technical_score"],
    }


def sweep(name, candidates, fixed, shared_index, samples, catalog_ids, categories, products):
    print(f"\n=== sweeping {name} (others fixed at {fixed}) ===")
    runs = []
    for value in candidates:
        kwargs = {**fixed, name: value}
        r = run(shared_index, samples, catalog_ids, categories, products, **kwargs)
        runs.append(r)
        print(f"  {name}={value!r:>6}  technical_score={r['technical_score']:.6f}  "
              f"hit={r['hit_rate_at_10']:.4f}  mrr={r['mrr']:.4f}  mttc={r['mttc']:.4f}")
    best = max(runs, key=lambda r: r["technical_score"])
    print(f"  -> best {name} = {best['params'][name]!r} (technical_score={best['technical_score']:.6f})")
    return best["params"][name], runs


def main() -> None:
    catalog_path = "data/catalog.jsonl"
    dataset_path = "data/public_set.jsonl"
    samples = load_jsonl(dataset_path)
    catalog_ids, categories, products = catalog_index(catalog_path)

    print("Building shared catalog index (once)...")
    shared_index = build_catalog_index(catalog_path)

    all_runs = []

    best_pool_threshold, runs = sweep(
        "pool_threshold", POOL_THRESHOLD_CANDIDATES, DEFAULTS,
        shared_index, samples, catalog_ids, categories, products,
    )
    all_runs.extend(runs)

    fixed = {**DEFAULTS, "pool_threshold": best_pool_threshold}
    best_budget_tolerance, runs = sweep(
        "budget_tolerance", BUDGET_TOLERANCE_CANDIDATES, fixed,
        shared_index, samples, catalog_ids, categories, products,
    )
    all_runs.extend(runs)

    fixed = {**fixed, "budget_tolerance": best_budget_tolerance}
    best_preference_tag_bonus, runs = sweep(
        "preference_tag_bonus", PREFERENCE_TAG_BONUS_CANDIDATES, fixed,
        shared_index, samples, catalog_ids, categories, products,
    )
    all_runs.extend(runs)

    final_params = {
        "pool_threshold": best_pool_threshold,
        "budget_tolerance": best_budget_tolerance,
        "preference_tag_bonus": best_preference_tag_bonus,
    }
    print(f"\n=== FINAL RECOMMENDED PARAMS: {final_params} ===")
    final_result = run(shared_index, samples, catalog_ids, categories, products, **final_params)
    print(json.dumps(final_result, indent=2))

    Path("results_policy_sweep.json").write_text(
        json.dumps({"all_runs": all_runs, "final": final_result}, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
