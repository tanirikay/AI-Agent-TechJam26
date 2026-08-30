"""Task 2: does a narrowing pass on top of the OR recall pool pay off?

Sweeps the `narrow_mode` / `narrow_param` / `narrow_recommendations` knobs
added to agent.Agent, against the 200 public sessions, reusing one shared
FTS5 index. Also records the narrowed pool size per turn so we can see
whether `pool_threshold` engages at all under each configuration.

Config 1 is the no-narrowing control and MUST reproduce the current score
exactly -- it is the regression check that the new code paths are inert
when disabled.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from agent.agent import Agent
from agent.retrieval import build_catalog_index
from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl

CANDIDATES: list[dict] = [
    {"name": "control: no narrowing", "kwargs": {}},
    {"name": "score cutoff 0.30 (decision only)", "kwargs": {"narrow_mode": "score", "narrow_param": 0.30}},
    {"name": "score cutoff 0.50 (decision only)", "kwargs": {"narrow_mode": "score", "narrow_param": 0.50}},
    {"name": "score cutoff 0.70 (decision only)", "kwargs": {"narrow_mode": "score", "narrow_param": 0.70}},
    {"name": "score cutoff 0.85 (decision only)", "kwargs": {"narrow_mode": "score", "narrow_param": 0.85}},
    {"name": "score cutoff 0.50 (+recommendations)", "kwargs": {"narrow_mode": "score", "narrow_param": 0.50, "narrow_recommendations": True}},
    {"name": "conjunction on slots (decision only)", "kwargs": {"narrow_mode": "conjunction"}},
    {"name": "conjunction on slots (+recommendations)", "kwargs": {"narrow_mode": "conjunction", "narrow_recommendations": True}},
]


class SizeLoggingAgent(Agent):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.decision_sizes: list[int] = []
        self.asked_turns = 0
        self.answered_turns = 0

    def respond(self, session_id, user_message, turn, top_k):
        out = super().respond(session_id, user_message, turn, top_k)
        # recompute the decision-pool size the same way respond() did, from
        # what respond() stored, without duplicating retrieval work
        self.decision_sizes.append(self._last_decision_size)
        if out["ask_attribute"] is None:
            self.answered_turns += 1
        else:
            self.asked_turns += 1
        return out


def main() -> None:
    catalog_path = "data/catalog.jsonl"
    dataset_path = "data/public_set.jsonl"

    samples = load_jsonl(dataset_path)
    catalog_ids, categories, products = catalog_index(catalog_path)
    print("Building shared catalog index (once)...")
    shared_index = build_catalog_index(catalog_path)

    results = []
    for candidate in CANDIDATES:
        name = candidate["name"]
        kwargs = candidate["kwargs"]
        print(f"\n=== {name} ===")
        agent = SizeLoggingAgent(catalog_path, index=shared_index, **kwargs)
        result = evaluate(agent, samples, catalog_ids, categories, products)
        sizes = agent.decision_sizes
        summary = {
            "name": name,
            "kwargs": kwargs,
            "hit_rate_at_10": result["hit_rate_at_10"],
            "mrr": result["mrr"],
            "mttc": result["mttc"],
            "technical_score": result["recommended_technical_score"],
            "turns": len(sizes),
            "decision_pool_min": min(sizes) if sizes else None,
            "decision_pool_mean": round(sum(sizes) / len(sizes), 2) if sizes else None,
            "turns_at_or_below_pool_threshold": sum(1 for s in sizes if s <= agent.pool_threshold),
            "turns_agent_chose_not_to_ask": agent.answered_turns,
            "decision_pool_histogram_head": dict(sorted(Counter(sizes).items())[:10]),
            "scenario_metrics": result["scenario_metrics"],
        }
        results.append(summary)
        print(json.dumps({k: v for k, v in summary.items() if k != "scenario_metrics"}, indent=2))

    results.sort(key=lambda r: r["technical_score"], reverse=True)
    print("\n=== RANKED BY technical_score ===")
    for r in results:
        print(
            f"{r['technical_score']:.6f}  hit={r['hit_rate_at_10']:.4f}  mrr={r['mrr']:.4f}  "
            f"mttc={r['mttc']:.4f}  below_thr={r['turns_at_or_below_pool_threshold']:>4}  {r['name']}"
        )

    Path("runs").mkdir(exist_ok=True)
    Path("runs/pool_narrowing_sweep.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
