"""Measure the recall-in-pool vs recall-in-top-10 gap.

Answers: when a session misses, was the target sitting somewhere in the
~200-candidate BM25 pool (a ranking problem a reranker could fix) or was it
never retrieved at all (a recall problem a reranker can't fix)? This decides
whether building a semantic reranker is worth the effort before building it.

Reimplements the evaluator's per-session protocol (so the simulated customer
behaves identically) but drives the loop manually so it can inspect
agent._sessions[session_id].last_candidate_pool -- the full pre-top-10 pool
-- after every turn, not just the top 10 the Agent contract exposes.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from agent.agent import Agent
from evaluator.local_evaluator import (
    MAX_TURNS,
    TOP_K,
    catalog_index,
    coarse_category,
    customer_reply,
    initial_message,
    load_jsonl,
    materialize_hidden_fields,
    normalize_recommendations,
)


def main() -> None:
    catalog_path = "data/catalog.jsonl"
    dataset_path = "data/public_set.jsonl"

    samples = load_jsonl(dataset_path)
    catalog_ids, categories, products = catalog_index(catalog_path)
    agent = Agent(catalog_path)

    scenario_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "hit_top10": 0, "hit_pool": 0})
    session_details = []

    for sample in samples:
        session_id = f"diag_{sample['sample_id']}"
        agent.reset(session_id, sample["user_profile"])
        target = str(sample["ground_truth"]["parent_asin"])
        card, behavior = materialize_hidden_fields(sample, products)
        effective_sample = {**sample, "intent_card": card, "behavior": behavior}
        disclosed: set[str] = set()
        boundary_used = False
        override_applied = sample["scenario_type"] != "intent_override"
        user_message = initial_message(effective_sample, coarse_category(categories.get(target, [])), disclosed)

        hit_top10_turn = None
        hit_pool_turn = None

        for turn in range(1, MAX_TURNS + 1):
            response = agent.respond(session_id, user_message, turn, TOP_K)
            ranked = normalize_recommendations(response.get("recommendations"), catalog_ids)
            pool = agent._sessions[session_id].last_candidate_pool
            pool_asins = {str(p["parent_asin"]) for p in pool}

            if hit_top10_turn is None and target in ranked:
                hit_top10_turn = turn
            if hit_pool_turn is None and target in pool_asins:
                hit_pool_turn = turn

            if hit_top10_turn is not None:
                break
            if turn == MAX_TURNS:
                break

            override = effective_sample.get("behavior", {}).get("override") or {}
            if not override_applied and turn + 1 == int(override.get("turn", 3)):
                override_applied = True
                new_value = str(override.get("new_value", ""))
                if new_value:
                    disclosed.add(new_value)
                user_message = str(override.get("message", "Actually, please ignore my earlier preference."))
            else:
                user_message, boundary_used = customer_reply(
                    effective_sample, response.get("ask_attribute"), disclosed, boundary_used
                )

        scenario = sample["scenario_type"]
        scenario_stats[scenario]["n"] += 1
        scenario_stats[scenario]["hit_top10"] += int(hit_top10_turn is not None)
        scenario_stats[scenario]["hit_pool"] += int(hit_pool_turn is not None)
        session_details.append({
            "sample_id": sample["sample_id"],
            "scenario_type": scenario,
            "hit_top10_turn": hit_top10_turn,
            "hit_pool_turn": hit_pool_turn,
            # in pool at some point, but never surfaced into the scored top 10
            "rerank_recoverable": hit_top10_turn is None and hit_pool_turn is not None,
        })

    overall_n = len(samples)
    overall_top10 = sum(1 for s in session_details if s["hit_top10_turn"] is not None)
    overall_pool = sum(1 for s in session_details if s["hit_pool_turn"] is not None)
    overall_recoverable = sum(1 for s in session_details if s["rerank_recoverable"])

    print(f"Overall (n={overall_n}):")
    print(f"  recall@top10 (current hit rate) = {overall_top10 / overall_n:.4f}")
    print(f"  recall@pool (~200 candidates)   = {overall_pool / overall_n:.4f}")
    print(f"  reranker-recoverable misses     = {overall_recoverable} sessions "
          f"({overall_recoverable / overall_n:.4f} of all sessions, "
          f"{overall_recoverable / (overall_n - overall_top10):.4f} of current misses)")

    print("\nBy scenario:")
    for scenario in sorted(scenario_stats):
        stats = scenario_stats[scenario]
        n = stats["n"]
        print(f"  {scenario:16s} n={n:3d}  top10={stats['hit_top10']/n:.4f}  pool={stats['hit_pool']/n:.4f}  "
              f"gap={stats['hit_pool']/n - stats['hit_top10']/n:.4f}")

    Path("results_recall_gap.json").write_text(
        json.dumps({"sessions": session_details, "scenario_stats": dict(scenario_stats)}, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
