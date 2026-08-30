"""One-off: dump the full turn-by-turn transcript for every Boundary session
that misses, so the 4 remaining Boundary misses can be read by hand.

Reimplements the evaluator's per-session loop (identical simulated-customer
protocol) but prints, per turn: the user message, the agent's ask_attribute,
the candidate-pool size, the filled slots, and where the target sits in the
pool / top-10.
"""

from __future__ import annotations

from agent.agent import Agent
from agent import memory
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
    samples = load_jsonl("data/public_set.jsonl")
    catalog_ids, categories, products = catalog_index(catalog_path)
    agent = Agent(catalog_path)

    boundary = [s for s in samples if s["scenario_type"] == "boundary"]
    print(f"{len(boundary)} boundary sessions\n")

    for sample in boundary:
        session_id = f"trace_{sample['sample_id']}"
        agent.reset(session_id, sample["user_profile"])
        target = str(sample["ground_truth"]["parent_asin"])
        card, behavior = materialize_hidden_fields(sample, products)
        eff = {**sample, "intent_card": card, "behavior": behavior}
        disclosed: set[str] = set()
        boundary_used = False
        override_applied = True
        user_message = initial_message(eff, coarse_category(categories.get(target, [])), disclosed)

        transcript = []
        hit_turn = None
        for turn in range(1, MAX_TURNS + 1):
            response = agent.respond(session_id, user_message, turn, TOP_K)
            ranked = normalize_recommendations(response.get("recommendations"), catalog_ids)
            state = agent._sessions[session_id]
            pool = [str(p["parent_asin"]) for p in state.last_candidate_pool]
            pool_rank = pool.index(target) + 1 if target in pool else None
            top_rank = ranked.index(target) + 1 if target in ranked else None
            filled = memory.get_filled_slots(state)
            transcript.append({
                "turn": turn,
                "user_message": user_message,
                "ask_attribute": response.get("ask_attribute"),
                "pool_size": len(pool),
                "pool_rank": pool_rank,
                "top10_rank": top_rank,
                "filled_slots": dict(filled),
            })
            if top_rank is not None:
                hit_turn = turn
                break
            if turn == MAX_TURNS:
                break
            user_message, boundary_used = customer_reply(
                eff, response.get("ask_attribute"), disclosed, boundary_used
            )

        status = f"HIT@turn{hit_turn}" if hit_turn else "*** MISS ***"
        tgt = products[target]
        print("=" * 100)
        print(f"{sample['sample_id']}  {status}  difficulty={sample.get('difficulty_bucket')}")
        print(f"  target={target}  title={str(tgt.get('title'))[:90]!r}")
        print(f"  target categories={tgt.get('categories')}")
        print(f"  target price={tgt.get('price')}  store={tgt.get('store')!r}")
        print(f"  intent_card hard={card.get('hard_constraints')}")
        print(f"              soft={card.get('soft_preferences')}")
        print(f"  user_profile preference_tags={sample['user_profile'].get('preference_tags')}")
        for row in transcript:
            print(f"  T{row['turn']}: ask={row['ask_attribute']!r:>14}  pool={row['pool_size']:>3}  "
                  f"pool_rank={row['pool_rank']}  top10_rank={row['top10_rank']}")
            print(f"       msg={row['user_message'][:95]!r}")
            print(f"       filled={row['filled_slots']}")
        print()


if __name__ == "__main__":
    main()
