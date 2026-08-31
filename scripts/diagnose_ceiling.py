"""Phase 1 ceiling diagnostic: is the remaining HR@10 gap a RETRIEVAL problem
(gold never in the candidate pool) or a RANKING problem (gold in the pool,
ranked >10)?

Changes NO agent behaviour. Replays all 200 public sessions with the exact
evaluator simulated-customer protocol (same break-on-hit semantics, same
intent_override gate) but drives the loop manually so it can read
agent._sessions[sid].last_candidate_pool -- the full post-rerank pool of up
to 200 products, of which the scored recommendations are just the first 10.

Per session it records:
  pool_recall        gold in last_candidate_pool on ANY turn
  first_pool_turn    first turn gold entered the pool (1-indexed) or None
  best_rank          best (lowest) rank gold ever reached, UNBOUNDED, or None
  final_rank         gold's rank in the last turn's pool, or None
  converged          gold reached top-10 (== evaluator "hit")
  turns_to_convergence
  scenario

Run:  python3 -m scripts.diagnose_ceiling            # both bonus=0.2 and 1.5
      python3 -m scripts.diagnose_ceiling --bonus 1.5
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from agent_dev.agent import Agent
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

CATALOG = "data/catalog.jsonl"
DATASET = "data/public_set.jsonl"


def replay_session(agent, sample, catalog_ids, categories, products) -> dict:
    session_id = f"ceil_{sample['sample_id']}"
    agent.reset(session_id, sample["user_profile"])
    target = str(sample["ground_truth"]["parent_asin"])
    card, behavior = materialize_hidden_fields(sample, products)
    eff = {**sample, "intent_card": card, "behavior": behavior}
    disclosed: set[str] = set()
    boundary_used = False
    override_applied = sample["scenario_type"] != "intent_override"
    user_message = initial_message(eff, coarse_category(categories.get(target, [])), disclosed)

    first_pool_turn = None
    best_rank = None
    final_rank = None
    converged_turn = None

    for turn in range(1, MAX_TURNS + 1):
        response = agent.respond(session_id, user_message, turn, TOP_K)
        ranked_top10 = normalize_recommendations(response.get("recommendations"), catalog_ids)
        pool = [str(p["parent_asin"]) for p in agent._sessions[session_id].last_candidate_pool]

        rank_this_turn = (pool.index(target) + 1) if target in pool else None
        final_rank = rank_this_turn if rank_this_turn is not None else final_rank
        if rank_this_turn is not None:
            first_pool_turn = first_pool_turn or turn
            best_rank = rank_this_turn if best_rank is None else min(best_rank, rank_this_turn)

        if override_applied and target in ranked_top10:
            converged_turn = turn
            # evaluator scores rank at this turn and stops
            final_rank = ranked_top10.index(target) + 1
            best_rank = min(best_rank, final_rank) if best_rank else final_rank
            break
        if turn == MAX_TURNS:
            break

        override = eff.get("behavior", {}).get("override") or {}
        if not override_applied and turn + 1 == int(override.get("turn", 3)):
            override_applied = True
            new_value = str(override.get("new_value", ""))
            if new_value:
                disclosed.add(new_value)
            user_message = str(override.get("message", "Actually, please ignore my earlier preference."))
        else:
            user_message, boundary_used = customer_reply(
                eff, response.get("ask_attribute"), disclosed, boundary_used
            )

    return {
        "sample_id": sample["sample_id"],
        "scenario": sample["scenario_type"],
        "pool_recall": first_pool_turn is not None,
        "first_pool_turn": first_pool_turn,
        "best_rank": best_rank,
        "final_rank": final_rank,
        "converged": converged_turn is not None,
        "turns_to_convergence": converged_turn,
        "reciprocal_rank": (1.0 / final_rank) if (converged_turn and final_rank) else 0.0,
    }


def _hist(values, edges, labels):
    out = Counter()
    for v in values:
        for (lo, hi), lab in zip(edges, labels):
            if lo <= v <= hi:
                out[lab] += 1
                break
    return {lab: out.get(lab, 0) for lab in labels}


def report(rows: list[dict], tag: str) -> None:
    n = len(rows)
    hits = [r for r in rows if r["converged"]]
    misses = [r for r in rows if not r["converged"]]
    in_pool = [r for r in rows if r["pool_recall"]]

    cur_hr = len(hits) / n
    oracle_hr = len(in_pool) / n
    cur_mrr = statistics.fmean(
        (1.0 / r["final_rank"]) if r["converged"] and r["final_rank"] else 0.0 for r in rows
    )
    oracle_mrr_hits_at1 = len(hits) / n
    oracle_mrr_pool_at1 = len(in_pool) / n

    miss_never_pool = [r for r in misses if not r["pool_recall"]]
    miss_in_pool = [r for r in misses if r["pool_recall"]]

    print(f"\n{'='*78}\n{tag}   (n={n})\n{'='*78}")
    print(f"  current   HR@10 = {cur_hr:.4f}   ({len(hits)} hits / {len(misses)} misses)")
    print(f"  current   MRR   = {cur_mrr:.4f}")
    print(f"  pool recall@200 = {oracle_hr:.4f}   ({len(in_pool)}/{n} sessions had gold in the pool on some turn)")
    print(f"  ORACLE HR@10    = {oracle_hr:.4f}   (ceiling for any pure reranker on the current ask path)")
    print(f"  ORACLE MRR      = {oracle_mrr_pool_at1:.4f}   (every in-pool session ranked #1)")
    print(f"            (vs {oracle_mrr_hits_at1:.4f} if only current hits were all ranked #1)")

    print(f"\n  -- the {len(misses)} current misses --")
    print(f"     gold NEVER in pool (retrieval failure) : {len(miss_never_pool):2d}"
          f"  ({len(miss_never_pool)/max(1,len(misses)):.0%} of misses)")
    print(f"     gold in pool but ranked >10 (ranking)  : {len(miss_in_pool):2d}"
          f"  ({len(miss_in_pool)/max(1,len(misses)):.0%} of misses)")
    if miss_in_pool:
        br = [r["best_rank"] for r in miss_in_pool]
        h = _hist(br, [(11, 20), (21, 50), (51, 100), (101, 200)],
                  ["rank 11-20", "rank 21-50", "rank 51-100", "rank 101-200"])
        print(f"       best_rank distribution of those {len(miss_in_pool)}: {h}")
        print(f"       best_rank  min={min(br)}  median={int(statistics.median(br))}  max={max(br)}")
        fpt = Counter(r["first_pool_turn"] for r in miss_in_pool)
        print(f"       first_pool_turn (was it retrievable from the start?): "
              f"{dict(sorted(fpt.items()))}")
        from_t1 = sum(1 for r in miss_in_pool if r["first_pool_turn"] == 1)
        print(f"       -> {from_t1}/{len(miss_in_pool)} were in the pool already on turn 1 "
              f"(never-retrievable vs session-too-short)")

    print(f"\n  -- the {len(hits)} current hits --")
    fr = [r["final_rank"] for r in hits if r["final_rank"]]
    h = _hist(fr, [(1, 1), (2, 3), (4, 10)], ["rank 1", "rank 2-3", "rank 4-10"])
    print(f"     final rank distribution: {h}")
    print(f"     {h['rank 1']}/{len(hits)} hits are already at rank 1")
    ttc = [r["turns_to_convergence"] for r in hits]
    print(f"     turns_to_convergence: mean={statistics.fmean(ttc):.2f}  "
          f"hist={dict(sorted(Counter(ttc).items()))}")

    # per scenario
    print(f"\n  -- by scenario --")
    by = defaultdict(list)
    for r in rows:
        by[r["scenario"]].append(r)
    print(f"     {'scenario':16s} {'n':>3} {'curHR':>7} {'oracleHR':>9} {'gap':>6} "
          f"{'never_pool':>11} {'rank>10':>8}")
    for sc in sorted(by):
        rs = by[sc]
        nn = len(rs)
        ch = sum(r["converged"] for r in rs) / nn
        oh = sum(r["pool_recall"] for r in rs) / nn
        nm = [r for r in rs if not r["converged"]]
        npool = sum(1 for r in nm if not r["pool_recall"])
        nrank = sum(1 for r in nm if r["pool_recall"])
        print(f"     {sc:16s} {nn:>3} {ch:>7.3f} {oh:>9.3f} {oh-ch:>6.3f} "
              f"{npool:>11} {nrank:>8}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bonus", type=float, default=None,
                    help="single preference_tag_bonus; default runs 0.2 and 1.5")
    ap.add_argument("--out", default="runs/ceiling_diagnostic.json")
    args = ap.parse_args()

    samples = load_jsonl(DATASET)
    catalog_ids, categories, products = catalog_index(CATALOG)

    bonuses = [args.bonus] if args.bonus is not None else [0.2, 1.5]
    all_out = {}
    for b in bonuses:
        agent = Agent(CATALOG, preference_tag_bonus=b)
        rows = [replay_session(agent, s, catalog_ids, categories, products) for s in samples]
        report(rows, f"preference_tag_bonus = {b}")
        all_out[str(b)] = rows

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(all_out, indent=2) + "\n", encoding="utf-8")
    print(f"\nper-session rows -> {args.out}")


if __name__ == "__main__":
    main()
