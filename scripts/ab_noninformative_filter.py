"""iter 3 Phase 2: A/B the STRUCTURAL non-informative-turn query filter
(Agent.noninformative_query_filter) against control, on top of the adopted
preference_tag_bonus=1.5. Also runs the iter-2 LEXICAL filter
(strip_boilerplate_query) for reference.

Full protocol:
  - full 200 + alternating split-half (both must win to adopt)
  - paired per-session diff vs control (how many sessions flip, which way,
    is the aggregate carried by a handful)
  - bootstrap CI on the technical_score delta (2000 resamples over the 200)
  - also reports how many turns each filter actually removes, and how many
    of the removed turns were the informative "For that, what matters is:"
    branch (the vocabulary-can't-parse-it tradeoff)
"""

from __future__ import annotations

import random
import statistics

from agent_dev.agent import Agent
from src.retrieval import build_catalog_index
from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl

CATALOG = "data/catalog.jsonl"


def per_session(result):
    return {s["sample_id"]: s for s in result["sessions"]}


def bootstrap_ts_delta(ctrl_rows, test_rows, ids, n=2000, seed=0):
    rng = random.Random(seed)

    def ts(rows_by_id, sample_ids):
        hits = [rows_by_id[i]["hit"] for i in sample_ids]
        rr = [rows_by_id[i]["reciprocal_rank"] for i in sample_ids]
        ttc = [rows_by_id[i]["first_hit_turn"] or 11 for i in sample_ids]
        hr = sum(hits) / len(sample_ids)
        mrr = statistics.fmean(rr)
        mttc = statistics.fmean(ttc)
        eff = max(0.0, min(1.0, (11 - mttc) / 10))
        return 0.5 * hr + 0.3 * mrr + 0.2 * eff

    deltas = []
    for _ in range(n):
        samp = [rng.choice(ids) for _ in ids]
        deltas.append(ts(test_rows, samp) - ts(ctrl_rows, samp))
    deltas.sort()
    return deltas[int(0.025 * n)], statistics.fmean(deltas), deltas[int(0.975 * n)]


def summ(tag, r):
    print(f"{tag:30s} HR={r['hit_rate_at_10']:.4f} MRR={r['mrr']:.5f} MTTC={r['mttc']:.3f} "
          f"TS={r['recommended_technical_score']:.6f}  "
          f"{ {k:(v['hit_rate_at_10'],round(v['mrr'],3)) for k,v in r['scenario_metrics'].items()} }")


def turns_removed(flag_kwargs, samples, ci, cat, pr, idx):
    """Replay and count how many turns the filter drops, split by whether the
    dropped reply was the informative 'For that, what matters is:' branch."""
    from evaluator.local_evaluator import (
        MAX_TURNS, TOP_K, coarse_category, customer_reply, initial_message,
        materialize_hidden_fields, normalize_recommendations,
    )
    agent = Agent(CATALOG, index=idx, **flag_kwargs)
    dropped_total = dropped_informative_branch = 0
    for s in samples:
        sid = f"tr_{s['sample_id']}"
        agent.reset(sid, s["user_profile"])
        target = str(s["ground_truth"]["parent_asin"])
        card, beh = materialize_hidden_fields(s, pr)
        eff = {**s, "intent_card": card, "behavior": beh}
        disclosed, boundary_used = set(), False
        override_applied = s["scenario_type"] != "intent_override"
        msg = initial_message(eff, coarse_category(cat.get(target, [])), disclosed)
        msgs_by_turn = {}
        for turn in range(1, MAX_TURNS + 1):
            msgs_by_turn[turn] = msg
            resp = agent.respond(sid, msg, turn, TOP_K)
            ranked = normalize_recommendations(resp.get("recommendations"), ci)
            if override_applied and target in ranked:
                break
            if turn == MAX_TURNS:
                break
            ov = eff.get("behavior", {}).get("override") or {}
            if not override_applied and turn + 1 == int(ov.get("turn", 3)):
                override_applied = True
                if str(ov.get("new_value", "")):
                    disclosed.add(str(ov["new_value"]))
                msg = str(ov.get("message", "Actually, please ignore my earlier preference."))
            else:
                msg, boundary_used = customer_reply(eff, resp.get("ask_attribute"), disclosed, boundary_used)
        st = agent._sessions[sid]
        for t in st.noninformative_turns:
            dropped_total += 1
            if msgs_by_turn.get(t, "").startswith("For that, what matters is"):
                dropped_informative_branch += 1
    return dropped_total, dropped_informative_branch


def main() -> None:
    samples = load_jsonl("data/public_set.jsonl")
    ci, cat, pr = catalog_index(CATALOG)
    idx = build_catalog_index(CATALOG)
    ids = [s["sample_id"] for s in samples]
    a_half, b_half = samples[0::2], samples[1::2]

    configs = {
        "control": {},
        "structural (noninformative)": {"noninformative_query_filter": True},
        "lexical (strip_boilerplate)": {"strip_boilerplate_query": True},
    }

    results = {}
    for name, kw in configs.items():
        results[name] = evaluate(Agent(CATALOG, index=idx, **kw), samples, ci, cat, pr)

    print("=== FULL 200 ===")
    for name, r in results.items():
        summ(name, r)

    print("\n=== SPLIT-HALF ===")
    for name, kw in configs.items():
        ra = evaluate(Agent(CATALOG, index=idx, **kw), a_half, ci, cat, pr)
        rb = evaluate(Agent(CATALOG, index=idx, **kw), b_half, ci, cat, pr)
        print(f"{name:30s} A: TS={ra['recommended_technical_score']:.4f} HR={ra['hit_rate_at_10']:.3f}   "
              f"B: TS={rb['recommended_technical_score']:.4f} HR={rb['hit_rate_at_10']:.3f}")

    ctrl = per_session(results["control"])
    print("\n=== PAIRED PER-SESSION DIFF vs control ===")
    for name in ("structural (noninformative)", "lexical (strip_boilerplate)"):
        test = per_session(results[name])
        flipped_to_hit, flipped_to_miss, rank_better, rank_worse = [], [], [], []
        for sid in ids:
            c, t = ctrl[sid], test[sid]
            if not c["hit"] and t["hit"]:
                flipped_to_hit.append(sid)
            elif c["hit"] and not t["hit"]:
                flipped_to_miss.append(sid)
            elif c["hit"] and t["hit"] and c["best_rank"] != t["best_rank"]:
                (rank_better if t["best_rank"] < c["best_rank"] else rank_worse).append(sid)
        lo, mean, hi = bootstrap_ts_delta(ctrl, test, ids)
        d = results[name]["recommended_technical_score"] - results["control"]["recommended_technical_score"]
        print(f"\n  {name}:  TS delta = {d:+.5f}")
        print(f"    -> hit  ({len(flipped_to_hit)}): {flipped_to_hit}")
        print(f"    -> miss ({len(flipped_to_miss)}): {flipped_to_miss}")
        print(f"    rank better among shared hits ({len(rank_better)}), worse ({len(rank_worse)})")
        print(f"    total sessions changed: {len(flipped_to_hit)+len(flipped_to_miss)+len(rank_better)+len(rank_worse)}")
        print(f"    bootstrap TS delta 95% CI: [{lo:+.5f}, {hi:+.5f}]  mean {mean:+.5f}")

    print("\n=== turns removed by the structural filter ===")
    tot, inf = turns_removed({"noninformative_query_filter": True}, samples, ci, cat, pr, idx)
    print(f"  {tot} turns dropped across 200 sessions; {inf} of them were the informative "
          f"'For that, what matters is:' branch (vocab couldn't parse the disclosure)")


if __name__ == "__main__":
    main()
