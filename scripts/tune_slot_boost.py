"""iter 3 step 1: sweep slot_boost_k -- how many EXTRA times to repeat each
confirmed slot value's terms in the BM25 MATCH expression -- on top of the
adopted preference_tag_bonus=1.5.

Full protocol per the brief:
  - full 200 + alternating split-half (a real win must hold on BOTH halves)
  - paired per-session diff vs control (k=0): sessions flipped hit<->miss,
    rank moved among shared hits, is the aggregate carried by a handful
  - bootstrap 95% CI on the technical_score delta (2000 resamples)
  - determinism is checked separately by re-running run_memory_agent

slot_boost_k=0 == the pre-boost agent (the slot values are already in the
query once via _query_text), so effective repetition is k+1.
"""

from __future__ import annotations

import random
import statistics

from agent.agent import Agent
from agent.retrieval import build_catalog_index
from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl

CATALOG = "data/catalog.jsonl"
K_VALUES = [0, 1, 2, 3, 5, 8]


def rows_by_id(result):
    return {s["sample_id"]: s for s in result["sessions"]}


def ts_of(rbid, ids):
    hits = [rbid[i]["hit"] for i in ids]
    rr = [rbid[i]["reciprocal_rank"] for i in ids]
    ttc = [rbid[i]["first_hit_turn"] or 11 for i in ids]
    hr = sum(hits) / len(ids)
    mrr = statistics.fmean(rr)
    mttc = statistics.fmean(ttc)
    eff = max(0.0, min(1.0, (11 - mttc) / 10))
    return 0.5 * hr + 0.3 * mrr + 0.2 * eff


def bootstrap(ctrl, test, ids, n=2000, seed=0):
    rng = random.Random(seed)
    ds = []
    for _ in range(n):
        s = [rng.choice(ids) for _ in ids]
        ds.append(ts_of(test, s) - ts_of(ctrl, s))
    ds.sort()
    return ds[int(0.025 * n)], statistics.fmean(ds), ds[int(0.975 * n)]


def summ(tag, r):
    sc = {k: (v["hit_rate_at_10"], round(v["mrr"], 3)) for k, v in r["scenario_metrics"].items()}
    print(f"{tag:14s} HR={r['hit_rate_at_10']:.4f} MRR={r['mrr']:.5f} MTTC={r['mttc']:.3f} "
          f"TS={r['recommended_technical_score']:.6f}  {sc}")


def main() -> None:
    samples = load_jsonl("data/public_set.jsonl")
    ci, cat, pr = catalog_index(CATALOG)
    idx = build_catalog_index(CATALOG)
    ids = [s["sample_id"] for s in samples]
    a_half, b_half = samples[0::2], samples[1::2]

    results = {k: evaluate(Agent(CATALOG, index=idx, slot_boost_k=k), samples, ci, cat, pr)
               for k in K_VALUES}

    print("=== FULL 200 ===")
    for k in K_VALUES:
        summ(f"k={k}", results[k])

    print("\n=== SPLIT-HALF ===")
    for k in K_VALUES:
        ra = evaluate(Agent(CATALOG, index=idx, slot_boost_k=k), a_half, ci, cat, pr)
        rb = evaluate(Agent(CATALOG, index=idx, slot_boost_k=k), b_half, ci, cat, pr)
        print(f"k={k}:  A TS={ra['recommended_technical_score']:.4f} HR={ra['hit_rate_at_10']:.3f} "
              f"MRR={ra['mrr']:.4f}   B TS={rb['recommended_technical_score']:.4f} "
              f"HR={rb['hit_rate_at_10']:.3f} MRR={rb['mrr']:.4f}")

    ctrl = rows_by_id(results[0])
    print("\n=== PAIRED DIFF vs k=0 ===")
    for k in K_VALUES[1:]:
        test = rows_by_id(results[k])
        to_hit, to_miss, better, worse = [], [], [], []
        for sid in ids:
            c, t = ctrl[sid], test[sid]
            if not c["hit"] and t["hit"]:
                to_hit.append(sid)
            elif c["hit"] and not t["hit"]:
                to_miss.append(sid)
            elif c["hit"] and t["hit"] and c["best_rank"] != t["best_rank"]:
                (better if t["best_rank"] < c["best_rank"] else worse).append(sid)
        lo, mean, hi = bootstrap(ctrl, test, ids)
        d = results[k]["recommended_technical_score"] - results[0]["recommended_technical_score"]
        print(f"\n  k={k}:  TS delta {d:+.5f}   bootstrap 95% CI [{lo:+.5f}, {hi:+.5f}] mean {mean:+.5f}")
        print(f"    hit<-miss: {len(to_hit)} {to_hit}")
        print(f"    miss<-hit: {len(to_miss)} {to_miss}")
        print(f"    rank better/worse among shared hits: {len(better)} / {len(worse)}")
        print(f"    total sessions changed: {len(to_hit)+len(to_miss)+len(better)+len(worse)}")


if __name__ == "__main__":
    main()
