"""Task 2: A/B the boilerplate-query filter (Agent.strip_boilerplate_query),
on top of the adopted preference_tag_bonus=1.5, full 200 + alternating split.
"""

from __future__ import annotations

from agent_dev.agent import Agent
from src.retrieval import build_catalog_index
from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl


def line(tag, r):
    by = {k: (v["hit_rate_at_10"], round(v["mrr"], 4)) for k, v in r["scenario_metrics"].items()}
    print(f"{tag:34s} tech={r['recommended_technical_score']:.6f} hit={r['hit_rate_at_10']:.4f} "
          f"mrr={r['mrr']:.5f} mttc={r['mttc']:.3f}")
    print(f"{'':34s} {by}")


def main() -> None:
    samples = load_jsonl("data/public_set.jsonl")
    ci, cat, pr = catalog_index("data/catalog.jsonl")
    idx = build_catalog_index("data/catalog.jsonl")
    a, b = samples[0::2], samples[1::2]

    for strip in (False, True):
        for tag, subset in (("full", samples), ("halfA", a), ("halfB", b)):
            agent = Agent("data/catalog.jsonl", index=idx, strip_boilerplate_query=strip)
            line(f"strip={strip} [{tag}]", evaluate(agent, subset, ci, cat, pr))
        print()


if __name__ == "__main__":
    main()
