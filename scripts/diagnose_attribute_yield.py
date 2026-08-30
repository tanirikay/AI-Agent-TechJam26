"""Diagnostic: for each ask_attribute, measure how often asking it FIRST (at
turn 1, given whatever the opening message already disclosed) actually
yields a real disclosure from evaluator.local_evaluator.customer_reply(),
versus falling through to "I don't have an additional preference for X."

Read-only measurement against the public 200 -- does not change agent
behaviour. Motivated by classify_constraint() being a much narrower keyword
matcher than agent/vocab.py's controlled vocabulary (e.g. color only
recognizes black/white/blue/red/pink/green + the literal word "color"), so
some askable attributes may be structurally low-yield against this specific
evaluator template regardless of how much entropy they carry over the
candidate pool. Also measures "other", which bypasses classify_constraint
entirely and is currently never produced by select_clarification_attribute.
"""

from __future__ import annotations

from collections import Counter

from evaluator.local_evaluator import (
    ALLOWED_ATTRIBUTES,
    catalog_index,
    coarse_category,
    customer_reply,
    initial_message,
    load_jsonl,
    materialize_hidden_fields,
)

ASKABLE = sorted(ALLOWED_ATTRIBUTES)


def main() -> None:
    samples = load_jsonl("data/public_set.jsonl")
    catalog_ids, categories, products = catalog_index("data/catalog.jsonl")

    yield_count: Counter[str] = Counter()
    by_scenario_yield: dict[str, Counter[str]] = {}
    scenario_totals: Counter[str] = Counter()

    for sample in samples:
        target = str(sample["ground_truth"]["parent_asin"])
        card, behavior = materialize_hidden_fields(sample, products)
        effective_sample = {**sample, "intent_card": card, "behavior": behavior}
        disclosed: set[str] = set()
        category_text = coarse_category(categories.get(target, []))
        initial_message(effective_sample, category_text, disclosed)  # mutates disclosed

        scenario = sample["scenario_type"]
        scenario_totals[scenario] += 1
        by_scenario_yield.setdefault(scenario, Counter())

        for attribute in ASKABLE:
            reply, _ = customer_reply(effective_sample, attribute, set(disclosed), True)
            if reply.startswith("For that, what matters is:"):
                yield_count[attribute] += 1
                by_scenario_yield[scenario][attribute] += 1

    total = len(samples)
    print(f"Turn-1 yield rate per ask_attribute (n={total} public sessions)\n")
    print(f"{'attribute':<12}{'yield':>10}{'rate':>10}")
    for attribute in sorted(ASKABLE, key=lambda a: -yield_count[a]):
        n = yield_count[attribute]
        print(f"{attribute:<12}{n:>7}/{total:<3}{n/total:>9.1%}")

    print("\nBy scenario:")
    for scenario in sorted(by_scenario_yield):
        counts = by_scenario_yield[scenario]
        n_total = scenario_totals[scenario]
        print(f"\n-- {scenario} (n={n_total}) --")
        for attribute in sorted(ASKABLE, key=lambda a: -counts.get(a, 0)):
            n = counts.get(attribute, 0)
            print(f"   {attribute:<12}{n:>4}/{n_total:<3}{n/n_total:>8.1%}")


if __name__ == "__main__":
    main()
