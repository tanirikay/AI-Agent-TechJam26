"""Live demo: replay a real public-set session turn by turn, with clean
formatted output for screen recording. Advances on Enter so the presenter
controls pacing while narrating.

Usage:
    python -m scripts.demo_conversation                  # default session
    python -m scripts.demo_conversation --session public_0027
    python -m scripts.demo_conversation --auto            # no pauses, for a dry run

This replays the simulator's own scripted customer against the real agent --
nothing here is faked or pre-recorded. The target/session choice is fixed
(not free-typed) so the demo is reproducible on camera; verified in advance
to converge to rank 1 by turn 3 for the default session.
"""

from __future__ import annotations

import argparse

from src import memory
from agent import Agent
from src.constraint_rerank import normalize_constraint_text
from src.retrieval import build_catalog_index
from evaluator.local_evaluator import (
    catalog_index,
    coarse_category,
    customer_reply,
    initial_message,
    load_jsonl,
    materialize_hidden_fields,
)

WIDTH = 78


def rule(char: str = "-") -> None:
    print(char * WIDTH)


def pause(auto: bool) -> None:
    if not auto:
        input("\n[press Enter to continue]\n")
    else:
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Live conversation demo")
    parser.add_argument("--session", default="public_0108")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--auto", action="store_true", help="don't wait for Enter between turns")
    parser.add_argument("--max-turns", type=int, default=10)
    args = parser.parse_args()

    print("Loading catalog and building index...")
    samples = load_jsonl(args.dataset)
    catalog_ids, categories, products = catalog_index(args.catalog)
    index = build_catalog_index(args.catalog)
    print("Ready.\n")

    sample = next((s for s in samples if s["sample_id"] == args.session), None)
    if sample is None:
        raise SystemExit(f"session {args.session!r} not found in {args.dataset}")

    target = str(sample["ground_truth"]["parent_asin"])
    card, behavior = materialize_hidden_fields(sample, products)
    effective = {**sample, "intent_card": card, "behavior": behavior}
    disclosed: set[str] = set()
    category_text = coarse_category(categories.get(target, []))
    msg = initial_message(effective, category_text, disclosed)

    agent = Agent(index=index)
    sid = f"live_{args.session}"
    agent.reset(sid, sample["user_profile"])
    boundary_used = False

    rule("=")
    print(f"SESSION {args.session}  |  scenario: {sample['scenario_type']}")
    print(f"(target hidden from the agent -- shown here only for the demo narration)")
    print(f"TARGET: {products[target]['title']}")
    rule("=")
    pause(args.auto)

    for turn in range(1, args.max_turns + 1):
        rule()
        print(f"TURN {turn}")
        rule()
        print(f'CUSTOMER: "{msg}"')

        resp = agent.respond(sid, msg, turn, 10)
        state = agent._sessions[sid]

        active = [c.original_text for c in state.raw_constraints if c.active]
        print(f"\nAgent's remembered facts so far: {active}")

        ranked = [r["parent_asin"] for r in resp["recommendations"]]
        rank = ranked.index(target) + 1 if target in ranked else None

        print("\nTop recommendations:")
        for i, asin in enumerate(ranked[:5], start=1):
            marker = "  <-- TARGET" if asin == target else ""
            print(f"  {i}. {products[asin]['title'][:60]}{marker}")

        if rank is not None:
            print(f"\n*** TARGET FOUND at rank {rank} on turn {turn} ***")
            evidence = index.normalized_evidence[target]
            print("\nWhy: every disclosed fact checked against the target's own listing text:")
            for clause in active:
                norm = normalize_constraint_text(clause)
                found = norm in evidence
                print(f'  "{clause}" -> found verbatim: {found}')
            rule("=")
            print("DONE")
            rule("=")
            return

        print(f"\nAGENT ASKS ({resp['ask_attribute']}): \"{resp['message']}\"")
        pause(args.auto)

        override = effective.get("behavior", {}).get("override") or {}
        override_applied = effective["scenario_type"] != "intent_override"
        if not override_applied and turn + 1 == int(override.get("turn", 3)):
            new_value = str(override.get("new_value", ""))
            if new_value:
                disclosed.add(new_value)
            msg = str(override.get("message", ""))
        else:
            msg, boundary_used = customer_reply(effective, resp.get("ask_attribute"), disclosed, boundary_used)

    print("\n(session ended without a hit within max_turns)")


if __name__ == "__main__":
    main()
