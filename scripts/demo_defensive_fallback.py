"""Live demo: the defensive ask-fallback tier (agent.py's
PRIMARY_UNASKABLE_SLOTS retry) -- falls back to asking about brand/
category/budget only once every more-productive question is exhausted,
rather than going silent for the rest of the session.

This also never fires on the public 200 sessions (there's always a more
productive question left before turn 10), so this demo constructs the
exhausted-state scenario directly to prove the mechanism.

Usage:
    python -m scripts.demo_defensive_fallback
    python -m scripts.demo_defensive_fallback --auto
"""

from __future__ import annotations

import argparse

from src import memory
from src.vocab import Vocabulary

PRODUCTS = [
    {"parent_asin": "a1", "title": "Runner Pro Sneaker", "categories": ["Clothing", "Shoes", "Running Shoes"], "store": "Nimbus Athletics", "price": 60.0, "features": []},
    {"parent_asin": "a2", "title": "Trail Runner Sneaker", "categories": ["Clothing", "Shoes", "Trail Shoes"], "store": "Peakline", "price": 75.0, "features": []},
    {"parent_asin": "a3", "title": "City Runner Sneaker", "categories": ["Clothing", "Shoes", "Street Shoes"], "store": "Nimbus Athletics", "price": 55.0, "features": []},
]

PRIMARY_UNASKABLE_SLOTS = frozenset({"brand", "category", "budget"})


def pause(auto: bool) -> None:
    if not auto:
        input("\n[press Enter to continue]\n")
    else:
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Live defensive-fallback-tier demo")
    parser.add_argument("--auto", action="store_true")
    args = parser.parse_args()

    print("=" * 78)
    print("SCENARIO (self-produced): every productive attribute -- material,")
    print("color, size, style, feature, use_case -- is already filled or the")
    print("customer has said they have no preference for it.")
    print("=" * 78)

    vocab = Vocabulary()
    state = memory.reset("fallback_demo", {})
    state.slots["material"].value = "mesh"
    for slot_name in ("color", "size", "style", "feature", "use_case"):
        state.slots[slot_name].source = memory.Source.NO_PREFERENCE
        state.slots[slot_name].answered = True

    for slot_name in ("material", "color", "size", "style", "feature", "use_case"):
        slot = state.slots[slot_name]
        status = f"filled = {slot.value!r}" if slot.value is not None else "no preference"
        print(f"  {slot_name:<10} -> {status}")
    pause(args.auto)

    print("-" * 78)
    print("Step 1: ask the PRIMARY pool (brand/category/budget excluded)")
    print("-" * 78)
    primary = memory.select_clarification_attribute(state, PRODUCTS, vocab, 1.5, PRIMARY_UNASKABLE_SLOTS)
    print(f"select_clarification_attribute(...) -> {primary!r}")
    print("(None: every productive attribute is genuinely exhausted)")
    pause(args.auto)

    print("-" * 78)
    print("Step 2: fall back to the full set instead of asking nothing")
    print("-" * 78)
    fallback = memory.select_clarification_attribute(state, PRODUCTS, vocab, 1.5, frozenset())
    print(f"select_clarification_attribute(...) -> {fallback!r}")
    print(f"\n-> the agent asks about {fallback!r} rather than going silent for the")
    print("   rest of the session. It's picked because, in THIS candidate pool,")
    print("   it's the one of brand/category/budget that actually varies --")
    print("   not a fixed default, real information-gain scoring either way.")
    print("=" * 78)


if __name__ == "__main__":
    main()
