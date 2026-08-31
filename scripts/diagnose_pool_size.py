"""Instrument pool size + ask_attribute per turn across the public 200.

Answers the Task-2 question: now that the clarification stuck-loop is fixed,
does the OR-based BM25 pool ever fall below `POOL_THRESHOLD` on its own, or
is it still pinned at the RETRIEVAL_POOL_SIZE cap on every turn?

Also reports repeated-ask streaks, so the stuck-loop pattern can be measured
before/after the memory.py fix with the same instrument.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from agent_dev.agent import Agent
from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl


class InstrumentedAgent(Agent):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.turn_log: list[dict] = []

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        out = super().respond(session_id, user_message, turn, top_k)
        state = self._sessions[session_id]
        self.turn_log.append({
            "session_id": session_id,
            "turn": turn,
            "pool_size": len(state.last_candidate_pool),
            "ask_attribute": out["ask_attribute"],
            "n_filled_slots": len(
                [s for s in state.slots.values() if s.value is not None]
            ),
        })
        return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--output", default="runs/pool_diagnostic.json")
    args = parser.parse_args()

    samples = load_jsonl(args.dataset)
    catalog_ids, categories, products = catalog_index(args.catalog)
    agent = InstrumentedAgent(args.catalog)
    result = evaluate(agent, samples, catalog_ids, categories, products)

    log = agent.turn_log
    sizes = [row["pool_size"] for row in log]
    threshold = agent.pool_threshold

    # repeated-ask streaks: same non-null ask_attribute on consecutive turns
    streaks: list[int] = []
    by_session: dict[str, list] = {}
    for row in log:
        by_session.setdefault(row["session_id"], []).append(row)
    stuck_sessions = 0
    for rows in by_session.values():
        best = 1
        run = 1
        for prev, cur in zip(rows, rows[1:]):
            if cur["ask_attribute"] is not None and cur["ask_attribute"] == prev["ask_attribute"]:
                run += 1
            else:
                run = 1
            best = max(best, run)
        streaks.append(best)
        if best >= 3:
            stuck_sessions += 1

    summary = {
        "total_turns": len(sizes),
        "sessions": len(by_session),
        "pool_threshold": threshold,
        "pool_size_min": min(sizes) if sizes else None,
        "pool_size_max": max(sizes) if sizes else None,
        "pool_size_mean": round(sum(sizes) / len(sizes), 3) if sizes else None,
        "turns_at_or_below_threshold": sum(1 for s in sizes if s <= threshold),
        "turns_below_200": sum(1 for s in sizes if s < 200),
        "pool_size_histogram": dict(sorted(Counter(sizes).items())[:20]),
        "sessions_with_ask_repeated_3plus": stuck_sessions,
        "max_repeat_streak_histogram": dict(sorted(Counter(streaks).items())),
        "metrics": {k: v for k, v in result.items() if k not in ("sessions",)},
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps({"summary": summary, "turn_log": log}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
