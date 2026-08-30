"""Run the public-set evaluator against agent.agent.Agent (memory + follow-up module).

Mirrors evaluator/local_evaluator.py's main(), but points at the new Agent
instead of the weak BM25 starter, so the two can be compared side by side
without touching the evaluator or the reference baseline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agent.agent import Agent
from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the memory+follow-up agent over the public set")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--output", default="results_memory_agent.json")
    args = parser.parse_args()

    samples = load_jsonl(args.dataset)
    catalog_ids, categories, products = catalog_index(args.catalog)
    result = evaluate(Agent(args.catalog), samples, catalog_ids, categories, products)

    Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "sessions"}, indent=2))


if __name__ == "__main__":
    main()
