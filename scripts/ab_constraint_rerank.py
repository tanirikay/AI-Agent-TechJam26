"""Iteration 4 A/B for deterministic raw-constraint exact reranking.

Runs the untouched local evaluator protocol once per configuration, then
reports full-set, scenario, alternating split-half, paired, bootstrap, and
runtime metrics.  The control explicitly passes ``None`` so it stays inert
even if an exact window is later adopted as Agent's default.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path

from agent_dev.agent import Agent
from src.retrieval import build_catalog_index
from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl, metric_summary


CATALOG = "data/catalog.jsonl"
DATASET = "data/public_set.jsonl"
CONFIGS = (
    ("control", None),
    ("exact_20", 20),
    ("exact_40", 40),
    ("exact_100", 100),
    ("exact_200", 200),
)


def _summarize(rows: list[dict]) -> dict:
    overall = metric_summary(rows)
    mttc = float(overall["mttc"])
    efficiency = max(0.0, min(1.0, (11.0 - mttc) / 10.0))
    technical_score = (
        0.50 * overall["hit_rate_at_10"]
        + 0.30 * overall["mrr"]
        + 0.20 * efficiency
    )
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["scenario_type"]].append(row)
    return {
        **overall,
        "efficiency": round(efficiency, 6),
        "recommended_technical_score": round(technical_score, 6),
        "scenario_metrics": {
            name: metric_summary(grouped[name]) for name in sorted(grouped)
        },
    }


def _rows_by_id(result: dict) -> dict[str, dict]:
    return {row["sample_id"]: row for row in result["sessions"]}


def _technical_score(rows: dict[str, dict], sample_ids: list[str]) -> float:
    hits = [int(rows[sample_id]["hit"]) for sample_id in sample_ids]
    reciprocal_ranks = [rows[sample_id]["reciprocal_rank"] for sample_id in sample_ids]
    turns = [rows[sample_id]["first_hit_turn"] or 11 for sample_id in sample_ids]
    hit_rate = sum(hits) / len(sample_ids)
    mrr = statistics.fmean(reciprocal_ranks)
    mttc = statistics.fmean(turns)
    efficiency = max(0.0, min(1.0, (11.0 - mttc) / 10.0))
    return 0.50 * hit_rate + 0.30 * mrr + 0.20 * efficiency


def _bootstrap_delta(
    control: dict[str, dict],
    treatment: dict[str, dict],
    sample_ids: list[str],
    resamples: int,
    seed: int = 0,
) -> dict:
    rng = random.Random(seed)
    deltas: list[float] = []
    for _ in range(resamples):
        sample = [rng.choice(sample_ids) for _ in sample_ids]
        deltas.append(
            _technical_score(treatment, sample) - _technical_score(control, sample)
        )
    deltas.sort()
    return {
        "resamples": resamples,
        "mean": statistics.fmean(deltas),
        "lower_95": deltas[int(0.025 * resamples)],
        "upper_95": deltas[min(resamples - 1, int(0.975 * resamples))],
    }


def _session_utility(row: dict) -> float:
    turn = row["first_hit_turn"] or 11
    efficiency = max(0.0, min(1.0, (11.0 - turn) / 10.0))
    return 0.50 * int(row["hit"]) + 0.30 * row["reciprocal_rank"] + 0.20 * efficiency


def _paired_summary(
    control: dict[str, dict], treatment: dict[str, dict], sample_ids: list[str]
) -> dict:
    miss_to_hit: list[str] = []
    hit_to_miss: list[str] = []
    gained: list[str] = []
    lost: list[str] = []
    rank_better: list[str] = []
    rank_worse: list[str] = []
    for sample_id in sample_ids:
        before = control[sample_id]
        after = treatment[sample_id]
        if not before["hit"] and after["hit"]:
            miss_to_hit.append(sample_id)
        elif before["hit"] and not after["hit"]:
            hit_to_miss.append(sample_id)
        if before["hit"] and after["hit"] and before["best_rank"] != after["best_rank"]:
            (rank_better if after["best_rank"] < before["best_rank"] else rank_worse).append(sample_id)
        delta = _session_utility(after) - _session_utility(before)
        if delta > 1e-12:
            gained.append(sample_id)
        elif delta < -1e-12:
            lost.append(sample_id)
    return {
        "gained_count": len(gained),
        "gained_sessions": gained,
        "lost_count": len(lost),
        "lost_sessions": lost,
        "miss_to_hit_count": len(miss_to_hit),
        "miss_to_hit_sessions": miss_to_hit,
        "hit_to_miss_count": len(hit_to_miss),
        "hit_to_miss_sessions": hit_to_miss,
        "rank_better_shared_hits": len(rank_better),
        "rank_worse_shared_hits": len(rank_worse),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default=CATALOG)
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--output", default="runs/constraint_rerank_ab.json")
    args = parser.parse_args()

    samples = load_jsonl(args.dataset)
    catalog_ids, categories, products = catalog_index(args.catalog)
    index = build_catalog_index(args.catalog)
    sample_ids = [sample["sample_id"] for sample in samples]

    report: dict = {
        "configs": {},
        "paired_vs_control": {},
        "paired_best_window_comparisons": {},
    }
    full_results: dict[str, dict] = {}
    for name, window in CONFIGS:
        started = time.perf_counter()
        result = evaluate(
            Agent(args.catalog, index=index, constraint_rerank_window=window),
            samples,
            catalog_ids,
            categories,
            products,
        )
        elapsed = time.perf_counter() - started
        turns = sum(row["first_hit_turn"] or 10 for row in result["sessions"])
        full_results[name] = result
        report["configs"][name] = {
            "window": window,
            "full": {key: value for key, value in result.items() if key not in ("sessions", "reported_token_usage")},
            "alternating_half_a": _summarize(result["sessions"][0::2]),
            "alternating_half_b": _summarize(result["sessions"][1::2]),
            "runtime_seconds": elapsed,
            "evaluated_turns": turns,
            "approx_ms_per_turn": 1000.0 * elapsed / turns,
            "sessions": result["sessions"],
        }

    control_rows = _rows_by_id(full_results["control"])
    for name, window in CONFIGS[1:]:
        treatment_rows = _rows_by_id(full_results[name])
        report["paired_vs_control"][name] = {
            **_paired_summary(control_rows, treatment_rows, sample_ids),
            "technical_score_delta": (
                full_results[name]["recommended_technical_score"]
                - full_results["control"]["recommended_technical_score"]
            ),
            "bootstrap_technical_score_delta": _bootstrap_delta(
                control_rows,
                treatment_rows,
                sample_ids,
                args.bootstrap_resamples,
            ),
        }

    best_name, best_window = max(
        CONFIGS[1:],
        key=lambda config: full_results[config[0]]["recommended_technical_score"],
    )
    best_rows = _rows_by_id(full_results[best_name])
    report["best_exact_config"] = {"name": best_name, "window": best_window}
    for name, window in CONFIGS[1:]:
        if name == best_name:
            continue
        candidate_rows = _rows_by_id(full_results[name])
        comparison_name = f"{best_name}_minus_{name}"
        report["paired_best_window_comparisons"][comparison_name] = {
            "larger_window": best_window,
            "comparison_window": window,
            "technical_score_delta": (
                full_results[best_name]["recommended_technical_score"]
                - full_results[name]["recommended_technical_score"]
            ),
            "bootstrap_technical_score_delta": _bootstrap_delta(
                candidate_rows,
                best_rows,
                sample_ids,
                args.bootstrap_resamples,
            ),
        }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("=== FULL / SPLIT / RUNTIME ===")
    for name, _ in CONFIGS:
        item = report["configs"][name]
        full = item["full"]
        half_a = item["alternating_half_a"]
        half_b = item["alternating_half_b"]
        print(
            f"{name:10s} HR={full['hit_rate_at_10']:.4f} MRR={full['mrr']:.6f} "
            f"MTTC={full['mttc']:.3f} TS={full['recommended_technical_score']:.6f} | "
            f"A HR/TS={half_a['hit_rate_at_10']:.3f}/{half_a['recommended_technical_score']:.6f} "
            f"B HR/TS={half_b['hit_rate_at_10']:.3f}/{half_b['recommended_technical_score']:.6f} | "
            f"{item['runtime_seconds']:.2f}s, {item['approx_ms_per_turn']:.2f}ms/turn"
        )
        print(f"           scenarios={full['scenario_metrics']}")

    print("\n=== PAIRED VS CONTROL ===")
    for name, _ in CONFIGS[1:]:
        item = report["paired_vs_control"][name]
        ci = item["bootstrap_technical_score_delta"]
        print(
            f"{name:10s} delta={item['technical_score_delta']:+.6f} "
            f"gained/lost={item['gained_count']}/{item['lost_count']} "
            f"miss->hit={item['miss_to_hit_count']} hit->miss={item['hit_to_miss_count']} "
            f"CI95=[{ci['lower_95']:+.6f}, {ci['upper_95']:+.6f}]"
        )

    print(f"\n=== BEST EXACT WINDOW: {best_name} ===")
    for name, item in report["paired_best_window_comparisons"].items():
        ci = item["bootstrap_technical_score_delta"]
        print(
            f"{name:25s} delta={item['technical_score_delta']:+.6f} "
            f"CI95=[{ci['lower_95']:+.6f}, {ci['upper_95']:+.6f}]"
        )
    print(f"\nDetailed report: {output_path}")


if __name__ == "__main__":
    main()
