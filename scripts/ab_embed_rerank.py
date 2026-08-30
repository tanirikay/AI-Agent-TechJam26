"""Task 4: A/B the embedding-similarity reranker (Agent.embed_rerank_window)
against the BM25+budget baseline, on top of preference_tag_bonus=1.5.

Windows mirror the earlier (rejected) slot-match experiment: rerank only the
top N of the 200-pool so a semantically-plausible but BM25-distant candidate
can't leapfrog a confirmed top candidate. Full 200 + alternating split-half.

Model: sentence-transformers all-MiniLM-L6-v2 (CPU, ~90MB, offline once
cached). Product embeddings (title + features) are built once and cached to
$TECHJAM_EMBED_CACHE.
"""

from __future__ import annotations

import time

from agent.agent import Agent
from agent.retrieval import build_catalog_index
from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl

WINDOWS = [None, 200, 100, 40, 20]  # None in the "control" slot below


def line(tag, r):
    by = {k: (v["hit_rate_at_10"], round(v["mrr"], 4)) for k, v in r["scenario_metrics"].items()}
    print(f"{tag:24s} tech={r['recommended_technical_score']:.6f} hit={r['hit_rate_at_10']:.4f} "
          f"mrr={r['mrr']:.5f} mttc={r['mttc']:.3f}")
    print(f"{'':24s} {by}")


def main() -> None:
    samples = load_jsonl("data/public_set.jsonl")
    ci, cat, pr = catalog_index("data/catalog.jsonl")
    idx = build_catalog_index("data/catalog.jsonl")
    a, b = samples[0::2], samples[1::2]

    # warm the embedding cache once
    t = time.time()
    from agent.embed_rerank import get_shared_reranker
    get_shared_reranker(idx.products)
    print(f"embedding cache ready in {time.time()-t:.1f}s\n")

    configs = [("control", None), ("win=200", 200), ("win=100", 100), ("win=40", 40), ("win=20", 20)]
    for tag, win in configs:
        for sub_tag, subset in (("full", samples), ("A", a), ("B", b)):
            t = time.time()
            agent = Agent("data/catalog.jsonl", index=idx, embed_rerank_window=win)
            r = evaluate(agent, subset, ci, cat, pr)
            line(f"{tag} [{sub_tag}] ({time.time()-t:.0f}s)", r)
        print()


if __name__ == "__main__":
    main()
