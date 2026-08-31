"""Task 3 follow-up: A/B two slot-extraction precision fixes surfaced by the
boundary-miss trace, against the full public 200 (+ split-half).

FIX A -- "jeans" should not fill material=denim.
  MATERIAL_VOCAB["denim"] lists {"jean","jeans"} as synonyms. "jeans" is a
  garment/category word, not a reliable material. In public_0169 the target
  is a *knit cotton* jegging in the Jeans category; "Women Jeans" filled
  material=denim on turn 1, which (a) poisoned retrieval with "denim" and
  (b) made `material` permanently non-askable, so the real hard constraint
  "cotton" (classify_constraint -> material) was never disclosed. Target
  never entered the 200-pool.

FIX B -- bare single-character tokens should not match SIZE_VOCAB.
  TOKEN_RE splits "I'm" into ["i", "m"]; "m" matches SIZE_VOCAB key "m", so
  size="m" is filled on turn 1 of essentially every session from the stock
  opener "I'm looking for ...". It's filtered out of the BM25 query (1-char),
  but it suppresses the `size` clarification for the whole session.

Both are implemented as monkeypatches here so the A/B is isolated; the real
change (if adopted) edits agent/vocab.py.
"""

from __future__ import annotations

import copy

from src import vocab
from agent_dev.agent import Agent
from src.retrieval import build_catalog_index
from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl

ORIG_MATERIAL = copy.deepcopy(vocab.MATERIAL_VOCAB)
ORIG_MATCH = vocab._match_vocab


def apply_fix_a():
    vocab.MATERIAL_VOCAB["denim"] = {"jean"}  # drop "jeans"; keep "jean" (still ambiguous but rarer as garment plural)
    vocab._LOOKUP_CACHE.clear()


def apply_fix_b():
    def _match_vocab_minlen(text_norm, tokens, v):
        return ORIG_MATCH(text_norm, [t for t in tokens if len(t) >= 2], v)
    vocab._match_vocab = _match_vocab_minlen


def reset_all():
    vocab.MATERIAL_VOCAB.clear()
    vocab.MATERIAL_VOCAB.update(copy.deepcopy(ORIG_MATERIAL))
    vocab._LOOKUP_CACHE.clear()
    vocab._match_vocab = ORIG_MATCH


def score(idx, samples, catalog_ids, categories, products, tag):
    # fresh index so product_slot_cache reflects the current vocab patch
    agent = Agent("data/catalog.jsonl", index=idx)
    r = evaluate(agent, samples, catalog_ids, categories, products)
    by = {k: (v["hit_rate_at_10"], round(v["mrr"], 4)) for k, v in r["scenario_metrics"].items()}
    print(f"{tag:26s} tech={r['recommended_technical_score']:.6f}  hit={r['hit_rate_at_10']:.4f}  "
          f"mrr={r['mrr']:.5f}  mttc={r['mttc']:.3f}")
    print(f"{'':26s} {by}")
    return r


def main() -> None:
    samples = load_jsonl("data/public_set.jsonl")
    catalog_ids, categories, products = catalog_index("data/catalog.jsonl")
    half_a, half_b = samples[0::2], samples[1::2]

    configs = [
        ("control", lambda: None),
        ("fix A (no jeans->denim)", apply_fix_a),
        ("fix B (no 1-char size)", apply_fix_b),
        ("fix A + B", lambda: (apply_fix_a(), apply_fix_b())),
    ]

    for tag, apply in configs:
        reset_all()
        apply()
        idx = build_catalog_index("data/catalog.jsonl")  # rebuild -> fresh vocab + caches
        score(idx, samples, catalog_ids, categories, products, tag + " [full]")
        score(idx, half_a, catalog_ids, categories, products, tag + " [halfA]")
        score(idx, half_b, catalog_ids, categories, products, tag + " [halfB]")
        print()

    reset_all()


if __name__ == "__main__":
    main()
