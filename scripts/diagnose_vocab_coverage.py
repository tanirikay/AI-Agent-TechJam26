"""Diagnostic: how much of the FULL 50,000-product catalog does our
controlled vocabulary (agent/vocab.py) actually recognize, versus the
narrower slice implicitly exercised by the 200 public sessions?

Motivation: the 200 public sessions only sample 200 of the 50,000 catalog
products as targets. The private 800 sample a different, much larger slice
of the SAME catalog. Since evaluator.local_evaluator's customer dialogue is
built entirely by interpolating literal catalog title/feature/detail text
into fixed templates (verified by reading initial_message/customer_reply/
intent_card -- there is no free-text paraphrasing anywhere in the shipped
simulator), the real generalization risk is not "the customer might phrase
things differently" -- it's "our controlled vocab might not recognize
words this specific slice of 50,000 products happens to use," which IS
measurable locally, right now, without needing the private 800 at all.

This also cross-checks our MATERIAL_VOCAB/COLOR_VOCAB against the
evaluator's OWN narrow MATERIAL_RE/COLOR_RE (used to build customer
disclosures) -- any evaluator-recognized term we don't recognize is a
zero-ambiguity, guaranteed-safe addition, not a guess.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from agent.vocab import (
    COLOR_VOCAB,
    FEATURE_VOCAB,
    MATERIAL_VOCAB,
    SIZE_VOCAB,
    STYLE_VOCAB,
    USE_CASE_VOCAB,
    Vocabulary,
)
from evaluator.local_evaluator import COLOR_RE, MATERIAL_RE, searchable_text

VOCABS = {
    "material": MATERIAL_VOCAB,
    "color": COLOR_VOCAB,
    "size": SIZE_VOCAB,
    "style": STYLE_VOCAB,
    "use_case": USE_CASE_VOCAB,
    "feature": FEATURE_VOCAB,
}


def product_text(product: dict) -> str:
    """Exactly what agent/memory.py's get_attribute_value actually searches
    in production: title + features only."""
    title = str(product.get("title") or "")
    features = product.get("features")
    features_text = " ".join(str(f) for f in features) if isinstance(features, list) else str(features or "")
    return f"{title} {features_text}"


def _flatten(value: object) -> str:
    if isinstance(value, dict):
        return " ".join(f"{k} {_flatten(v)}" for k, v in value.items())
    if isinstance(value, list):
        return " ".join(_flatten(v) for v in value)
    return str(value) if value is not None else ""


def product_text_moderate(product: dict) -> str:
    """Candidate widened scope: title+features+details+description, WITHOUT
    store/categories -- those are exactly the fields that caused the
    original brand false-positive bug (store names containing ordinary
    words), so they're deliberately excluded from this general-extraction
    blob even though the evaluator's own searchable_text() includes them."""
    parts = [
        str(product.get("title") or ""),
        _flatten(product.get("features")),
        _flatten(product.get("details")),
        _flatten(product.get("description")),
    ]
    return " ".join(parts)


def main() -> None:
    vocabulary = Vocabulary()
    catalog_path = Path("data/catalog.jsonl")

    hit_counts = {name: 0 for name in VOCABS}
    hit_counts_wide = {name: 0 for name in VOCABS}
    hit_counts_moderate = {name: 0 for name in VOCABS}
    total = 0
    evaluator_material_hits = 0
    evaluator_color_hits = 0
    our_miss_evaluator_material_hit = 0
    our_miss_evaluator_color_hit = 0
    wide_miss_evaluator_material_hit = 0
    wide_miss_evaluator_color_hit = 0
    missed_material_words: dict[str, int] = {}
    missed_color_words: dict[str, int] = {}

    with catalog_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            product = json.loads(line)
            total += 1
            corpus = searchable_text(product)  # everything the evaluator itself searches

            text = product_text(product)  # title+features only -- what get_attribute_value actually sees today
            extracted = vocabulary.extract(text, include_catalog_terms=False)
            extracted_wide = vocabulary.extract(corpus, include_catalog_terms=False)
            extracted_moderate = vocabulary.extract(product_text_moderate(product), include_catalog_terms=False)
            for name in VOCABS:
                if extracted.get(name) is not None:
                    hit_counts[name] += 1
                if extracted_wide.get(name) is not None:
                    hit_counts_wide[name] += 1
                if extracted_moderate.get(name) is not None:
                    hit_counts_moderate[name] += 1

            m = MATERIAL_RE.search(corpus)
            c = COLOR_RE.search(corpus)
            if m:
                evaluator_material_hits += 1
                if extracted.get("material") is None:
                    our_miss_evaluator_material_hit += 1
                    word = m.group(1).lower()
                    missed_material_words[word] = missed_material_words.get(word, 0) + 1
                if extracted_wide.get("material") is None:
                    wide_miss_evaluator_material_hit += 1
            if c:
                evaluator_color_hits += 1
                if extracted.get("color") is None:
                    our_miss_evaluator_color_hit += 1
                    word = c.group(1).lower()
                    missed_color_words[word] = missed_color_words.get(word, 0) + 1
                if extracted_wide.get("color") is None:
                    wide_miss_evaluator_color_hit += 1

    print(f"Full catalog: {total} products\n")
    print("Our vocab.py coverage, PRODUCTION field scope (title+features only, matches agent/memory.py today):")
    for name, count in hit_counts.items():
        print(f"  {name:<10}{count:>7}/{total:<7}{count/total:>7.1%}")

    print("\nSame vocab, MODERATE field scope (title+features+details+description, excludes store/categories to avoid reintroducing the brand false-positive risk):")
    for name, count in hit_counts_moderate.items():
        delta = count - hit_counts[name]
        print(f"  {name:<10}{count:>7}/{total:<7}{count/total:>7.1%}   (+{delta} vs production scope)")

    print("\nSame vocab, WIDE field scope (title+features+details+description+categories+store, matches what evaluator itself searches):")
    for name, count in hit_counts_wide.items():
        delta = count - hit_counts[name]
        delta_mod = count - hit_counts_moderate[name]
        print(f"  {name:<10}{count:>7}/{total:<7}{count/total:>7.1%}   (+{delta} vs production scope, +{delta_mod} vs moderate scope)")

    print("\nCross-check against evaluator's OWN narrow regex (used to build customer disclosures), PRODUCTION scope vs WIDE scope:")
    print(f"  evaluator material regex hits: {evaluator_material_hits}/{total} ({evaluator_material_hits/total:.1%})")
    print(f"    production-scope misses:    {our_miss_evaluator_material_hit} ({our_miss_evaluator_material_hit/max(evaluator_material_hits,1):.1%} of evaluator's hits)")
    print(f"    wide-scope misses:          {wide_miss_evaluator_material_hit} ({wide_miss_evaluator_material_hit/max(evaluator_material_hits,1):.1%} of evaluator's hits)  <- true vocabulary gap")
    if missed_material_words:
        print(f"    production-scope missed words: {dict(sorted(missed_material_words.items(), key=lambda kv: -kv[1]))}")
    print(f"  evaluator color regex hits:    {evaluator_color_hits}/{total} ({evaluator_color_hits/total:.1%})")
    print(f"    production-scope misses:    {our_miss_evaluator_color_hit} ({our_miss_evaluator_color_hit/max(evaluator_color_hits,1):.1%} of evaluator's hits)")
    print(f"    wide-scope misses:          {wide_miss_evaluator_color_hit} ({wide_miss_evaluator_color_hit/max(evaluator_color_hits,1):.1%} of evaluator's hits)  <- true vocabulary gap")
    if missed_color_words:
        print(f"    production-scope missed words: {dict(sorted(missed_color_words.items(), key=lambda kv: -kv[1]))}")


if __name__ == "__main__":
    main()
