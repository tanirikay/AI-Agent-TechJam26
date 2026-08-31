"""Catalog indexing and candidate retrieval.

Not part of the memory/follow-up design doc (that module explicitly leaves
retrieval to a separate owner) -- this is the minimum retrieval layer needed
to run the memory module end-to-end against the catalog for testing.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from src.constraint_rerank import build_normalized_product_evidence
from src.vocab import Vocabulary, deepest_category_segment

TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "would", "you", "your",
}


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def _terms(text: str) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) > 1 and token.lower() not in STOPWORDS
    ]


@dataclass
class CatalogIndex:
    connection: sqlite3.Connection
    products: dict[str, dict]
    vocabulary: Vocabulary
    normalized_evidence: dict[str, str]


def build_catalog_index(catalog_path: str | Path) -> CatalogIndex:
    connection = sqlite3.connect(":memory:")
    cursor = connection.cursor()
    cursor.execute(
        "CREATE VIRTUAL TABLE products USING fts5("
        "parent_asin UNINDEXED, title, categories, features, details, store, description, "
        "tokenize='unicode61 remove_diacritics 2')"
    )

    products: dict[str, dict] = {}
    normalized_evidence: dict[str, str] = {}
    category_terms: set[str] = set()
    brand_terms: set[str] = set()

    batch: list[tuple[str, str, str, str, str, str, str]] = []
    with Path(catalog_path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            product = json.loads(line)
            parent_asin = str(product["parent_asin"])
            products[parent_asin] = product
            normalized_evidence[parent_asin] = build_normalized_product_evidence(product)

            deepest = deepest_category_segment(product.get("categories"))
            if deepest:
                category_terms.add(deepest.lower())
            store = product.get("store")
            if store:
                brand_terms.add(str(store).strip().lower())

            batch.append((
                parent_asin,
                _text(product.get("title")),
                _text(product.get("categories")),
                _text(product.get("features")),
                _text(product.get("details")),
                _text(product.get("store")),
                _text(product.get("description")),
            ))
            if len(batch) >= 1000:
                cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
                batch.clear()
    if batch:
        cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
    connection.commit()

    vocabulary = Vocabulary(category_terms=category_terms, brand_terms=brand_terms)
    return CatalogIndex(
        connection=connection,
        products=products,
        vocabulary=vocabulary,
        normalized_evidence=normalized_evidence,
    )


# Column order in the FTS5 table (see build_catalog_index): parent_asin is
# UNINDEXED so it always gets weight 0.0; the six weights below correspond to
# title, categories, features, details, store, description in that order.
#
# Chosen by scripts/tune_field_weights.py, re-run after fixing the brand/
# category false-positive extraction bug in vocab.py (naive substring
# matching against ~19,749 raw store names was matching "king" inside
# "looking", filling brand with garbage on turn 1 of nearly every session).
# Under that buggy pipeline "flatten weights" (4,4,3,3,3,2) won; once brand
# stopped injecting noise, "categories-dominant" took over cleanly on hit
# rate, MRR, and MTTC simultaneously -- consistent with the original catalog
# audit's finding that the deepest category segment is highly discriminative
# (unlike categories[0], which is ~99.98% identical catalog-wide).
DEFAULT_FIELD_WEIGHTS: tuple[float, float, float, float, float, float] = (5.0, 6.0, 2.5, 2.0, 3.0, 1.0)


def _match_expression(query_text: str, boost_terms=(), boost_k: int = 0) -> str:
    """OR-of-terms FTS5 MATCH expression.

    `boost_terms` (the tokenised confirmed-slot values) are appended `boost_k`
    EXTRA times without de-duplication. SQLite's bm25() sums IDF-weighted
    contributions per query-phrase occurrence, so a term repeated N times in
    the MATCH expression is scored ~N times heavier -- this is how you
    up-weight the constraints we are certain about relative to raw chat
    tokens, staying entirely inside BM25 (no rerank pass, no pool shrink:
    recall is unchanged because every original term is still present).
    Verified empirically that FTS5 rewards the repetition (iter 3 step 1).
    """
    unique_terms = list(dict.fromkeys(_terms(query_text)))[:60]
    terms = list(unique_terms)
    if boost_k and boost_terms:
        boost = [t for t in boost_terms if t]
        for _ in range(boost_k):
            terms.extend(boost)
    return " OR ".join(f'"{term}"' for term in terms)


def retrieve(
    index: CatalogIndex,
    query_text: str,
    rejected_asins: set[str],
    limit: int,
    field_weights: tuple[float, float, float, float, float, float] = DEFAULT_FIELD_WEIGHTS,
    boost_terms=(),
    boost_k: int = 0,
) -> list[str]:
    expression = _match_expression(query_text, boost_terms, boost_k)
    if not expression:
        return []
    fetch_limit = limit + len(rejected_asins)
    weight_args = ", ".join(f"{w:.4f}" for w in (0.0, *field_weights))
    rows = index.connection.execute(
        "SELECT parent_asin FROM products WHERE products MATCH ? "
        f"ORDER BY bm25(products, {weight_args}) LIMIT ?",
        (expression, fetch_limit),
    ).fetchall()
    result = [str(row[0]) for row in rows if str(row[0]) not in rejected_asins]
    return result[:limit]


# --------------------------------------------------------------------------
# Narrowing passes (Task 2)
#
# These sit ON TOP OF the OR-based recall pass above -- `retrieve()` is
# unchanged and still owns recall. Everything here only ever *shrinks or
# reorders an already-retrieved pool*, and always falls back to the full OR
# pool rather than returning nothing, so recall can never regress below the
# current behaviour.
#
# Motivation: because the recall query joins terms with OR, the pool can only
# grow as terms accumulate -- measured at exactly the `RETRIEVAL_POOL_SIZE`
# cap on 100% of turns both before and after the clarification-loop fix, which
# makes agent.py's `pool_threshold` ("small enough, just answer") dead code.
# --------------------------------------------------------------------------


def retrieve_scored(
    index: CatalogIndex,
    query_text: str,
    rejected_asins: set[str],
    limit: int,
    field_weights: tuple[float, float, float, float, float, float] = DEFAULT_FIELD_WEIGHTS,
    boost_terms=(),
    boost_k: int = 0,
) -> list[tuple[str, float]]:
    """Same as `retrieve()` but also returns each candidate's bm25 score.

    SQLite's bm25() is negative-is-better, so the best candidate has the
    smallest (most negative) value.
    """
    expression = _match_expression(query_text, boost_terms, boost_k)
    if not expression:
        return []
    fetch_limit = limit + len(rejected_asins)
    weight_args = ", ".join(f"{w:.4f}" for w in (0.0, *field_weights))
    rows = index.connection.execute(
        f"SELECT parent_asin, bm25(products, {weight_args}) AS score FROM products "
        "WHERE products MATCH ? ORDER BY score LIMIT ?",
        (expression, fetch_limit),
    ).fetchall()
    scored = [(str(row[0]), float(row[1])) for row in rows if str(row[0]) not in rejected_asins]
    return scored[:limit]


def narrow_by_score(scored: list[tuple[str, float]], keep_ratio: float) -> list[str]:
    """Keep candidates whose bm25 score is within `keep_ratio` of the best.

    Scores are negative (better = more negative), so "at least keep_ratio as
    good as the best" is `score <= keep_ratio * best_score`. keep_ratio=1.0
    keeps only ties with the best; 0.0 keeps everything.

    Pure truncation of the tail -- it does not reorder, so it cannot change
    which items occupy the top 10. Its only effect is on the ask-vs-answer
    decision.
    """
    if not scored:
        return []
    best = scored[0][1]
    cutoff = keep_ratio * best
    kept = [asin for asin, score in scored if score <= cutoff]
    return kept or [scored[0][0]]


def narrow_by_conjunction(
    index: CatalogIndex,
    pool_asins: list[str],
    core_terms: list[str],
    fetch_limit: int = 5000,
) -> list[str]:
    """Keep only pool members that contain ALL `core_terms` (AND semantics).

    Order is inherited from the pool (i.e. still BM25 order), so this is a
    filter, not a rerank. Unlike `narrow_by_score` this CAN change the top 10,
    because dropping a non-conforming candidate promotes the one behind it.
    Falls back to the unfiltered pool when the conjunction matches nothing.
    """
    terms = [t for t in dict.fromkeys(core_terms) if t]
    if not terms or not pool_asins:
        return pool_asins
    expression = " AND ".join(f'"{term}"' for term in terms)
    try:
        rows = index.connection.execute(
            "SELECT parent_asin FROM products WHERE products MATCH ? LIMIT ?",
            (expression, fetch_limit),
        ).fetchall()
    except sqlite3.OperationalError:
        return pool_asins
    allowed = {str(row[0]) for row in rows}
    if not allowed:
        return pool_asins
    kept = [asin for asin in pool_asins if asin in allowed]
    return kept or pool_asins
