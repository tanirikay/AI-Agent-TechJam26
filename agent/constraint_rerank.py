"""Deterministic exact-clause evidence reranking.

This module deliberately contains no learned or approximate matching.  Raw
constraint text and catalog evidence go through the same normalization, then
candidate products receive one point for each active constraint that appears
as a contiguous substring of their evidence text.  Python's stable sort keeps
the incoming BM25/budget order whenever scores tie.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Protocol


EVIDENCE_FIELDS = (
    "title", "categories", "features", "details", "store", "description",
)


class RawConstraintLike(Protocol):
    normalized_text: str
    active: bool


def normalize_constraint_text(text: object) -> str:
    """Lowercase text and erase punctuation/whitespace differences.

    Replacing punctuation with spaces (rather than deleting it) preserves word
    order and boundaries: ``"Hand-wash"`` and ``"Hand wash"`` normalize the
    same way, while unrelated neighboring words are not concatenated.
    """

    lowered = str(text or "").lower().replace("_", " ")
    return re.sub(r"\s+", " ", re.sub(r"[^\w]+", " ", lowered)).strip()


def _flatten_evidence(value: object) -> Iterable[str]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from _flatten_evidence(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _flatten_evidence(item)
    elif value not in (None, ""):
        yield str(value)


def build_normalized_product_evidence(product: dict) -> str:
    """Flatten all ranking evidence, including detail keys and values."""

    parts: list[str] = []
    for field_name in EVIDENCE_FIELDS:
        parts.extend(_flatten_evidence(product.get(field_name)))
    return normalize_constraint_text(" ".join(parts))


def evidence_score(evidence: str, constraints: Iterable[RawConstraintLike]) -> int:
    """Count distinct active exact clauses present in normalized evidence."""

    active_clauses = dict.fromkeys(
        constraint.normalized_text
        for constraint in constraints
        if constraint.active and constraint.normalized_text
    )
    return sum(clause in evidence for clause in active_clauses)


def rerank_by_exact_constraints(
    pool_asins: list[str],
    normalized_evidence: dict[str, str],
    constraints: Iterable[RawConstraintLike],
    window: int,
) -> list[str]:
    """Stably rerank the first ``window`` candidates by exact-clause count."""

    if window <= 0 or not pool_asins:
        return list(pool_asins)

    active_constraints = tuple(
        constraint
        for constraint in constraints
        if constraint.active and constraint.normalized_text
    )
    if not active_constraints:
        return list(pool_asins)

    window_size = min(window, len(pool_asins))
    head = pool_asins[:window_size]
    tail = pool_asins[window_size:]
    ranked_head = sorted(
        head,
        key=lambda asin: -evidence_score(
            normalized_evidence.get(asin, ""), active_constraints
        ),
    )
    return ranked_head + tail
