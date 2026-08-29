"""Local components used by the submission agent."""

from __future__ import annotations

"""Stage 2 retrieval: hard filters plus hybrid lexical and semantic search."""

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import sqlite3


TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?", re.IGNORECASE)
PRICE_RE = re.compile(
    r"\b(?:under|below|less than|up to|max(?:imum)?)\s*\$?\s*(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking", "need",
}
COLORS = {
    "black", "white", "blue", "brown", "red", "green", "yellow", "pink", "purple",
    "orange", "grey", "gray", "beige", "navy", "tan", "gold", "silver",
}
MATERIALS = {
    "cotton", "leather", "linen", "silk", "wool", "denim", "polyester", "nylon",
    "suede", "cashmere", "velvet", "lace",
}

# Dependency-free semantic expansion for common fashion/shopping concepts.
CONCEPTS = {
    "sneaker": {"sneaker", "sneakers", "trainer", "trainers", "running", "athletic", "shoe", "shoes"},
    "beach": {"beach", "resort", "vacation", "summer", "sundress", "swim", "swimsuit", "coverup", "linen"},
    "formal": {"formal", "occasion", "evening", "cocktail", "party", "wedding", "dressy"},
    "warm": {"winter", "warm", "thermal", "fleece", "insulated", "coat", "jacket", "parka"},
    "casual": {"casual", "everyday", "relaxed", "comfortable", "lounge"},
    "workout": {"workout", "gym", "fitness", "yoga", "active", "athletic", "running"},
}
ALIAS_TO_CONCEPT = {
    word: concept
    for concept, words in CONCEPTS.items()
    for word in words
}


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {_text(item)}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(_text(item) for item in value)
    return str(value)


def _terms(text: str) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if token.lower() not in STOPWORDS
    ]


def _semantic_terms(text: str) -> list[str]:
    terms = _terms(text)
    concepts = [ALIAS_TO_CONCEPT[term] for term in terms if term in ALIAS_TO_CONCEPT]
    return list(dict.fromkeys(terms + concepts))


@dataclass(frozen=True)
class SearchResult:
    parent_asin: str
    score: float


@dataclass(frozen=True)
class _Product:
    parent_asin: str
    searchable: str
    price: float | None
    terms: Counter[str]


class HybridSearch:
    """In-memory BM25 and semantic-concept retrieval for the frozen catalog."""

    def __init__(self, catalog_path: str | Path) -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        self.products: list[_Product] = []
        self.postings: dict[str, dict[int, int]] = defaultdict(dict)
        self._build_index()

    def _build_index(self) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "tokenize='unicode61 remove_diacritics 2')"
        )

        batch: list[tuple[str, str, str, str, str, str, str]] = []
        with self.catalog_path.open(encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                item = json.loads(line)
                fields = (
                    str(item["parent_asin"]),
                    _text(item.get("title")),
                    _text(item.get("categories")),
                    _text(item.get("features")),
                    _text(item.get("details")),
                    _text(item.get("store")),
                    _text(item.get("description")),
                )

                searchable = " ".join(fields[1:]).lower()
                counts = Counter(_semantic_terms(searchable))
                self.products.append(
                    _Product(
                        parent_asin=fields[0],
                        searchable=searchable,
                        price=self._price(item.get("price")),
                        terms=counts,
                    )
                )
                for term, count in counts.items():
                    self.postings[term][index] = count

                batch.append(fields)
                if len(batch) == 1000:
                    cursor.executemany(
                        "INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)",
                        batch,
                    )
                    batch.clear()

        if batch:
            cursor.executemany(
                "INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)",
                batch,
            )
        self.connection.commit()

    @staticmethod
    def _price(value: object) -> float | None:
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    def retrieve(
        self,
        query: str,
        mode: str,
        candidate_count: int = 30,
    ) -> list[SearchResult]:
        """Return a 20–30 item retrieval shortlist; no Stage 3 re-ranking."""
        candidate_count = max(20, min(candidate_count, 30))

        # Hard constraints apply only in Buying mode and are never relaxed.
        eligible = self._hard_filter(query) if mode == "buying" else None

        lexical = self._bm25(query, eligible, candidate_count)
        semantic = self._semantic(query, eligible, candidate_count)

        lexical_scores = self._normalise(lexical)
        semantic_scores = self._normalise(semantic)

        # Exact terms matter more for Buying; concepts matter more for Browsing.
        lexical_weight, semantic_weight = (
            (0.72, 0.28) if mode == "buying" else (0.42, 0.58)
        )

        results = [
            SearchResult(
                parent_asin=self.products[index].parent_asin,
                score=(
                    lexical_weight * lexical_scores.get(index, 0.0)
                    + semantic_weight * semantic_scores.get(index, 0.0)
                ),
            )
            for index in set(lexical_scores) | set(semantic_scores)
        ]
        return sorted(
            results,
            key=lambda result: (-result.score, result.parent_asin),
        )[:candidate_count]

    def _hard_filter(self, query: str) -> set[int] | None:
        """Remove products violating stated Buying constraints before scoring."""
        query_terms = set(_terms(query))
        colors = query_terms & COLORS
        materials = query_terms & MATERIALS
        budget = PRICE_RE.search(query)
        size = self._requested_size(query)

        if not (colors or materials or budget or size):
            return None

        eligible: set[int] = set()
        for index, product in enumerate(self.products):
            if colors and not colors.issubset(product.terms):
                continue
            if materials and not materials.issubset(product.terms):
                continue
            if size and not re.search(
                rf"\b(?:size\s*)?{re.escape(size)}\b",
                product.searchable,
            ):
                continue
            if budget and (
                product.price is None
                or product.price > float(budget.group(1))
            ):
                continue
            eligible.add(index)

        return eligible

    @staticmethod
    def _requested_size(query: str) -> str | None:
        match = re.search(
            r"\bsize\s*(\d{1,2}(?:\.5)?)\b",
            query,
            re.IGNORECASE,
        )
        return match.group(1) if match else None

    def _bm25(
        self,
        query: str,
        eligible: set[int] | None,
        limit: int,
    ) -> dict[int, float]:
        terms = list(dict.fromkeys(_terms(query)))[:40]
        if not terms:
            return {}

        expression = " OR ".join(f'"{term}"' for term in terms)
        rows = self.connection.execute(
            "SELECT rowid - 1, bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0) "
            "FROM products WHERE products MATCH ? ORDER BY 2 LIMIT ?",
            (expression, max(limit * 8, 240)),
        ).fetchall()

        scores: dict[int, float] = {}
        for index, bm25 in rows:
            if eligible is not None and index not in eligible:
                continue
            scores[index] = 1.0 / (1.0 + max(0.0, float(bm25)))
            if len(scores) == limit:
                break
        return scores

    def _semantic(
        self,
        query: str,
        eligible: set[int] | None,
        limit: int,
    ) -> dict[int, float]:
        terms = _semantic_terms(query)
        if not terms:
            return {}

        scores: Counter[int] = Counter()
        product_count = max(1, len(self.products))

        for term in terms:
            posting = self.postings.get(term, {})
            idf = math.log((product_count + 1) / (len(posting) + 1)) + 1.0

            for index, frequency in posting.items():
                if eligible is None or index in eligible:
                    scores[index] += (1.0 + math.log(frequency)) * idf

        query_norm = math.sqrt(len(terms))
        return {
            index: score / (
                query_norm
                * math.sqrt(max(1, sum(self.products[index].terms.values())))
            )
            for index, score in scores.most_common(limit)
        }

    @staticmethod
    def _normalise(scores: dict[int, float]) -> dict[int, float]:
        if not scores:
            return {}

        maximum = max(scores.values())
        if not maximum:
            return {index: 0.0 for index in scores.items()}

        return {
            index: score / maximum
            for index, score in scores.items()
        }