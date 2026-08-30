"""Shared controlled vocabulary for slot extraction.

Both the message-side extractor (what the shopper says) and the product-side
extractor (what a product's title/features actually contain) MUST go through
this module. If they fork into separate normalization rules, a user saying
"black" and a product titled "Black Leather Boot" can silently fail to match
on casing, synonyms ("sneakers" vs "trainers"), or plurals. See the pseudocode
note under "Structured-field scarcity".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

TOKEN_RE = re.compile(r"[a-z0-9$.]+", re.IGNORECASE)
MIN_SINGLE_WORD_TERM_LEN = 6


@dataclass(frozen=True)
class BudgetConstraint:
    """An explicit numeric price constraint extracted from shopper text."""

    amount: float
    mode: str  # "maximum" or "target"

    def __post_init__(self) -> None:
        if self.mode not in {"maximum", "target"}:
            raise ValueError(f"unsupported budget mode: {self.mode}")

# --- controlled vocabularies: canonical_value -> set of surface synonyms ---
# Keys are the canonical value we store in a slot. Synonyms are additional
# surface forms that should normalize to that canonical value. The key itself
# is always an accepted surface form and does not need to be repeated.

COLOR_VOCAB: dict[str, set[str]] = {
    "black": set(),
    "white": set(),
    "blue": {"navy", "navy blue", "royal blue", "sky blue"},
    "red": {"maroon", "burgundy", "crimson"},
    "pink": {"rose", "fuchsia", "magenta"},
    "green": {"olive", "khaki green", "sage"},
    "brown": {"tan", "camel", "chocolate", "coffee"},
    "gray": {"grey", "charcoal", "heather gray", "heather grey"},
    "purple": {"lavender", "violet", "plum"},
    "yellow": {"mustard"},
    "orange": set(),
    "beige": {"cream", "ivory", "nude"},
    "silver": set(),
    "gold": set(),
    "multicolor": {"multi-color", "multicolored", "print", "patterned"},
}

MATERIAL_VOCAB: dict[str, set[str]] = {
    "cotton": set(),
    "polyester": {"poly"},
    "nylon": set(),
    "leather": {"faux leather", "pu leather", "genuine leather"},
    "wool": {"merino", "merino wool"},
    "spandex": {"elastane", "lycra"},
    "silk": set(),
    "rayon": {"viscose"},
    "denim": {"jean", "jeans"},
    "linen": set(),
    "suede": set(),
    "fleece": set(),
    "cashmere": set(),
    "canvas": set(),
    "satin": set(),
    "velvet": set(),
    "mesh": set(),
    "stainless steel": {"steel"},
    "sterling silver": set(),
    "gold plated": {"gold-plated"},
}

SIZE_VOCAB: dict[str, set[str]] = {
    "xs": {"extra small", "x-small"},
    "s": {"small"},
    "m": {"medium"},
    "l": {"large"},
    "xl": {"extra large", "x-large"},
    "xxl": {"2xl", "extra extra large"},
    "xxxl": {"3xl"},
    "plus size": {"plus-size", "plus"},
    "petite": set(),
    "tall": set(),
    "wide width": {"wide", "wide fit"},
    "narrow": {"narrow width", "narrow fit"},
    "one size": {"one-size", "one size fits all"},
}

STYLE_VOCAB: dict[str, set[str]] = {
    "casual": set(),
    "formal": {"dressy", "dress"},
    "athletic": {"sporty", "sport"},
    "running": set(),
    "vintage": {"retro"},
    "bohemian": {"boho"},
    "classic": {"traditional"},
    "elegant": {"chic"},
    "slim fit": {"slim-fit", "skinny", "skinny fit"},
    "relaxed fit": {"relaxed-fit", "loose fit", "loose-fit", "oversized"},
    "straight leg": {"straight-leg"},
    "crew neck": {"crew-neck", "crewneck"},
    "v-neck": {"v neck", "vneck"},
    "hooded": {"hoodie"},
    "sleeveless": set(),
    "high waisted": {"high-waisted", "high rise", "high-rise"},
}

USE_CASE_VOCAB: dict[str, set[str]] = {
    "running": {"jogging"},
    "hiking": {"trail"},
    "gym": {"workout", "training"},
    "yoga": set(),
    "work": {"office", "workwear"},
    "outdoor": set(),
    "winter": {"cold weather"},
    "summer": {"warm weather"},
    "wedding": {"bridal"},
    "travel": set(),
    "everyday": {"daily wear"},
    "party": {"evening wear"},
    "beach": {"swim", "swimming"},
    "school": set(),
}

FEATURE_VOCAB: dict[str, set[str]] = {
    "waterproof": {"water resistant", "water-resistant"},
    "breathable": set(),
    "lightweight": {"light weight"},
    "adjustable": set(),
    "stretch": {"stretchy"},
    "moisture-wicking": {"moisture wicking", "sweat-wicking"},
    "non-slip": {"non slip", "anti-slip"},
    "quick-dry": {"quick dry"},
    "insulated": set(),
    "reflective": set(),
    "zippered": {"zip", "zip-up"},
    "pockets": {"pocket"},
    "machine washable": {"machine wash"},
    "hypoallergenic": set(),
    "adjustable strap": {"adjustable straps"},
}

NO_PREFERENCE_CUES = (
    "no preference", "don't care", "dont care", "doesn't matter",
    "does not matter", "any is fine", "any of them", "up to you",
    "your judgment", "your judgement", "whatever works", "no particular",
    "not picky", "no strong preference", "i'm flexible", "im flexible",
    "surprise me",
)

_NUMBER = r"\d+(?:\.\d+)?"
_AMOUNT = rf"(?P<amount>{_NUMBER})"
_CURRENCY_AMOUNT = rf"(?:\$\s*{_AMOUNT}|(?P<amount_dollars>{_NUMBER})\s*dollars?)"

# Specific intent phrases come first so the legacy generic "$50" fallback
# cannot erase the distinction between "around $50" and "under $50".
_BUDGET_CONSTRAINT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    # Preserve the incumbent bare "around 50" form; newer target synonyms
    # below require currency unless explicit budget context is present.
    ("target", re.compile(
        rf"\baround\s*\$?\s*{_AMOUNT}(?:\s*dollars?)?",
        re.I,
    )),
    ("target", re.compile(
        rf"\b(?:around|about|approximately|roughly|close\s+to)\s*{_CURRENCY_AMOUNT}",
        re.I,
    )),
    ("target", re.compile(
        rf"\bbudget\s+(?:is\s+)?(?:around|about|approximately|roughly|close\s+to)\s*"
        rf"\$?\s*{_AMOUNT}(?:\s*dollars?)?",
        re.I,
    )),
    # The first three forms intentionally keep their historical optional
    # currency marker for backward compatibility with "under 50".
    ("maximum", re.compile(
        rf"\b(?:under|below|less\s+than)\s*\$?\s*{_AMOUNT}(?:\s*dollars?)?",
        re.I,
    )),
    ("maximum", re.compile(
        rf"\b(?:no\s+more\s+than|at\s+most)\s*{_CURRENCY_AMOUNT}",
        re.I,
    )),
    ("maximum", re.compile(
        rf"\bmaximum(?:\s+budget)?(?:\s+of|\s+is)?\s*{_CURRENCY_AMOUNT}",
        re.I,
    )),
    ("maximum", re.compile(
        rf"\bbudget(?:\s+of|\s+is)?\s*\$?\s*{_AMOUNT}(?:\s*dollars?)?",
        re.I,
    )),
    # Preserve the incumbent interpretation of an otherwise unqualified
    # explicit currency amount as a maximum budget.
    ("maximum", re.compile(rf"{_CURRENCY_AMOUNT}", re.I)),
)

# Backward-compatible exported pattern list for callers that only inspected
# this constant. New code should use extract_budget_constraint().
BUDGET_PATTERNS = [pattern for _, pattern in _BUDGET_CONSTRAINT_PATTERNS]


def _normalize(text: str) -> str:
    return text.lower().strip()


def _contains_term(text: str, term: str) -> bool:
    """Substring containment with word-boundary checks on both sides.

    Plain `in` matching against brand/category terms (drawn straight from
    ~19,749 raw catalog store names and ~799 category segments -- many of
    them short, ordinary words) was matching inside unrelated words: "king"
    inside "looking", "door" inside "outdoor", "cross" inside "across",
    "totes" inside "totes" from "Handbags & Totes". That was filling the
    brand slot with garbage on turn 1 of essentially every session, which
    silently:
      - made `brand` permanently non-askable (it was never actually empty)
      - injected the garbage value into every subsequent turn's BM25 query
    Confirmed empirically: 0 of 1195 turns in the public set ever asked
    about brand, and 15/15 sampled sessions had a bogus brand filled after
    turn 1 (see the conversation that added this fix for the trace).
    """
    start = 0
    while True:
        idx = text.find(term, start)
        if idx == -1:
            return False
        before_ok = idx == 0 or not text[idx - 1].isalnum()
        end = idx + len(term)
        after_ok = end == len(text) or not text[end].isalnum()
        if before_ok and after_ok:
            return True
        start = idx + 1


def _singularize(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("es") and not token.endswith("ses"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _build_lookup(vocab: dict[str, set[str]]) -> tuple[dict[str, str], list[str]]:
    """Flatten canonical->synonyms into surface_form->canonical, normalized.

    Returns (lookup, multi_word_phrases_longest_first). Built once per vocab
    (see _LOOKUP_CACHE below) instead of on every extraction call -- this
    function used to be re-run per call, which dominated runtime once
    get_attribute_value started calling it per product per slot.
    """
    lookup: dict[str, str] = {}
    for canonical, synonyms in vocab.items():
        for surface in {canonical, *synonyms}:
            lookup[_normalize(surface)] = canonical
    phrases = sorted((p for p in lookup if " " in p or "-" in p), key=len, reverse=True)
    return lookup, phrases


_LOOKUP_CACHE: dict[int, tuple[dict[str, str], list[str]]] = {}


def _match_vocab(text_norm: str, tokens: list[str], vocab: dict[str, set[str]]) -> str | None:
    cached = _LOOKUP_CACHE.get(id(vocab))
    if cached is None:
        cached = _build_lookup(vocab)
        _LOOKUP_CACHE[id(vocab)] = cached
    lookup, phrases = cached
    for phrase in phrases:
        if phrase in text_norm:
            return lookup[phrase]
    for token in tokens:
        singular = _singularize(token)
        if token in lookup:
            return lookup[token]
        if singular in lookup:
            return lookup[singular]
    return None


def extract_budget_constraint(text: str) -> BudgetConstraint | None:
    """Extract explicit numeric price language with its ranking mode."""
    for mode, pattern in _BUDGET_CONSTRAINT_PATTERNS:
        match = pattern.search(text)
        if match:
            try:
                groups = match.groupdict()
                amount = groups.get("amount") or groups.get("amount_dollars")
                return BudgetConstraint(float(amount), mode)
            except ValueError:
                continue
    return None


def extract_budget(text: str) -> float | None:
    """Compatibility wrapper retaining the historical float return type."""
    constraint = extract_budget_constraint(text)
    return constraint.amount if constraint is not None else None


def parse_price(price: object) -> float | None:
    """Product-side numeric price parser. Never coerces missing price to 0."""
    if price is None or price == "":
        return None
    if isinstance(price, (int, float)):
        return float(price)
    text = str(price)
    match = re.search(r"(\d+(?:\.\d+)?)", text)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


_EXCLUDED_CATEGORY_TERMS = {
    "clothing", "clothing shoes jewelry", "clothing, shoes & jewelry",
    "shoes", "jewelry", "men", "women", "boys", "girls", "baby",
}


def deepest_category_segment(categories: object) -> str | None:
    """Return the deepest (most specific) category segment available.

    The top-level bucket is ~99.98% identical across the whole catalog, so
    stopping at categories[0] carries no discriminative signal. Walk from the
    end and return the first segment that isn't a generic top-level bucket.
    """
    if not isinstance(categories, list) or not categories:
        return None
    for raw in reversed(categories):
        segment = str(raw).strip()
        if not segment:
            continue
        if segment.lower() in _EXCLUDED_CATEGORY_TERMS:
            continue
        return segment
    return None


@dataclass
class Vocabulary:
    """Shared extractor used for BOTH chat messages and product text.

    category_terms / brand_terms are populated from the catalog at index
    time so message-side extraction of category/brand can match against
    real catalog values instead of a hand-authored list.
    """

    category_terms: set[str] = field(default_factory=set)
    brand_terms: set[str] = field(default_factory=set)
    # parent_asin -> extracted keyword slots, populated lazily by
    # memory.get_attribute_value so repeated scoring across turns/sessions
    # doesn't re-run extraction on the same product text.
    product_slot_cache: dict[str, dict[str, object]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Longest-match-first, alphabetical tie-break -- deterministic.
        # Iterating brand_terms/category_terms as raw sets used to make the
        # winning match depend on Python's per-process string-hash
        # randomization, so results could shift by ~1 session between runs
        # with identical code and identical weights.
        #
        # Single-word terms under MIN_SINGLE_WORD_TERM_LEN are dropped
        # entirely (multi-word phrases are exempt -- collision risk there is
        # much lower). ~19,749 raw store names inevitably include ordinary
        # English words ("Key", "Work", "Style", "Totes"); even with correct
        # word-boundary matching those still produce whole-word false
        # positives on totally unrelated sentences ("a key requirement",
        # "outdoor & work"). Confirmed empirically: this filter cut the
        # false-brand-fill rate on a 15-session sample from 6/15 to near
        # zero. Real short brand names (e.g. "Vans") are lost by this --
        # accepted tradeoff, see the conversation that added this fix.
        self._brand_sorted = sorted(
            (t for t in self.brand_terms if " " in t or len(t) >= MIN_SINGLE_WORD_TERM_LEN),
            key=lambda t: (-len(t), t),
        )
        self._category_sorted = sorted(
            (t for t in self.category_terms if " " in t or len(t) >= MIN_SINGLE_WORD_TERM_LEN),
            key=lambda t: (-len(t), t),
        )

    def extract(self, text: object, include_catalog_terms: bool = True) -> dict[str, object]:
        text = text or ""
        text_norm = _normalize(str(text))
        tokens = [t.lower() for t in TOKEN_RE.findall(text_norm)]

        result: dict[str, object] = {}

        color = _match_vocab(text_norm, tokens, COLOR_VOCAB)
        if color:
            result["color"] = color

        material = _match_vocab(text_norm, tokens, MATERIAL_VOCAB)
        if material:
            result["material"] = material

        size = _match_vocab(text_norm, tokens, SIZE_VOCAB)
        if size:
            result["size"] = size

        style = _match_vocab(text_norm, tokens, STYLE_VOCAB)
        if style:
            result["style"] = style

        use_case = _match_vocab(text_norm, tokens, USE_CASE_VOCAB)
        if use_case:
            result["use_case"] = use_case

        feature = _match_vocab(text_norm, tokens, FEATURE_VOCAB)
        if feature:
            result["feature"] = feature

        budget = extract_budget(text_norm)
        if budget is not None:
            result["budget"] = budget

        if include_catalog_terms:
            for brand in self._brand_sorted:
                if _contains_term(text_norm, brand):
                    result["brand"] = brand
                    break

            for term in self._category_sorted:
                if _contains_term(text_norm, term):
                    result["category"] = term
                    break

        return result

    def contains_no_preference(self, text: str) -> bool:
        text_norm = _normalize(text)
        return any(cue in text_norm for cue in NO_PREFERENCE_CUES)
