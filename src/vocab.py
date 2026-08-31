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
    "black": {"jet black", "onyx black", "pitch black", "obsidian", "midnight black", "ebony", "coal black", "ink black"},
    "white": {"pure white", "snow white", "bright white", "chalk white", "off-white", "milk white"},
    "blue": {"navy", "navy blue", "royal blue", "sky blue", "cobalt", "cerulean", "teal blue", "indigo", "sapphire", "denim blue", "baby blue", "powder blue", "midnight blue", "azure"},
    "red": {"maroon", "burgundy", "crimson", "scarlet", "wine", "brick red", "ruby", "cherry red", "blood red", "rust red", "vermilion"},
    "pink": {"rose", "fuchsia", "magenta", "salmon", "coral pink", "blush", "hot pink", "baby pink", "dusty rose", "carnation", "bubblegum"},
    "green": {"olive", "khaki green", "sage", "emerald", "forest green", "hunter green", "mint", "lime", "moss green", "army green", "seafoam", "jade", "pine green", "pistachio"},
    "brown": {"tan", "camel", "chocolate", "coffee", "mocha", "walnut", "cognac", "chestnut", "espresso", "tawny", "hazel", "caramel", "cacao", "copper", "mahogany"},
    "gray": {"grey", "charcoal", "heather gray", "heather grey", "slate", "smoke", "gunmetal", "ash", "ash gray", "ash grey", "stone gray", "granite", "silver gray"},
    "purple": {"lavender", "violet", "plum", "mauve", "orchid", "eggplant", "lilac", "amethyst", "grape", "mulberry", "periwinkle"},
    "yellow": {"mustard", "lemon", "canary", "gold yellow", "amber", "sunflower", "butter yellow", "blonde"},
    "orange": {"tangerine", "rust", "burnt orange", "coral", "peach", "apricot", "terracotta", "marigold", "pumpkin"},
    # "natural" deliberately excluded from beige's synonyms despite being in
    # the source proposal -- too common in unrelated retail copy ("natural
    # fiber", "all-natural"), same false-positive shape as the brand
    # "king"/"looking" bug this file already documents.
    "beige": {"cream", "ivory", "nude", "sand", "khaki", "oatmeal", "ecru", "taupe", "biscuit", "stone"},
    "silver": {"chrome", "platinum", "metallic silver", "sterling", "pewter"},
    "gold": {"golden", "rose gold", "yellow gold", "metallic gold", "champagne gold", "bronze"},
    # "multi" deliberately excluded despite being in the source proposal --
    # TOKEN_RE doesn't treat "-" as a token character, so "multi-pocket"
    # (a FEATURE_VOCAB synonym below) tokenizes as ["multi", "pocket"],
    # which would spuriously fill color=multicolor on any multi-pocket
    # feature mention. Demonstrated, not hypothetical -- caught by
    # cross-checking this batch against itself before adopting.
    "multicolor": {"multi-color", "multicolored", "print", "patterned", "rainbow", "tie-dye", "ombre", "variegated", "colorblock", "color-block", "multicolour", "multicoloured"},
}

MATERIAL_VOCAB: dict[str, set[str]] = {
    "cotton": {"organic cotton", "combed cotton", "cotton blend", "pima cotton", "egyptian cotton", "ring-spun cotton", "canvas cotton", "cotton twill"},
    "polyester": {"poly", "polyester blend", "poly blend", "recycled polyester", "microfiber", "dacron", "terylene"},
    "nylon": {"ripstop nylon", "polyamide", "ballistic nylon", "cordura"},
    "leather": {"faux leather", "pu leather", "genuine leather", "vegan leather", "leatherette", "full grain leather", "top grain leather", "nappa leather", "nappa", "pleather"},
    "wool": {"merino", "merino wool", "lambswool", "virgin wool", "boiled wool", "wool blend", "tweed", "sheepswool"},
    "spandex": {"elastane", "lycra", "elastane blend", "stretch spandex"},
    # "silk satin" deliberately omitted from BOTH silk and satin -- it was
    # in the source proposal for both, a genuine same-string collision
    # (not resolved by phrase-priority, which only disambiguates DIFFERENT
    # strings like "khaki" vs "khaki green"). Left unregistered as a
    # phrase; token-level matching still resolves it deterministically to
    # "silk" (the first matching token encountered), which is fine.
    "silk": {"mulberry silk", "raw silk", "silk blend", "charmeuse", "tussar silk"},
    "rayon": {"viscose", "modal", "lyocell", "tencel", "bamboo rayon", "cupro"},
    "denim": {"jean", "jeans", "chambray", "raw denim", "selvedge denim", "stretch denim"},
    "linen": {"linen blend", "pure linen", "flax", "irish linen"},
    "suede": {"faux suede", "suede leather", "microsuede", "split suede"},
    "fleece": {"polar fleece", "sherpa", "fleece lined", "microfleece", "french terry"},
    "cashmere": {"pashmina", "cashmere blend", "mongolian cashmere"},
    "canvas": {"duck canvas", "cotton canvas", "heavyweight canvas"},
    "satin": {"poly satin", "duchess satin", "sateen"},
    "velvet": {"velour", "crushed velvet", "cotton velvet", "plush"},
    "mesh": {"netting", "breathable mesh", "fishnet", "power mesh"},
    "stainless steel": {"steel", "surgical steel", "stainless"},
    "sterling silver": {"925 silver", "925 sterling silver", "sterling"},
    "gold plated": {"gold-plated", "gold vermeil", "gold-filled", "gold dipped", "18k gold plated", "14k gold plated"},
}

SIZE_VOCAB: dict[str, set[str]] = {
    "xs": {"extra small", "x-small", "xsmall"},
    "s": {"small", "sm"},
    "m": {"medium", "med"},
    "l": {"large", "lg"},
    "xl": {"extra large", "x-large", "xlarge"},
    "xxl": {"2xl", "extra extra large", "2x", "xx-large", "double xl"},
    "xxxl": {"3xl", "3x", "xxxlarge", "triple xl"},
    "plus size": {"plus-size", "plus", "curvy", "extended sizes", "full figure"},
    "petite": {"petite fit", "short length"},
    "tall": {"tall fit", "long length", "extra long"},
    # "ew" deliberately excluded from wide width despite being in the
    # source proposal -- a very common informal interjection ("ew, gross"),
    # same risk category as "natural"/"multi" above. "ee" kept: it's a real
    # US shoe-width designation and much less likely to appear as ordinary
    # English text.
    "wide width": {"wide", "wide fit", "extra wide", "ee", "wide toe box"},
    "narrow": {"narrow width", "narrow fit", "slim width"},
    "one size": {"one-size", "one size fits all", "os", "osfa", "universal fit"},
}

STYLE_VOCAB: dict[str, set[str]] = {
    "casual": {"everyday style", "informal", "laid-back", "relaxed style"},
    "formal": {"dressy", "dress", "business formal", "cocktail", "black tie", "black-tie", "semi-formal", "evening wear"},
    "athletic": {"sporty", "sport", "activewear", "performance", "athleisure", "sportswear"},
    "running": {"runner style", "running gear"},
    "vintage": {"retro", "throwback", "antique", "y2k", "90s style", "80s style", "heritage"},
    "bohemian": {"boho", "boho chic", "peasant style", "ethnic print"},
    "classic": {"traditional", "timeless", "preppy", "minimalist"},
    "elegant": {"chic", "sophisticated", "glam", "glamorous", "refined", "classy"},
    "slim fit": {"slim-fit", "skinny", "skinny fit", "tailored fit", "tapered fit"},
    "relaxed fit": {"relaxed-fit", "loose fit", "loose-fit", "oversized", "baggy", "slouchy", "roomy fit"},
    "fitted": {"body-hugging", "form fitting", "form-fitting", "snug fit", "bodycon", "tight fit"},
    "cropped": {"crop", "crop top", "cropped length", "short cut"},
    "straight leg": {"straight-leg", "straight cut", "straight fit"},
    "crew neck": {"crew-neck", "crewneck", "round neck", "round-neck"},
    "v-neck": {"v neck", "vneck", "v-cut"},
    "hooded": {"hoodie", "with hood", "hooded style"},
    "sleeveless": {"no sleeves", "tank top", "tank style"},
    "high waisted": {"high-waisted", "high rise", "high-rise", "high waist"},
}

USE_CASE_VOCAB: dict[str, set[str]] = {
    "running": {"jogging", "marathon", "sprinting", "road running", "trail running"},
    "hiking": {"trail", "trekking", "backpacking", "mountaineering", "outdoor walking"},
    "gym": {"workout", "training", "fitness", "exercise", "weightlifting", "crossfit", "bodybuilding"},
    "yoga": {"pilates", "barre", "stretching", "meditation"},
    "work": {"office", "workwear", "business", "professional", "corporate", "commute"},
    "outdoor": {"camping", "outdoors", "excursion", "nature walk"},
    "winter": {"cold weather", "snow", "skiing", "snowboarding", "sub-zero"},
    "summer": {"warm weather", "hot weather", "resort wear", "sunny weather"},
    "wedding": {"bridal", "bridesmaid", "groomsman", "wedding guest"},
    "travel": {"vacation", "holiday", "airplane", "road trip", "touring"},
    "everyday": {"daily wear", "casual wear", "lounging", "lounge", "streetwear"},
    "party": {"evening wear", "night out", "clubwear", "celebration", "festive"},
    "beach": {"swim", "swimming", "poolside", "resort", "seaside", "surf"},
    "school": {"college", "university", "campus", "back to school"},
    "cycling": {"biking", "road cycling", "mountain biking", "spin class"},
    "golf": {"golfing", "country club"},
    "tennis": {"pickleball", "racquet sports", "court sports"},
}

FEATURE_VOCAB: dict[str, set[str]] = {
    "waterproof": {"water resistant", "water-resistant", "weatherproof", "rainproof", "impermeable"},
    "breathable": {"ventilated", "air-permeable", "air circulation", "mesh ventilated"},
    "lightweight": {"light weight", "featherweight", "ultra-light", "ultralight"},
    "adjustable": {"adjustable fit", "customizable fit", "cinchable"},
    "stretch": {"stretchy", "4-way stretch", "2-way stretch", "flexible", "elastic"},
    "moisture-wicking": {"moisture wicking", "sweat-wicking", "dry-fit", "wicking", "dri-fit"},
    "non-slip": {"non slip", "anti-slip", "slip resistant", "high traction", "grip sole"},
    "quick-dry": {"quick dry", "fast-drying", "fast drying", "quick drying"},
    "insulated": {"thermal", "fleece-lined", "thinsulate", "warm lined", "padded insulation"},
    "reflective": {"high-visibility", "hi-vis", "reflective detail", "night visibility"},
    "zippered": {"zip", "zip-up", "zipper closure", "full-zip", "half-zip", "quarter-zip"},
    "pockets": {"pocket", "pocketed", "multi-pocket", "zippered pockets", "cargo pockets"},
    "machine washable": {"machine wash", "easy care", "washable"},
    "hypoallergenic": {"sensitive skin safe", "nickel-free", "allergy friendly"},
    "adjustable strap": {"adjustable straps", "removable strap", "padded strap"},
    "padded": {"cushioned", "cushioning", "impact absorbing", "shock absorbing"},
    "arch support": {"orthotic", "supportive", "arch-support", "orthotic friendly", "footbed support"},
    "uv protection": {"upf", "sun protection", "upf 50+", "uv resistant"},
    "wrinkle-free": {"wrinkle resistant", "no-iron", "non-iron", "crease resistant"},
    "reversible": {"2-in-1", "dual-sided", "two-sided"},
}

# Coarser groupings over MATERIAL_VOCAB's canonical values, used only by
# extract_negated_values() below. A closed, hand-curated table over an
# already-closed ~19-material catalog vocabulary -- deliberately NOT an
# attempt at a general material taxonomy, just enough to resolve the
# category-level negations this catalog's clothing/shoe/jewelry materials
# actually produce ("not an animal fiber", "not synthetic"). Incomplete by
# construction, same caveat as every other hand-curated table in this file:
# an unanticipated phrasing still falls through uncaught. Four groupings
# span all 19 MATERIAL_VOCAB canonical values: animal-derived, synthetic/
# man-made, plant-based, and metal -- "natural" deliberately covers BOTH
# plant and animal fiber (common usage: "natural" means "not synthetic",
# not "not animal"), so it's the union, not a fifth disjoint group.
MATERIAL_CATEGORY_ALIASES: dict[str, set[str]] = {
    "animal": {"wool", "silk", "cashmere", "leather", "suede"},
    "animal fiber": {"wool", "silk", "cashmere", "leather", "suede"},
    "animal fibers": {"wool", "silk", "cashmere", "leather", "suede"},
    "animal-derived": {"wool", "silk", "cashmere", "leather", "suede"},
    "animal derived": {"wool", "silk", "cashmere", "leather", "suede"},
    "animal product": {"wool", "silk", "cashmere", "leather", "suede"},
    "animal products": {"wool", "silk", "cashmere", "leather", "suede"},
    "animal based": {"wool", "silk", "cashmere", "leather", "suede"},
    "animal-based": {"wool", "silk", "cashmere", "leather", "suede"},
    "synthetic": {"polyester", "nylon", "spandex", "rayon", "mesh", "fleece"},
    "synthetic fiber": {"polyester", "nylon", "spandex", "rayon", "mesh", "fleece"},
    "synthetic fibers": {"polyester", "nylon", "spandex", "rayon", "mesh", "fleece"},
    "synthetic material": {"polyester", "nylon", "spandex", "rayon", "mesh", "fleece"},
    "man-made": {"polyester", "nylon", "spandex", "rayon", "mesh", "fleece"},
    "man made": {"polyester", "nylon", "spandex", "rayon", "mesh", "fleece"},
    "man-made fiber": {"polyester", "nylon", "spandex", "rayon", "mesh", "fleece"},
    "manmade": {"polyester", "nylon", "spandex", "rayon", "mesh", "fleece"},
    "plant-based": {"cotton", "linen", "denim", "canvas"},
    "plant based": {"cotton", "linen", "denim", "canvas"},
    "plant fiber": {"cotton", "linen", "denim", "canvas"},
    "plant fibers": {"cotton", "linen", "denim", "canvas"},
    "vegetable fiber": {"cotton", "linen", "denim", "canvas"},
    "natural fiber": {"cotton", "linen", "denim", "canvas", "wool", "silk", "cashmere", "leather", "suede"},
    "natural fibers": {"cotton", "linen", "denim", "canvas", "wool", "silk", "cashmere", "leather", "suede"},
    "natural material": {"cotton", "linen", "denim", "canvas", "wool", "silk", "cashmere", "leather", "suede"},
    "natural materials": {"cotton", "linen", "denim", "canvas", "wool", "silk", "cashmere", "leather", "suede"},
    "metal": {"stainless steel", "sterling silver", "gold plated"},
    "metallic": {"stainless steel", "sterling silver", "gold plated"},
    "metal alloy": {"stainless steel", "sterling silver", "gold plated"},
}

# Cue phrases that introduce a negated/excluded span within a disclosed
# clause. Broader than a first pass, but still a closed, reviewed list
# rather than general negation parsing -- same discipline as
# CONTRADICTION_CUES and NO_PREFERENCE_CUES below. Bare "no" is
# deliberately excluded: it's extremely common in ordinary text ("no
# problem"), it would collide with contains_no_preference()'s own "no
# preference" handling, and unlike the other cues here it isn't reliably
# followed by the thing being ruled out.
_NEGATION_CUE_RE = re.compile(
    r"\b(?:"
    r"not|never|none\s+of|"
    r"isn['’]?t|aren['’]?t|wasn['’]?t|weren['’]?t|"
    r"doesn['’]?t|don['’]?t|won['’]?t|can['’]?t|cannot|"
    r"without|"
    r"unlike|instead\s+of|rather\s+than|other\s+than|"
    r"except|excluding|besides|apart\s+from|aside\s+from|"
    r"as\s+opposed\s+to|in\s+place\s+of"
    r")\b\s*(.+)",
    re.I,
)


def _match_vocab_all(text_norm: str, tokens: list[str], vocab: dict[str, set[str]]) -> set[str]:
    """Like _match_vocab, but returns EVERY distinct canonical value found,
    not just the first. A negation span can list more than one excluded
    value ("not wool or leather") -- _match_vocab (used by extract(), which
    fills a single-valued slot) deliberately stops at the first match, which
    is wrong for this use case.
    """
    cached = _LOOKUP_CACHE.get(id(vocab))
    if cached is None:
        cached = _build_lookup(vocab)
        _LOOKUP_CACHE[id(vocab)] = cached
    lookup, phrases = cached
    found: set[str] = set()
    for phrase in phrases:
        if phrase in text_norm:
            found.add(lookup[phrase])
    for token in tokens:
        singular = _singularize(token)
        if token in lookup:
            found.add(lookup[token])
        elif singular in lookup:
            found.add(lookup[singular])
    return found


_NEGATABLE_VOCABS: tuple[tuple[str, dict[str, set[str]]], ...] = (
    ("color", COLOR_VOCAB),
    ("material", MATERIAL_VOCAB),
    ("size", SIZE_VOCAB),
    ("style", STYLE_VOCAB),
    ("use_case", USE_CASE_VOCAB),
    ("feature", FEATURE_VOCAB),
)


def extract_negated_values(text: str, vocabulary: "Vocabulary") -> dict[str, set[str]]:
    """Return, per slot, canonical values a clause explicitly rules out.

    Two sources, both deterministic and exact -- no embedding involved:
    literal vocabulary terms found in the negated span (e.g. "not wool"),
    and MATERIAL_CATEGORY_ALIASES for category-level phrasing the plain
    vocabulary can't name directly (e.g. "not an animal fiber"). Used to
    exclude candidates from the semantic-similarity fallback bonus in
    agent.py BEFORE the embedding runs, rather than asking the embedding
    to understand negation -- which it demonstrably cannot (see CLAUDE.md
    iteration 7's negation finding).

    `vocabulary` is accepted for interface symmetry with the rest of this
    module but unused -- the module-level vocab dicts are matched directly
    via _match_vocab_all so multiple excluded values in one span are all
    captured, unlike Vocabulary.extract()'s single-value-per-slot design.

    Only ever adds exclusions; returns {} if no negation cue is present.
    Stops the negated span at the next clause boundary so "not wool, but
    soft" doesn't over-capture "soft" as also excluded.
    """
    match = _NEGATION_CUE_RE.search(text)
    if not match:
        return {}
    span = re.split(r"[,.;]", match.group(1), maxsplit=1)[0]
    span_norm = _normalize(span)
    span_tokens = [t.lower() for t in TOKEN_RE.findall(span_norm)]

    excluded: dict[str, set[str]] = {}
    for slot_name, vocab_dict in _NEGATABLE_VOCABS:
        found = _match_vocab_all(span_norm, span_tokens, vocab_dict)
        if found:
            excluded.setdefault(slot_name, set()).update(found)

    for alias, materials in MATERIAL_CATEGORY_ALIASES.items():
        if _contains_term(span_norm, alias):
            excluded.setdefault("material", set()).update(materials)

    return excluded


# contains_no_preference() below checks these via plain substring
# containment (`cue in text`), NOT the word-boundary-checked _contains_term
# or token-exact _match_vocab used everywhere else in this file -- so a
# short/common entry here is riskier than an equivalent vocab-dict entry.
# Confirmed empirically: "any" alone matches inside "many" ("how many
# colors"), and bare "either" matches inside "neither" -- both excluded for
# exactly that reason (the same false-positive shape as the brand
# `_contains_term` bug this file already documents, just with no
# word-boundary protection to catch it). The qualified phrasings ("any is
# fine", "either way", "anything is fine") are kept -- they're long/specific
# enough that the collision risk is negligible, and they cover the same
# intent without it.
NO_PREFERENCE_CUES = (
    # Base list
    "no preference", "don't care", "dont care", "doesn't matter",
    "does not matter", "any is fine", "any of them", "up to you",
    "your judgment", "your judgement", "whatever works", "no particular",
    "not picky", "no strong preference", "i'm flexible", "im flexible",
    "surprise me", "not sure", "idk",

    # Informal & conversational
    "whatever", "whichever", "any works", "anything works", "anything is fine",
    "either is fine", "either way", "either works", "whatever you think",
    "whatever you prefer", "dealer's choice", "dealers choice", "you decide",
    "you choose", "your call", "up to your discretion", "as you see fit",

    # Relaxed / indifferent
    "makes no difference", "it's all good", "its all good", "fine with anything",
    "happy with whatever", "open to anything", "no opinion", "no real preference",
    "no special preference", "don't mind", "dont mind", "doesn't bother me",
    "i don't mind", "i dont mind", "doesn't make a difference",

    # Short / slang / typos -- "any", bare "either", and bare "anything"
    # deliberately excluded, see the module comment above.
    "idc", "w/e", "watever", "doesn't matter to me",
    "doesnt matter to me", "anything's fine", "anythings fine", "doesn't matter much",

    # Indecisive / open-ended
    "hard to say", "not bothered", "could go either way", "no idea",
    "don't know", "dont know", "unsure", "i'm easy", "im easy",
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
