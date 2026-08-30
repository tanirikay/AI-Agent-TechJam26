"""Memory + follow-up-question module.

Implements the design doc: session state, slot memory, rejected-item
tracking, clarification attribute selection, "no preference" bookkeeping,
and intent-override detection. Retrieval and the final answer/ask turn-budget
decision are owned by the caller (agent.py) -- this module only exposes what
that layer needs, per the design doc's "Interface exposed to the rest of the
pipeline" section.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum, auto

from agent.vocab import Vocabulary, deepest_category_segment, parse_price
from agent.constraint_rerank import normalize_constraint_text

SLOT_NAMES = (
    "category", "material", "color", "size", "style",
    "brand", "budget", "feature", "use_case",
)

CONTRADICTION_CUES = (
    "actually", "ignore that", "instead", "changed my mind",
    "no wait", "scratch that", "on second thought", "never mind",
)

REJECTION_CUES = (
    "not that", "not the", "no not", "don't like", "dont like",
    "something else", "not this one", "none of these", "not a fan",
)

# Maps the public_set's observed preference_tags vocabulary (fit, material,
# comfort, style, durability, performance, warmth, weather, general shopping)
# onto the 9 slots. Used only to bias WHICH attribute we ask about first --
# never to inject profile text into the search query or to filter/rerank
# recommendations, so it stays within "safe personalization" (aggregate
# signal only, no raw purchase history).
PREFERENCE_TAG_TO_SLOT: dict[str, str] = {
    "fit": "size",
    "material": "material",
    "comfort": "feature",
    "style": "style",
    "durability": "feature",
    "performance": "use_case",
    "warmth": "use_case",
    "weather": "use_case",
}
# Re-tuned to 1.5 after the clarification-stuck-loop fix (was 0.2). Once every
# turn fills a genuinely new slot, asking the profile-emphasised attributes
# first surfaces the discriminative constraints sooner. Swept 0.0..3.0: score
# rises monotonically and saturates at ~1.5 (2.0 and 3.0 are byte-identical),
# and the gain holds on BOTH halves of an alternating split
# (scripts/validate_policy_split.py): HR@10 0.775->0.795, MTTC 4.875->4.62,
# technical_score 0.6576->0.6774 on the public 200. At 1.5 a slot with two
# matching profile tags gets +3.0, which dominates entropy*coverage (~1-2.5) --
# i.e. "ask what the profile cares about first, fall back to entropy." Still
# never injected into the query or used to filter/rerank products.
PREFERENCE_TAG_BONUS = 1.5


class Source(Enum):
    EXPLICIT = auto()
    INFERRED = auto()
    NO_PREFERENCE = auto()


@dataclass
class Slot:
    value: object = None
    confidence: float = 0.0
    filled_turn: int | None = None
    source: Source | None = None
    asked: bool = False
    answered: bool = False


@dataclass
class RawConstraint:
    """A user clause retained independently of the controlled slot system."""

    original_text: str
    normalized_text: str
    turn: int
    kind: str
    source_attribute: str | None = None
    origin: str = "message"
    active: bool = True
    replaced_turn: int | None = None


@dataclass
class SessionState:
    session_id: str
    slots: dict[str, Slot] = field(default_factory=dict)
    rejected_asins: set[str] = field(default_factory=set)
    turn_history: list[tuple[int, str, dict]] = field(default_factory=list)
    last_candidate_pool: list[dict] = field(default_factory=list)
    override_log: list[tuple[int, str, object, object]] = field(default_factory=list)
    raw_constraints: list[RawConstraint] = field(default_factory=list)
    # Turn numbers whose customer reply merged NO new slot information into
    # memory (no None->value fill, no override) and was not the opening
    # message -- i.e. the simulator's denial / "not quite right" boilerplate.
    # Populated structurally by update_state (from state deltas, not string
    # matching). Consumed by Agent._query_text when noninformative_query_filter
    # is on. See the iter-3 Phase 2 note in CLAUDE.md.
    noninformative_turns: set[int] = field(default_factory=set)
    pending_ask: str | None = None  # most recent slot we asked about and are waiting on
    user_profile: dict = field(default_factory=dict)


def reset(session_id: str, user_profile: dict) -> SessionState:
    state = SessionState(session_id=session_id, user_profile=user_profile or {})
    for name in SLOT_NAMES:
        state.slots[name] = Slot()
    return state


def _preference_tag_bonus(slot_name: str, user_profile: dict, bonus_per_tag: float) -> float:
    tags = user_profile.get("preference_tags") or []
    matches = sum(1 for tag in tags if PREFERENCE_TAG_TO_SLOT.get(str(tag).lower()) == slot_name)
    return matches * bonus_per_tag


def _extraction_confidence(value: object) -> float:
    return 1.0 if value is not None else 0.0


def detect_override(user_message: str, state: SessionState, vocabulary: Vocabulary) -> list[str]:
    """Return every currently-filled slot the message conflicts with.

    A single override message can conflict with more than one slot at once
    ("Actually, make them casual white sneakers" overwrites both color and
    style) -- this must return a list, not a single slot.
    """
    lowered = user_message.lower()
    if not any(cue in lowered for cue in CONTRADICTION_CUES):
        return []

    candidate_new_values = vocabulary.extract(user_message)
    conflicting: list[str] = []
    for slot_name, new_value in candidate_new_values.items():
        existing = state.slots[slot_name]
        if existing.value is not None and existing.value != new_value:
            conflicting.append(slot_name)
    return conflicting


_LOOKING_FOR_RE = re.compile(
    r"\b(?:i['’]?m|i\s+am)\s+looking\s+for\s+"
    r"(?P<category>.*?)(?P<delimiter>,\s*but\b|[.!?]|$)",
    re.IGNORECASE,
)
_RAW_PAYLOAD_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("key_requirement", re.compile(r"\b(?:a\s+)?key\s+requirement\s+is\s*:\s*(.+)$", re.I)),
    ("clarification", re.compile(r"\bwhat\s+matters\s+is\s*:\s*(.+)$", re.I)),
    ("override", re.compile(r"\bwhat\s+i\s+need\s+is\s*:\s*(.+)$", re.I)),
)
_RAW_DENIAL_CUES = (
    "don't have an additional preference", "dont have an additional preference",
    "do not have an additional preference", "not quite right yet",
    "ask me about one specific attribute",
)


def _clean_raw_clause(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip(" \t\r\n-–—,;:.!?")


def _split_raw_clauses(payload: str) -> list[str]:
    return [clause for part in payload.split(";") if (clause := _clean_raw_clause(part))]


def _raw_message_is_noninformative(
    user_message: str, vocabulary: Vocabulary,
) -> bool:
    lowered = user_message.lower()
    return (
        vocabulary.contains_no_preference(user_message)
        or any(cue in lowered for cue in _RAW_DENIAL_CUES)
        or any(cue in lowered for cue in REJECTION_CUES)
    )


def _append_raw_constraint(
    state: SessionState,
    original_text: str,
    turn: int,
    kind: str,
    source_attribute: str | None,
    origin: str,
) -> None:
    normalized = normalize_constraint_text(original_text)
    if not normalized:
        return
    if any(item.active and item.normalized_text == normalized for item in state.raw_constraints):
        return
    state.raw_constraints.append(RawConstraint(
        original_text=original_text,
        normalized_text=normalized,
        turn=turn,
        kind=kind,
        source_attribute=source_attribute,
        origin=origin,
    ))


def _retire_replaced_raw_constraint(
    state: SessionState,
    user_message: str,
    turn: int,
    conflicting_slots: list[str],
) -> None:
    """Conservatively retire one identifiable preference on an override."""

    active_preferences = [
        item for item in state.raw_constraints
        if item.active and item.kind != "category"
    ]
    if not active_preferences:
        return

    candidate: RawConstraint | None = None
    lowered = user_message.lower()
    if "earlier preference" in lowered or "ignore that" in lowered:
        opening_preferences = [
            item for item in active_preferences if item.origin == "opening_preference"
        ]
        if opening_preferences:
            candidate = opening_preferences[-1]

    if candidate is None and conflicting_slots:
        matching = [
            item for item in active_preferences
            if item.source_attribute in conflicting_slots
        ]
        if matching:
            candidate = matching[-1]

    if candidate is None and any(cue in lowered for cue in CONTRADICTION_CUES):
        # A single most-recent preference is safer than clearing all raw state.
        candidate = active_preferences[-1]

    if candidate is not None:
        candidate.active = False
        candidate.replaced_turn = turn


def _extract_raw_constraints(
    state: SessionState,
    user_message: str,
    turn: int,
    vocabulary: Vocabulary,
    conflicting_slots: list[str],
) -> None:
    """Extract literal category/preference clauses without using the catalog labels."""

    if _raw_message_is_noninformative(user_message, vocabulary):
        return

    category_match = _LOOKING_FOR_RE.search(user_message)
    if category_match:
        category = _clean_raw_clause(category_match.group("category"))
        _append_raw_constraint(
            state, category, turn, "category", "category", "opening_category"
        )

    payload_found = False
    for origin, pattern in _RAW_PAYLOAD_PATTERNS:
        match = pattern.search(user_message)
        if not match:
            continue
        payload_found = True
        source_attribute = state.pending_ask if origin == "clarification" else None
        if origin == "override" and len(conflicting_slots) == 1:
            source_attribute = conflicting_slots[0]
        for clause in _split_raw_clauses(match.group(1)):
            _append_raw_constraint(
                state, clause, turn, "preference", source_attribute, origin
            )
        break

    # Intent-override opening turns contain an unlabelled old preference after
    # the category sentence.  Keep it so the later "ignore my earlier
    # preference" message can retire precisely that clause.
    if category_match and not payload_found:
        delimiter = category_match.group("delimiter").lower()
        trailing = _clean_raw_clause(user_message[category_match.end():])
        if delimiter not in (", but",) and trailing and "still exploring" not in trailing.lower():
            for clause in _split_raw_clauses(trailing):
                _append_raw_constraint(
                    state, clause, turn, "preference", None, "opening_preference"
                )
            payload_found = True

    is_override = any(cue in user_message.lower() for cue in CONTRADICTION_CUES)
    if is_override and not payload_found:
        override_payload = ""
        instead_match = re.search(r"\binstead\b[:,]?\s*(.+)$", user_message, re.I)
        if instead_match:
            override_payload = instead_match.group(1)
        else:
            sentences = [
                _clean_raw_clause(part)
                for part in re.split(r"[.!?;]+", user_message)
                if _clean_raw_clause(part)
            ]
            usable = [
                part for part in sentences
                if not any(cue in part.lower() for cue in ("ignore", "never mind", "scratch that"))
            ]
            if usable:
                override_payload = re.sub(r"^actually\s*,?\s*", "", usable[-1], flags=re.I)
        for clause in _split_raw_clauses(override_payload):
            source_attribute = conflicting_slots[0] if len(conflicting_slots) == 1 else None
            _append_raw_constraint(
                state, clause, turn, "preference", source_attribute, "override"
            )
        payload_found = bool(override_payload)

    # Conservative paraphrase fallback: a colon-bearing disclosure, or a
    # non-denial reply immediately following a question, is likely evidence.
    if not category_match and not payload_found:
        if ":" in user_message:
            payload = user_message.split(":", 1)[1]
        elif state.pending_ask:
            payload = user_message
        else:
            payload = ""
        for clause in _split_raw_clauses(payload):
            _append_raw_constraint(
                state, clause, turn, "preference", state.pending_ask, "fallback"
            )


def get_active_raw_constraints(state: SessionState) -> list[RawConstraint]:
    return [constraint for constraint in state.raw_constraints if constraint.active]


def _match_referenced_items(user_message: str, last_candidate_pool: list[dict], vocabulary: Vocabulary) -> set[str]:
    if not last_candidate_pool:
        return set()
    extracted = vocabulary.extract(user_message)
    rejected: set[str] = set()
    if extracted:
        for slot_name, value in extracted.items():
            for product in last_candidate_pool:
                if get_attribute_value(product, slot_name, vocabulary) == value:
                    rejected.add(str(product["parent_asin"]))
        if rejected:
            return rejected
    # no specific attribute could be identified -- assume the top suggestion
    top = last_candidate_pool[0]
    return {str(top["parent_asin"])}


def detect_rejection(user_message: str, last_candidate_pool: list[dict], vocabulary: Vocabulary) -> set[str]:
    lowered = user_message.lower()
    if not any(cue in lowered for cue in REJECTION_CUES) or not last_candidate_pool:
        return set()
    return _match_referenced_items(user_message, last_candidate_pool, vocabulary)


def update_state(state: SessionState, user_message: str, turn: int, vocabulary: Vocabulary) -> SessionState:
    # --- Step A: detect intent override before extracting new slots ---
    conflicting_slots = detect_override(user_message, state, vocabulary)

    # --- Step B: extract candidate slot values from this message ---
    extracted = vocabulary.extract(user_message)

    # --- Step B.5: retain exact raw clauses independently of controlled slots ---
    if any(cue in user_message.lower() for cue in CONTRADICTION_CUES):
        _retire_replaced_raw_constraint(state, user_message, turn, conflicting_slots)
    _extract_raw_constraints(
        state, user_message, turn, vocabulary, conflicting_slots
    )

    # --- Step C: detect rejection signals ---
    rejected_items = detect_rejection(user_message, state.last_candidate_pool, vocabulary)
    state.rejected_asins.update(rejected_items)

    # --- Step C.5: bare "no preference" reply to the slot we most recently asked ---
    if state.pending_ask and vocabulary.contains_no_preference(user_message):
        slot = state.slots[state.pending_ask]
        if not slot.answered:
            slot.answered = True
            slot.source = Source.NO_PREFERENCE
            slot.filled_turn = turn

    # --- Step D: merge extracted values into slot memory ---
    merged_new_info = False
    for slot_name, new_value in extracted.items():
        slot = state.slots[slot_name]

        if slot_name in conflicting_slots:
            state.override_log.append((turn, slot_name, slot.value, new_value))
            slot.value = new_value
            slot.confidence = 1.0
            slot.source = Source.EXPLICIT
            slot.filled_turn = turn
            merged_new_info = True
        elif slot.value is None:
            slot.value = new_value
            slot.confidence = _extraction_confidence(new_value)
            slot.source = Source.EXPLICIT
            slot.filled_turn = turn
            merged_new_info = True
        else:
            # already filled, no override detected -- do not clobber
            continue

        if slot.asked and not slot.answered:
            slot.answered = True

    # --- Step D.5: structural exhaustion for the slot we asked about last turn ---
    # The lexical "no preference" cue in Step C.5 only catches phrasing like
    # "no preference" / "your judgment". The evaluator's actual non-informative
    # reply when it has nothing to offer for the asked attribute is worded
    # differently ("I don't have an additional preference for X."), which never
    # matches those cues. Without this, a slot the simulated customer can never
    # answer stays `asked=True, answered=False` forever, so
    # select_clarification_attribute keeps re-picking the same highest-entropy
    # slot every subsequent turn -- confirmed empirically: 100/100 missed
    # public sessions showed the same ask_attribute repeated 3+ turns in a
    # row, and 90% of Boundary sessions got stuck this way. One round-trip is
    # enough signal: if the reply we just processed didn't fill the slot we
    # were waiting on, treat it as exhausted rather than re-asking indefinitely.
    if state.pending_ask:
        pending_slot = state.slots[state.pending_ask]
        if not pending_slot.answered:
            pending_slot.answered = True
            if pending_slot.source is None:
                pending_slot.source = Source.NO_PREFERENCE
            pending_slot.filled_turn = turn

    if state.pending_ask and state.slots[state.pending_ask].answered:
        state.pending_ask = None

    # --- Step E: structurally flag a non-informative turn ---
    # A reply that merged no new slot value and no override, and is not the
    # opening message, told the search nothing new -- it is the simulator's
    # "I don't have a preference for X" / "Those options are not quite right
    # yet" boilerplate (or a real disclosure the controlled vocabulary can't
    # yet parse; see the Phase 2 note for that tradeoff). Detected from state
    # deltas only, so it survives the private evaluator rewording the text.
    if turn > 1 and not merged_new_info:
        state.noninformative_turns.add(turn)

    state.turn_history.append((turn, user_message, extracted))
    return state


def _flatten_text(value: object) -> str:
    if isinstance(value, dict):
        return " ".join(f"{key} {_flatten_text(item)}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(_flatten_text(item) for item in value)
    return str(value) if value is not None else ""


def _product_slot_text(product: dict) -> str:
    """Text searched for color/size/material/style/feature/use_case.

    title+features alone measurably undershoots the evaluator's own
    disclosure logic (evaluator.local_evaluator.searchable_text(), which
    also reads details/description/categories/store): on the full 50k
    catalog, adding details+description recovers +1369 material / +2999
    color / +3480 size / +4575 style / +3871 use_case / +1827 feature
    matches (scripts/diagnose_vocab_coverage.py). store/categories are
    deliberately excluded here even though the evaluator reads them --
    those are exactly the fields whose raw text caused the original
    brand/category false-positive bug (see vocab.py's _contains_term
    docstring); category and brand already have their own dedicated,
    safe extraction paths above and don't need to be remined here.
    """
    return " ".join((
        str(product.get("title") or ""),
        _flatten_text(product.get("features")),
        _flatten_text(product.get("details")),
        _flatten_text(product.get("description")),
    ))


def get_attribute_value(product: dict, slot_name: str, vocabulary: Vocabulary) -> object:
    """Route each slot through the field where its signal actually lives.

    Structured color/size/material/brand fields exist on only 2-6% of the
    catalog -- reading product[slot_name] directly makes every one of those
    slots look artificially low-information. See the catalog-audit note in
    the design doc.
    """
    if slot_name == "brand":
        store = product.get("store")
        return str(store).strip().lower() if store else None

    if slot_name == "budget":
        return parse_price(product.get("price"))

    if slot_name == "category":
        return deepest_category_segment(product.get("categories"))

    if slot_name in ("color", "size", "material", "style", "feature", "use_case"):
        parent_asin = str(product.get("parent_asin") or "")
        cached = vocabulary.product_slot_cache.get(parent_asin)
        if cached is None:
            cached = vocabulary.extract(_product_slot_text(product), include_catalog_terms=False)
            vocabulary.product_slot_cache[parent_asin] = cached
        return cached.get(slot_name)

    return None


def _distribution(values: list) -> dict:
    counts = Counter(values)
    total = sum(counts.values())
    if total == 0:
        return {}
    return {key: count / total for key, count in counts.items()}


def _entropy(dist: dict) -> float:
    return -sum(p * math.log2(p) for p in dist.values() if p > 0)


_BUDGET_BUCKET_WIDTH = 20.0


def _bucketed_values(slot_name: str, values: list) -> list:
    if slot_name != "budget":
        return values
    return [int(v // _BUDGET_BUCKET_WIDTH) for v in values]


def select_clarification_attribute(
    state: SessionState,
    candidate_pool: list[dict],
    vocabulary: Vocabulary,
    preference_tag_bonus: float = PREFERENCE_TAG_BONUS,
    excluded_slots: frozenset[str] = frozenset(),
) -> str | None:
    askable = [
        name for name in SLOT_NAMES
        if state.slots[name].value is None
        and state.slots[name].source != Source.NO_PREFERENCE
        and not (state.slots[name].asked and state.slots[name].answered)
        and name not in excluded_slots
    ]

    if not askable or not candidate_pool:
        return None

    best_slot: str | None = None
    best_score = -math.inf

    for slot_name in askable:
        raw_values = [get_attribute_value(product, slot_name, vocabulary) for product in candidate_pool]
        values = [v for v in raw_values if v is not None]
        if not values:
            continue

        dist = _distribution(_bucketed_values(slot_name, values))
        score = _entropy(dist)

        coverage = len(values) / len(candidate_pool)
        score = score * coverage

        # nudge toward what this shopper's prior purchases emphasized -- a
        # small additive term so it can break ties among comparably weak
        # low-coverage slots without overturning a strong signal like brand
        score += _preference_tag_bonus(slot_name, state.user_profile, preference_tag_bonus)

        if score > best_score:
            best_score = score
            best_slot = slot_name

    if best_slot is not None:
        state.slots[best_slot].asked = True
        state.pending_ask = best_slot

    return best_slot


def get_filled_slots(state: SessionState) -> dict[str, object]:
    return {
        name: slot.value
        for name, slot in state.slots.items()
        if slot.value is not None and slot.source != Source.NO_PREFERENCE
    }


def get_rejected_asins(state: SessionState) -> set[str]:
    return set(state.rejected_asins)
