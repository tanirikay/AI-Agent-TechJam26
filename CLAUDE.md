# Agent status (for Claude Code sessions)

This file tracks the `agent/` implementation for the TechJam conversational
search challenge — what it does, why it's built this way, and what's still
open. Read this before touching `agent/*` or the tuning scripts; it saves
re-deriving conclusions that were already measured.

The competition's own docs (`README.md`, `docs/competition_specification.md`,
`docs/agent_api_contract.json`) describe the task and scoring. This file is
about the participant solution in `agent/`, not the competition itself.

## What this is

`agent/agent.py::Agent` implements the required `reset`/`respond` interface
by combining:
- `agent/memory.py` — slot state, override/rejection/no-preference tracking,
  and clarification-attribute selection. A fairly literal port of a design
  doc's pseudocode (9-slot enum, entropy×coverage scoring, `NO_PREFERENCE`
  bookkeeping, multi-slot override detection), plus a separate raw-clause
  memory for exact catalog-evidence matching.
- `agent/vocab.py` — the controlled vocabulary shared by BOTH message-side
  extraction (what the shopper says) and product-side extraction (what a
  product's title/features contain). Also holds catalog-derived brand/
  category term lists built at index time.
- `agent/retrieval.py` — BM25 over SQLite FTS5, weighted per field, query
  built from cumulative turn history + filled slots (not just the current
  message, unlike the `starter/agent.py` baseline). It also builds the
  normalized product-evidence cache used by `agent/constraint_rerank.py`.
- `agent/constraint_rerank.py` — offline punctuation-insensitive exact-clause
  scoring over the cached catalog evidence. Stable ties preserve BM25 order.

Run `python -m scripts.run_memory_agent` to score it against the 200 public
sessions (writes `results_memory_agent.json`). `python -m evaluator.local_evaluator`
runs the untouched weak-BM25 starter for comparison.

## Current numbers (public 200, `docs/baseline_results.json` is the starter's reference)

| | HitRate@10 | MRR | MTTC | TechnicalScore |
|---|---|---|---|---|
| Starter (weak BM25, stateless) | 0.125 | 0.068 | 9.81 | 0.107 |
| This agent (before clarification-exhaustion fix) | 0.500 | 0.335 | 6.705 | 0.437 |
| This agent (iter 1, `preference_tag_bonus=0.2`) | 0.775 | 0.4919 | 4.875 | 0.6576 |
| This agent (pre-iter 4, `preference_tag_bonus=1.5`) | 0.795 | 0.5076 | 4.620 | 0.6774 |
| This agent (post-iter 4, exact window 40) | 0.915 | 0.6296 | 3.595 | 0.7945 |
| This agent (iter 5, field-scope widening) | 0.915 | 0.6325 | 3.645 | 0.794348 |
| **This agent (current, iter 6 budget exclusion + ask fallback tier)** | **0.920** | **0.631383** | **3.635** | **0.796715** |

Per scenario (post-iter 4): buying 0.9375 / browsing 0.9375 / intent_override
0.8667 / boundary 0.700 hit rate. Iteration 4 improved every scenario over
the disabled control; see the complete A/B below. Iteration 5 is a clean,
zero-hit-to-miss MRR improvement at essentially the same TechnicalScore
(the small MTTC give-back roughly offsets the MRR gain) — see below for why
it's adopted anyway.

Reproduce with `python -m scripts.run_memory_agent`. Numbers are
deterministic run-to-run (see the determinism fix below); verified by two
byte-identical consecutive runs.

**Careful — `python -m evaluator.local_evaluator` does NOT score this
agent.** It imports `starter.agent.Agent` and reproduces the weak starter
(0.125 / 0.068 / 9.81 / 0.107). Only `python -m scripts.run_memory_agent`
scores `agent/`.

## Design decisions and why

- **Query = full turn history + filled slots, not just the latest message.**
  This is the single biggest lever over the baseline — the baseline
  re-searches from scratch every turn and forgets turn 1 by turn 3.
- **Raw constraints remain separate from the nine controlled slots.** Literal
  category, requirement, clarification, override, and conservative fallback
  clauses are normalized and retained even when `agent/vocab.py` cannot
  represent them. Active clauses exact-match a once-built cache of title,
  categories, features, detail keys/values, store, and description. The
  adopted top-40 pass sorts by match count and uses the incoming BM25/budget
  order as the stable tie-breaker.
- **`get_attribute_value` routes each slot to where its real signal lives**
  (catalog audit, still valid): `brand`→`store` field (~99% coverage),
  `budget`→numeric `price` (~21% coverage, true nulls stay null, never
  coerced to 0), `category`→deepest `categories[]` segment (not `[0]`,
  which is ~99.98% identical catalog-wide), everything else → shared
  keyword vocabulary applied to `title + features`.
- **Field weights are tuned, not the starter's untouched defaults.**
  Current: `DEFAULT_FIELD_WEIGHTS = (5.0, 6.0, 2.5, 2.0, 3.0, 1.0)` for
  (title, categories, features, details, store, description). Chosen by
  `scripts/tune_field_weights.py`; validated with a split-half check
  (`scripts/validate_weight_split.py`) so it's not just full-set sweep noise.
- **`user_profile.preference_tags` biases which slot gets asked first**
  (additive bonus in `select_clarification_attribute`, re-tuned to **1.5**
  in iter 2 — see the "iter 2" Task 1 note below; was 0.2), never injected
  into the search query or used to filter/exclude products. Kept
  deliberately "safe": aggregate signal only, no raw purchase history.
- **Budget affects ranking via price-tier bucketing**
  (`Agent._apply_budget_bucketing`): confirmed-over-budget items sink,
  unknown-price items are never penalized. Correctly implemented — see
  "Known dead ends" below for why it barely ever fires on this dataset.
- **`brand`/`category` are excluded from what the agent asks about**
  (`UNASKABLE_SLOTS` in `agent/agent.py`), even though they're still used
  for retrieval. This is not a guess — `evaluator/local_evaluator.py`'s
  `classify_constraint()` has no code path that ever returns `"brand"` or
  `"category"`, so the simulated customer can structurally never answer
  either. Asking anyway is a guaranteed dead turn. Since the private
  evaluator "follows the same template," this should hold there too.

## Confirmed bugs fixed this pass (don't reintroduce)

1. **Nondeterminism**: brand/category longest-match iterated a raw `set`,
   whose order depends on Python's per-process hash randomization. Two
   identical runs scored 0.490 vs 0.495 HR before the fix. Fixed by
   precomputing a sorted, deterministic match order in `Vocabulary.__post_init__`.
2. **Brand/category false-positive extraction (the big one)**: naive
   substring matching against ~19,749 raw catalog store names matched
   `"king"` inside `"looking"`, `"door"` inside `"outdoor"`, `"totes"`
   inside `"Totes"` — **100% false-positive rate** on a 15-session sample,
   silently filling `brand` with garbage on turn 1 of nearly every session
   for the entire session before this was caught. Fixed with word-boundary
   matching (`_contains_term` in `vocab.py`) plus a minimum length of 6 for
   single-word terms (multi-word phrases exempt). If you touch
   `_brand_sorted`/`_category_sorted` matching logic, re-verify against a
   message like `"I'm looking for..."` (should NOT fill brand).

   Non-obvious trap: fixing this bug alone *dropped* the score
   (0.49→0.255) because it un-poisoned `brand`, which then started winning
   the clarification race — straight into the unasked-slot problem above.
   Both fixes are needed together; shipping one without the other regresses.

3. **Clarification stuck-loop (the biggest single win so far: +0.221
   TechnicalScore).** `Vocabulary.contains_no_preference` only recognises
   phrasings like "no preference" / "your judgment". But
   `local_evaluator.customer_reply()` line ~183 returns
   `"I don't have an additional preference for {attribute}."` whenever it
   has nothing to disclose for the asked slot — which matches none of those
   cues. The slot therefore stayed `asked=True, answered=False` forever,
   nothing else about the state changed, and
   `select_clarification_attribute` re-picked the same highest-entropy slot
   every remaining turn. Measured before the fix: the same `ask_attribute`
   repeated 3+ consecutive turns in 120/200 sessions, including **100/100
   of the missed sessions**; boundary was 90% stuck (HR 0.10).

   Fixed in `memory.py::update_state` **structurally, not lexically**
   (Step D.5): if the slot recorded in `state.pending_ask` still isn't
   filled after this turn's reply has been processed, mark it
   `answered=True` / `Source.NO_PREFERENCE`, which forces the next turn to
   pick a different slot. Deliberately keyed on "did this slot get filled",
   not on wording, so it does not depend on the exact phrasing the private
   800-session evaluator uses. `NO_PREFERENCE_CUES` in `vocab.py` was left
   untouched on purpose — broadening the cue list is the lossier, more
   brittle version of the same fix.

   After: repeated-ask streaks are gone entirely — max streak is 1 for all
   200 sessions (`scripts/diagnose_pool_size.py`).

## Iteration 2 (post clarification-loop fix) — 2026-08-30

Baseline going in: 0.775 / 0.4919 / 4.875 / 0.6576, verified byte-identical
across two runs and across `PYTHONHASHSEED` values (the agent is
seed-deterministic — the earlier nondeterminism bug is genuinely gone).

### Task 1 — re-tune field weights + policy constants

**Field weights: re-swept, NO CHANGE.** Re-ran `tune_field_weights.py` plus a
13-candidate finer search around its nominal winner
(`scripts/tune_field_weights2.py`). **HR@10 was frozen at 0.7750 for every
single candidate** (all four per-scenario HRs unmoved too); only MRR wiggled
±0.01 and MTTC ±0.09. Nominal best "boost store + title, cut description"
`(7,4,2.5,2,4,0.5)` scored tech 0.6600 vs incumbent 0.6576 — a pure MRR
reshuffle within already-hit sessions. Split-half: that winner *loses* to the
incumbent on half A (0.6302 vs 0.6310) while winning half B — fails the
both-halves test. Identical signature to the "0.00195 win that was noise"
already documented below. **Kept `DEFAULT_FIELD_WEIGHTS = (5,6,2.5,2,3,1)`.**

**`pool_threshold` / `budget_tolerance`: still inert.** Every swept value gave
byte-identical 0.657570. Confirmed structurally dead (OR-query pool pinned at
200; budget undisclosable), exactly as the "Known dead ends" section says.

**`preference_tag_bonus`: 0.2 → 1.5 — ADOPTED (+0.0198 technical_score).**
This was tuned to 0.2 under the *old stuck-loop* dynamics, where most turns
after the first carried no new info. Now that every turn fills a fresh slot,
asking the profile-emphasised attributes first surfaces the discriminative
constraint sooner. Swept 0.0→3.0 (`scripts/tune_pref_bonus.py`):
technical_score rises **monotonically** and saturates at ~1.5 (2.0 and 3.0
are byte-identical to 1.5 on HR/MRR). At bonus 1.5 a slot matching two
profile tags gets +3.0, which dominates entropy×coverage (~1–2.5) — i.e.
"ask what the profile cares about first, fall back to entropy for the rest."

| bonus | HR@10 | MRR | MTTC | tech |
|---|---|---|---|---|
| 0.2 (old) | 0.7750 | 0.4919 | 4.875 | 0.65757 |
| 1.0 | 0.7800 | 0.5052 | 4.745 | 0.66665 |
| **1.5 (new)** | **0.7950** | **0.5076** | **4.620** | **0.67738** |
| 2.0 | 0.7950 | 0.5076 | 4.615 | 0.67748 |

Split-half validated (`scripts/validate_policy_split.py`) — monotone in BOTH
technical_score AND HR@10 on **each** half independently:
half A 0.6310→0.6535 (HR 0.750→0.770), half B 0.6841→0.7013 (HR 0.800→0.820).
Gains land in browsing (0.775→0.800) and buying (0.7875→0.8125); boundary and
intent_override HR unchanged. Changed `memory.PREFERENCE_TAG_BONUS = 1.5` and
re-centred the `tune_policy_constants.py` candidate grid.

### Task 2 — strip boilerplate non-answers from the BM25 query: TESTED AND REJECTED

`evaluator.local_evaluator.customer_reply()` emits three zero-product-signal
reply shapes: `"Those options are not quite right yet. Ask me about one
specific attribute."`, `"I don't have an additional preference for {attr}."`,
`"I don't have a preference for {attr}; please use your judgment."`. Added a
regex (`agent.agent._NONINFORMATIVE_REPLY_RE`) and an opt-in
`Agent(strip_boilerplate_query=…)` flag (default **False**) that drops those
turns' text from `_query_text` while still using the turn for slot
bookkeeping.

A/B on top of `preference_tag_bonus=1.5` (`scripts/ab_strip_boilerplate.py`):

| | full tech / HR | half A tech | half B tech |
|---|---|---|---|
| off (kept) | 0.67738 / 0.795 | 0.6535 | 0.7013 |
| on | 0.66738 / 0.780 | 0.6521 | 0.6826 |

Loses on the full set (−0.010, −2 HR sessions) **and both halves.** Same
lesson as the pool-narrowing work below: against an OR-of-terms BM25 query,
removing terms only hurts — the boilerplate still carries the *asked slot
name* ("material", "color", …) as a weak positive term that nudges genuinely
relevant products, and dropping borderline terms drops candidates out of the
200-pool. Flag left in place, disabled, for the record.

### Task 3 — remaining Boundary misses (n=10, 4 misses): traced, no fix warranted

`scripts/trace_boundary_misses.py` dumps every boundary session turn-by-turn
(message, ask_attribute, pool size, pool-rank of target, filled slots). The 4
misses (all difficulty=medium): public_0035, 0169, 0180, 0187.

- **0035** (Skechers Air Mesh walking shoe; hard = `fabric` / `100% Textile`):
  target never enters the 200-pool. "Athletic Walking" + "fabric" are too
  generic; "fabric" isn't in `MATERIAL_VOCAB` so it never becomes a slot,
  "100% Textile" is never extracted. **BM25 recall failure.**
- **0169** (Amazon Essentials *knit cotton* jegging, in the "Jeans" category;
  hard = `cotton`): "Women Jeans" fills `material=denim` (`MATERIAL_VOCAB`
  denim synonym "jeans"), which is wrong for a knit-cotton jegging **and**
  makes `material` non-askable, so "cotton" is never disclosed. Target never
  in pool. Tested dropping "jeans" from the denim synonyms
  (`scripts/ab_extraction_fixes.py` fix A): **byte-identical** to control on
  full + both halves — the target still isn't retrievable (recall problem
  underneath the mis-inference). Not adopted (a zero-effect change is pure
  private-set risk).
- **0180** (Saucony Cohesion *running shoe*, in "Fashion Sneakers"; hard =
  `100% Mesh`): `material=mesh` disclosed by turn 5, still never in pool.
  "fashion sneakers" + "mesh" + "imported" matches too many near-identical
  sneakers. **BM25 recall failure.**
- **0187** (Eastland Newport slip-on; hard = `leather`): category +
  `material=leather` + size all filled correctly, target sits at **pool rank
  ~22–30 every turn**, never reaches top 10. The one **genuine ranking
  miss** — reranker territory, but see Task 4.

Also surfaced (not a boundary-specific issue): **`size="m"` is filled in ~every
session** because `TOKEN_RE` splits "I'm" → `["i","m"]` and "m" matches
`SIZE_VOCAB`. It's filtered out of the BM25 query (1-char), so near-harmless,
but it suppresses `size` clarifications. Tested blocking 1-char size tokens
(`ab_extraction_fixes.py` fix B): **loses** (0.795→0.780, both halves down) —
the spurious fill is *net-positive* because `size` is rarely a productive ask
and skipping it saves a turn. Reject.

Verdict: clarification logic is correct in all 4 misses (right slots asked,
constraints disclosed on cue). 3 of 4 are BM25 recall/ranking limits on
near-identical shoe products; at n=10 each miss is 0.10 HR of noise. Not
worth a targeted fix.

### Task 4 — embedding-similarity rerank: TESTED AND REJECTED (badly)

**Feasibility:** `sentence-transformers` + `all-MiniLM-L6-v2` (CPU, ~90 MB,
384-dim) installs from PyPI and runs fully offline once the model is cached —
within "free/zero-cost, no GPU, no API keys". Product embeddings (title +
features, capped at 500 chars) build in ~9 min for the 50k catalog and cache
to a 76 MB `.npy` (`agent/embed_rerank.py`, `$TECHJAM_EMBED_CACHE`). Per-turn
query encode is a few ms. So it *is* buildable — but:

Wired as an opt-in hook exactly like `slot_match_rerank`:
`Agent(embed_rerank_window=N)` reranks the top N of the BM25 pool by cosine
similarity of the accumulated query text to each product; `None` (default)
disables it and never imports torch. A/B (`scripts/ab_embed_rerank.py`), on
top of `preference_tag_bonus=1.5`:

| window | HR@10 | MRR | MTTC | tech | half A / half B tech |
|---|---|---|---|---|---|
| control (off) | **0.795** | 0.508 | 4.62 | **0.6774** | 0.653 / 0.701 |
| top-20 | 0.735 | 0.339 | 5.27 | 0.5840 | 0.549 / 0.620 |
| top-40 | 0.645 | 0.273 | 6.01 | 0.5042 | 0.497 / 0.511 |
| top-100 | 0.550 | 0.211 | 6.91 | 0.4203 | 0.415 / 0.425 |
| full-200 | 0.475 | 0.196 | 7.48 | 0.3667 | 0.361 / 0.372 |

**Every window hurts, monotonically worse as the window widens** — even the
smallest (top-20) drops HR@10 0.795→0.735. Root cause: the query text is an
accumulated bag of conversational fragments ("I'm looking for Athletic
Walking, but I'm still exploring. … For that, what matters is: fabric.
athletic athletic"). MiniLM embeds that whole blob into one vector, and
general-purpose sentence similarity to it has **no notion of "this exact
category/material constraint must hold"** — it promotes products that are
"vibe-similar" to the whole conversation over ones that lexically match the
discriminative terms. BM25's precise term matching on the deepest category
segment + material is exactly the signal that wins here, and the embedding
rerank washes it out. Same failure mode as the rejected keyword rerank, but
much worse.

The recall-gap diagnostic's premise still holds (target in pool ~89%, top-10
~50%), but **MiniLM sentence similarity is not the signal that closes it** —
it is strictly worse than BM25's own ordering. A constraint-aware reranker
(cross-encoder trained on "does this product satisfy this attribute", or an
LLM judge) might, but that is out of scope for the free/offline constraint
and is not "a small MiniLM model". Hook + `agent/embed_rerank.py` left in the
tree, disabled, so the negative result is reproducible; the default agent
does not import torch.

## Tested and rejected — don't redo without a new angle

**Slot-match reranking** (re-sort the BM25 pool by how many confirmed slots
each candidate matches): tried full pool, windowed (top 20/40), all-slots,
structured-only (brand+category). Every configuration matched or lost to
plain BM25+budget order on `technical_score`; several clearly hurt MRR (a
decoy ranked #150 by BM25 but coincidentally matching a slot would jump
ahead of the true target ranked #2 with no recorded match). Code kept as an
opt-in hook (`Agent(slot_match_fields=..., slot_match_window=...)`,
disabled by default via `frozenset()`) in case a better match *signal* (real
embedding similarity instead of binary keyword match) revives the idea —
the underlying opportunity is real (see next section), this specific
heuristic just isn't it.

## Known dead ends / structural findings (root cause understood, not fixed)

- **`pool_threshold` does nothing, and that is the right behaviour here —
  investigated and closed.** The BM25 query is built with `OR` between
  terms, so the candidate pool can only grow or stay the same as more terms
  accumulate. Re-measured *after* the clarification-exhaustion fix (in case
  faster slot-filling changed the dynamics): pool size is still exactly 200
  on **930/930 turns**, min = max = mean = 200. So `pool_threshold` is still
  dead code.

  Making it live was then implemented and A/B tested
  (`scripts/tune_pool_narrowing.py`) — two narrowing passes layered on top
  of, never replacing, the OR recall pass, both with fallback to the full
  pool so recall can't regress: `narrow_by_score` (keep candidates within a
  ratio of the best bm25 score — tail truncation only) and
  `narrow_by_conjunction` (keep only pool members containing ALL confirmed
  slot values — an AND filter that *can* change the top 10). Both are
  wired into `Agent` behind `narrow_mode` / `narrow_param` /
  `narrow_recommendations`, **disabled by default**; the no-narrowing
  control reproduces 0.657570 exactly, so the code paths are inert when off.

  | config | HR@10 | MRR | MTTC | TechScore | turns ≤ threshold |
  |---|---|---|---|---|---|
  | control (off) | 0.7750 | 0.4919 | 4.875 | **0.65757** | 0 |
  | score cutoff 0.50 | 0.7750 | 0.4984 | 4.875 | 0.65952 | 44 |
  | score cutoff 0.30 | 0.7750 | 0.4919 | 4.875 | 0.65757 | 10 |
  | conjunction (decision only) | 0.7650 | 0.4871 | 4.990 | 0.64884 | 60 |
  | score cutoff 0.70 | 0.7500 | 0.4879 | 5.095 | 0.63948 | 156 |
  | conjunction (+recs) | 0.7400 | 0.4494 | 5.210 | 0.62062 | 111 |
  | score cutoff 0.85 | 0.3850 | 0.2151 | 7.745 | 0.32212 | 1252 |

  The nominal best (score cutoff 0.50, +0.00195) is noise, not a mechanism:
  only 8/200 sessions change, all of them hits both ways, ranks moving in
  both directions; HR@10 and MTTC are bit-identical to the control. The
  monotonic collapse as the cutoff tightens is the real signal.

  **Root cause of why this can't pay off:** when `ask_attribute` is `None`,
  `customer_reply()` returns `"Those options are not quite right yet. Ask me
  about one specific attribute."` — zero new information, and the boilerplate
  still lands in the BM25 query as noise. Asking is the *only* channel
  through which the agent gains information, and `recommendations` are the
  head of the BM25 ordering regardless of the ask/answer branch, so
  "stop asking once confident" is strictly a loss against this harness.
  A score cutoff also can't help ranking by construction: it truncates the
  tail without reordering, so it never touches the top 10.

  Conclusion: leave `narrow_mode=None`. `pool_threshold` should be
  understood as intentionally inert against this evaluator, not as a bug
  waiting to be fixed. Don't re-attempt without a *new* mechanism — one that
  changes the top-10 ordering (e.g. embedding rerank), not the ask/answer
  branch.
- **Budget is structurally undisclosable by this harness, not just rare.**
  Measured: 178/200 session targets have a real price, but budget appears
  in the disclosable `hard_constraints`/`soft_preferences` in **0/200**
  sessions. Root cause is in `evaluator/local_evaluator.py::intent_card()`
  — the budget candidate string is *appended last* to a list that gets
  sliced to `[:2]`/`[2:4]`, so it's excluded by construction for any
  product with a normal number of features. This is a property of the
  evaluator (not ours to edit per the submission rules), so the budget
  bucketing fix above is correct but has near-zero measured ROI here.
- **Semantic reranking has real, measured upside**: target is in the
  ~200-candidate pool 89% of the time vs. only 50% in the scored top 10 —
  78% of all misses are sitting in the pool, just ranked wrong
  (`scripts/diagnose_recall_gap.py`). The keyword-based rerank attempt
  failed (see above); a real embedding-similarity rerank is untried — no
  model/API was verified available in this environment.
- **Boundary scenario** (n=10, so noisy) was the weakest at 0.10 HR; the
  clarification-exhaustion fix took it to 0.60, which was its dominant
  failure mode (90% of boundary sessions were stuck re-asking). It is still
  the weakest scenario and n=10 makes it noisy.
- **Field-weight tuning validated by split-half only**, not full
  cross-validation. Reasonable confidence, not proof, that it generalizes
  to the private 800-session set.

## Iteration 3 — Phase 1 ceiling diagnostic (2026-08-30, behaviour unchanged)

`scripts/diagnose_ceiling.py` replays all 200 public sessions (exact evaluator
protocol) and, per session, reads `agent._sessions[sid].last_candidate_pool`
— the full post-rerank pool of up to 200 products, of which the scored
recommendations are just the first 10 — to classify every miss as *retrieval*
(gold never in the pool) or *ranking* (gold in the pool, ranked >10). It
recomputes current HR@10/MRR from the replay and they match the canonical
scorer to 4 dp at both bonus values, so the replay is faithful. Byte-identical
across two runs / `PYTHONHASHSEED` values.

|  | current HR@10 | current MRR | pool recall@200 = **oracle HR@10** | misses: retrieval / ranking |
|---|---|---|---|---|
| `preference_tag_bonus = 0.2` | 0.7750 | 0.4919 | **0.9700** (194/200) | 45 = 6 / 39 |
| `preference_tag_bonus = 1.5` (current) | 0.7950 | 0.5076 | **0.9700** (194/200) | 41 = 6 / 35 |

**The remaining gap is almost entirely a RANKING problem.**
- Pool recall@200 is 97% at *both* bonus values — only **6 sessions** have
  gold that is never retrievable, and it is the *same 6* at both values
  (public_0020, 0035, 0043, 0169, 0175, 0180). Half of them are jeans
  (Levi's / Ariat / Amazon jegging) where the deepest category segment is the
  generic "Jeans" and the disclosed constraints are weak; the rest are the
  novelty-tee / walking-shoe / running-shoe tail already seen in the iter-2
  boundary trace. Not reranker-addressable.
- **85% of current misses (35/41) have gold in the pool, ranked >10.** Of
  those 35: best_rank ≤20 for 12, ≤30 for 19, and 16 are deeper than 30
  (worst: an intent_override at rank 173). 19/35 were already in the pool on
  turn 1 — "never surfaced", not "session ended too early".
- Even among the **159 hits, only 83 (52%) are at rank 1**; 25 at rank 2-3,
  51 at rank 4-10.
- The `preference_tag_bonus` 0.2→1.5 re-tune was a *clean* ranking win: it
  converted 4 ranking-misses to hits (public_0098, 0155, 0178, 0195) and
  broke **zero** — all 41 remaining misses are also misses at bonus 0.2.

**Headroom, against the marginal-value figures (+0.01 HR = +0.005 TS,
+0.01 MRR = +0.003 TS, −1.0 MTTC = +0.02 TS):**
- Tightening the **76 hits currently at rank 2-10 toward rank 1**: MRR
  0.508 → 0.795 ceiling = **+0.086 TS**, plus an MTTC drop. This is the
  single largest lever and the *safest* — these sessions already converge.
- Recovering the **12 ranking-misses at best_rank 11-20**: HR 0.795 → ~0.855
  = **+0.030 TS**. Another ~7 sit at 21-30.
- Perfect-reranker ceiling on the current ask path: HR 0.97 / MRR 0.97
  (TS ≈ 0.88). Fantasy, but it bounds the opportunity.

**Recommended next lever (not yet implemented — Phase 2 was deferred pending
this):** the two rerankers already tried (keyword slot-match, MiniLM
embedding) both failed because neither adds a *constraint-satisfaction*
signal. Before reaching for a cross-encoder / LLM judge (out of the current
free/offline/no-torch-default scope), try the cheap query-construction lever
that stays inside the OR-pool paradigm the harness rewards:
**up-weight the confirmed slot values in `_query_text` by repeating each one
K times** (BM25 term-frequency), so a document that satisfies every confirmed
slot outranks a lexical decoy that only shares chat words. ~5-line change +
a K sweep; low risk (no pool shrink, no reordering pass); plausibly moves
both the rank-2-10 hits and the rank-11-30 near-misses. If a K sweep does
nothing, that is strong evidence the remaining ranking signal genuinely
requires a learned constraint-aware model and the scope question becomes
unavoidable.

Per-session rows: `runs/ceiling_diagnostic.json` (keys `"0.2"` / `"1.5"`).

## Iteration 3 — Phase 2: structural query-hygiene filter — TESTED AND REJECTED (badly)

Hypothesis: when the simulator emits `"I don't have an additional preference
for material."` the word `material` lands in the BM25 query as noise pulling
in decoys. Iter-2's lexical fix (`strip_boilerplate_query`, regex-match the
three known templates) already lost −0.010 and failed both split halves; the
brief asked for a **structural** version that survives the private evaluator
rewording the boilerplate.

Implemented (`Agent(noninformative_query_filter=True)`, default off,
`memory.SessionState.noninformative_turns`): `update_state` now records, from
**state deltas only** (no string matching), every turn whose reply merged
**zero new slot information** — no `None→value` fill, no override — and isn't
the opening message. `_query_text` drops those turns' text.

A/B on top of `preference_tag_bonus=1.5`
(`scripts/ab_noninformative_filter.py`, full protocol):

| config | HR@10 | MRR | MTTC | TS | half A / B TS |
|---|---|---|---|---|---|
| control | 0.795 | 0.508 | 4.62 | **0.6774** | 0.6535 / 0.7013 |
| structural (this) | **0.585** | 0.430 | 6.19 | **0.5179** | 0.512 / 0.524 |
| lexical (iter-2, ref) | 0.780 | 0.504 | 4.69 | 0.6674 | 0.652 / 0.683 |

Paired per-session diff vs control: **42 sessions flipped hit→miss, 0
miss→hit.** Bootstrap 95% CI on the TS delta: **[−0.203, −0.115]** (mean
−0.159). Not noise — a large, unambiguous regression on the full set and both
halves.

**Root cause (instrumented):** the filter drops **815 turns across the 200
sessions, and 139 of them (17%) are the *informative* `"For that, what
matters is: …"` branch** — real disclosures the 9-slot controlled vocabulary
simply can't parse ("Imported", "Pull On closure", "Rubber sole", "Leather
sole", "GRID, Injection EVA", "3X women tops care: Hand wash…"). Those
specific phrases are the single highest-value lexical signal in the
conversation, and "no slot merged this turn" cannot tell them apart from a
genuine denial. So the structural signal is *strictly worse* than the lexical
one: it removes everything the lexical version removes (which already lost),
plus 17% genuine disclosures.

**Both versions of Phase 2 are now closed.** The premise is wrong for this
harness: against an OR-of-terms BM25 query (a) the denied slot-name is a weak
*positive* term, not noise; (b) removing any term can drop the gold out of
the 200-candidate pool; (c) the simulator's disclosures routinely fall
outside the controlled vocabulary, so "no slot merged" ≠ "no information".
The `noninformative_query_filter` flag + `noninformative_turns` tracking are
left in the tree, disabled, so the negative result reproduces; the default
agent (0.677383) is byte-identical and unaffected.

## Iteration 3 — step 1: up-weight confirmed slot values in the BM25 query — TESTED AND REJECTED

The Phase-1 diagnostic's recommended lever, and the last untried
query-construction idea. Mechanism: `retrieval._match_expression` now appends
each confirmed slot value's terms `slot_boost_k` EXTRA times to the FTS5
`MATCH` OR-expression (`Agent(slot_boost_k=K)`, default 0 = inert; slot values
are already in the query once via `_query_text`, so effective repetition is
K+1). Verified first that SQLite's `bm25()` actually rewards a repeated
query-phrase (it sums IDF-weighted contributions per occurrence). Recall is
untouched — every original term is still present, the pool is still 200.

Sweep on top of `preference_tag_bonus=1.5` (`scripts/tune_slot_boost.py`,
full protocol):

| K (effective ×) | HR@10 | MRR | MTTC | TS | ΔTS vs K=0 | bootstrap 95% CI |
|---|---|---|---|---|---|---|
| 0 (1×) | **0.795** | 0.508 | 4.62 | **0.67738** | — | — |
| 1 (2×) | 0.795 | 0.506 | 4.65 | 0.67640 | −0.001 | [−0.021, +0.018] |
| 2 (3×) | 0.780 | 0.484 | 4.81 | 0.65886 | −0.019 | [−0.045, +0.007] |
| 3 (4×) | 0.755 | 0.466 | 4.95 | 0.63822 | −0.039 | [−0.068, −0.009] |
| 5 (6×) | 0.730 | 0.446 | 5.25 | 0.61381 | −0.064 | [−0.095, −0.033] |
| 8 (9×) | 0.690 | 0.420 | 5.61 | 0.57865 | −0.099 | [−0.133, −0.064] |

- **K=1 is a wash** (ΔTS −0.001, CI straddles 0; 66 sessions move but evenly
  — 3 hit / 3 miss, 31 rank-better / 29 rank-worse). Pure churn.
- **K≥2 is a monotone regression** on HR@10, MRR *and* MTTC, and loses on
  both split halves; K≥3 bootstrap CIs exclude 0. At K=8, 23 sessions flip
  hit→miss vs 2 the other way.

**Root cause:** the terms we are *certain* about are exactly the *least
discriminative* ones. Confirmed slots are broad category/attribute words —
`style=athletic`, `category=athletic`, `size=m` (the spurious "I'm"→"m" fill),
`color=black`, `material=leather` — each shared by hundreds of products.
Repeating "athletic" ×9 makes every athletic-category product score high and
**drowns the true discriminative signal**, which lives in the chat text and
the unvocabbed disclosures (brand names, distinctive title tokens like
"Cohesion" / "Newport" / "Adilette"). BM25 term-frequency boosting rewards
*breadth of match on the boosted terms*, which is the opposite of what
separates the target from its category-mates.

**This closes the "better-prompt-BM25" avenue.** Every query-construction
lever is now explored: removing denied-slot terms (rejected ×2: lexical
−0.010, structural −0.159), adding weight to confirmed-slot terms (rejected,
monotone negative), and the raw chat text is already fully included. The
current `_query_text` is at its ceiling. `slot_boost_k` left in the tree,
default 0 (inert; default agent byte-identical at 0.677383).

The Phase-1 ranking headroom (76 hits at rank 2-10, ~19 near-misses at
11-30) is real but genuinely needs a signal BM25's bag-of-words cannot
express. With no LLM and no torch-by-default, **retrieval tuning is done** —
see the next-session list.

## Iteration 4 — raw-constraint exact-match reranker — ADOPTED (2026-08-30)

Iteration 3's conclusion above was too broad: the controlled slots were
exhausted, but the evaluator routinely discloses literal feature/detail
clauses that the controlled vocabulary cannot represent. Iteration 4 keeps
those clauses as a separate deterministic memory and checks whether each is
an exact contiguous substring of normalized catalog evidence. This adds the
missing constraint-satisfaction signal without an LLM, embeddings, a neural
model, an external service, a network call, or a new dependency.

### Mechanism

- `memory.RawConstraint` records original/normalized text, turn, kind,
  source attribute when known, origin, active state, and replacement turn.
  Category constraints are preserved through overrides; an identifiable old
  preference is retired conservatively rather than clearing all raw state.
- Extraction handles the opening `looking for` category, key requirements,
  semicolon-separated clarification disclosures, intent overrides, a
  conservative colon fallback, and non-denial replies following
  `pending_ask`. Denials, rejection boilerplate, and no-preference replies do
  not enter raw memory.
- `build_catalog_index()` creates normalized evidence once for every product,
  including title, categories, features, detail dictionary keys and values,
  store, and description.
- For each candidate, `evidence_score` is the number of distinct active raw
  clauses present verbatim after shared punctuation/whitespace normalization.
  The configured prefix is stably sorted by descending score; ties retain
  BM25 plus budget-tier order. The BM25 pool remains 200 and is never shrunk.
- `constraint_rerank_window=None` is inert and reproduces the pre-iteration
  control. The adopted default is 40. Approximate/fuzzy matching was not
  implemented or enabled.

### Full A/B (`scripts/ab_constraint_rerank.py`)

The disabled control exactly reproduced 0.795 / 0.507611 / 4.620 / 0.677383.
The previously suggested ~0.95 result was independently reproduced by the
full-200 experiment (HR@10 0.945).

| config | HitRate@10 | MRR | MTTC | TechnicalScore | delta TS |
|---|---:|---:|---:|---:|---:|
| control (`None`) | 0.795 | 0.507611 | 4.620 | 0.677383 | — |
| exact 20 | 0.850 | **0.645861** | 4.095 | 0.756858 | +0.079475 |
| **exact 40 (adopted)** | **0.915** | **0.629577** | **3.595** | **0.794473** | **+0.117090** |
| exact 100 | 0.935 | 0.610819 | 3.370 | 0.803346 | +0.125963 |
| exact 200 | 0.945 | 0.598208 | 3.245 | 0.807062 | +0.129679 |

Scenario HitRate@10 (all four improve over control; no unexplained scenario
regression):

| config | boundary | browsing | buying | intent_override |
|---|---:|---:|---:|---:|
| control | 0.600 | 0.8000 | 0.8125 | 0.8000 |
| exact 20 | 0.700 | 0.8500 | 0.8750 | 0.8333 |
| exact 40 | 0.700 | 0.9375 | 0.9375 | 0.8667 |
| exact 100 | 0.700 | 0.9500 | 0.9625 | 0.9000 |
| exact 200 | 0.700 | 0.9500 | 0.9875 | 0.9000 |

Alternating split-half results:

| config | half A HR / MRR / MTTC / TS | half B HR / MRR / MTTC / TS |
|---|---|---|
| control | 0.770 / 0.478230 / 4.750 / 0.653469 | 0.820 / 0.536992 / 4.490 / 0.701298 |
| exact 20 | 0.840 / 0.625968 / 4.070 / 0.746390 | 0.860 / 0.665754 / 4.120 / 0.767326 |
| **exact 40** | **0.920 / 0.629357 / 3.540 / 0.798007** | **0.910 / 0.629798 / 3.650 / 0.790939** |
| exact 100 | 0.940 / 0.623563 / 3.310 / 0.810869 | 0.930 / 0.598075 / 3.430 / 0.795822 |
| exact 200 | 0.940 / 0.616468 / 3.280 / 0.809340 | 0.950 / 0.579948 / 3.210 / 0.804784 |

Paired results versus control and 2,000-resample bootstrap intervals:

| config | gained / lost sessions | miss→hit | hit→miss | TS delta 95% CI |
|---|---:|---:|---:|---|
| exact 20 | 70 / 9 | 11 | **0** | [+0.053642, +0.107227] |
| **exact 40** | **82 / 16** | **24** | **0** | **[+0.083570, +0.154006]** |
| exact 100 | 86 / 20 | 28 | **0** | [+0.091319, +0.163769] |
| exact 200 | 86 / 24 | 30 | **0** | [+0.093360, +0.169332] |

The detailed gained/lost session IDs, all session rows, per-scenario metrics,
and bootstrap metadata are in `runs/constraint_rerank_ab.json`.

### Window choice, determinism, leakage, and latency

Full 200 had the highest point estimate, but its paired TS advantage over
window 100 was +0.003716 (95% CI [−0.005402, +0.016117]) and over window 40
was +0.012589 (95% CI [−0.003430, +0.032860]). Window 20 was significantly
worse than 200 (difference +0.050204, CI [+0.021510, +0.082802]). Per the
brief's tie-break rule, **40 is the simplest window statistically
indistinguishable from the point-estimate winner, so 40 is adopted.**

Determinism passed at the full JSON level: two consecutive
`PYTHONHASHSEED=1` runs, a `PYTHONHASHSEED=999` run, and the regenerated
`results_memory_agent.json` were byte-identical (SHA-256
`3f530500bbd14a0f4f1bd76b508432ee5a38007be21d5877cd1e435a5712eb44`).
All 14 tests pass, including regression coverage for `I'm`→size `m`,
clarification exhaustion, and brand boundaries.

End-to-end evaluator timing with a shared prebuilt index was practical:
control took 33.06 s / approximately 37.44 ms per evaluated turn; adopted
window 40 took 24.34 s / approximately 34.68 ms per turn (fewer total turns
because it converges earlier). Evidence is cached once at index construction.

Anti-leak audit passed: runtime code in `agent/` has no access to
`ground_truth`, target ASINs, `intent_card`, evaluator-private state, or
public labels. It uses only messages, `user_profile`, catalog contents, turn,
and internal session state. Evaluation labels occur only in the external A/B
script and untouched evaluator, never in the Agent or reranker.

**Verdict: ADOPTED.** Every adoption gate passed: exact disabled control,
full-set HR/TS gains, both-half gains, all-scenario gains, zero hit-to-miss
flips, positive bootstrap interval against control, deterministic output,
no target leakage, and practical latency. `results_memory_agent.json` was
regenerated with the adopted window-40 default.

## Iteration 5 — private-set generalization audit: field-scope widening (ADOPTED) + `fabric` vocab entry (TESTED AND REJECTED) — 2026-08-30

Motivated by a real concern: every tuning decision through iteration 4 was
validated only against the 200 public sessions, and the private 800 sample
a different, much larger slice of the *same* 50,000-product catalog. Traced
every place `evaluator/local_evaluator.py` generates a customer utterance
(`initial_message`, `customer_reply`, `behavior_for`) and confirmed there is
**no free-text paraphrasing anywhere in the shipped simulator** — every
utterance interpolates literal catalog title/feature/detail text into a
fixed template. So "the private set might phrase things differently" isn't
the actual risk; the real, locally-measurable question is whether our
controlled vocabulary and product-side extraction cover enough of the
*full* 50k catalog's literal text, not just the ~200 products the public
sessions happen to target.

### Diagnostic (`scripts/diagnose_vocab_coverage.py`)

Measured two things across all 50,000 catalog products, not just the 200
public targets:

1. **True vocabulary gap is small.** Cross-checked our MATERIAL_VOCAB/
   COLOR_VOCAB against evaluator's own narrow `MATERIAL_RE`/`COLOR_RE` (the
   regexes that drive what the simulated customer discloses). Once compared
   on equal field scope, our vocab misses only **4.0%** of the evaluator's
   material hits and **0.3%** of its color hits.
2. **The real, much bigger gap is field scope, not vocabulary.**
   `agent/memory.py::get_attribute_value` (product side, used by
   `select_clarification_attribute`'s entropy×coverage scoring) only ever
   searched `title+features`. The evaluator's own `searchable_text()` (used
   to build every customer disclosure) searches `title, features, details,
   description, categories, store`. Widening our search to just
   `title+features+details+description` (deliberately excluding
   `store`/`categories` — see below) recovers, out of 50,000 products:
   +1369 material, +2999 color, +3480 size, +4575 style, +3871 use_case,
   +1827 feature matches — using the exact same trusted exact-match logic,
   just pointed at more text.

### Change 1 — widen `get_attribute_value`'s field scope: ADOPTED

Added `_product_slot_text()` in `agent/memory.py`, searching
`title+features+details+description`. `store`/`categories` are deliberately
excluded even though the evaluator reads them — those are exactly the
fields whose raw text caused the original brand/category false-positive bug
(see the `_contains_term` docstring in `vocab.py`); category and brand
already have dedicated, safe extraction paths (`deepest_category_segment`,
`store` field) and don't need to be re-mined here.

A/B against the pre-iteration-5 baseline (`python -m scripts.run_memory_agent`):

| | HitRate@10 | MRR | MTTC | TechnicalScore | hit→miss |
|---|---|---|---|---|---|
| control (iter 4) | 0.915 | 0.629577 | 3.595 | 0.794473 | — |
| **field-scope widening (adopted)** | **0.915** | **0.632494** | 3.645 | 0.794348 | **0** |

Zero hit-to-miss regressions, HR@10 unchanged, MRR improved. TechnicalScore
is essentially flat (a small MRR gain roughly offset by a small MTTC
give-back) — adopted anyway because it fixes a real, measured, substantial
coverage gap against the full catalog the private 800 actually draws from,
not because of the (negligible) public-set delta. This is the more honest
form of "does this generalize" evidence available here: the public 200
can only prove *this specific change doesn't regress what's already
observed*, not that it helps on unseen private-set products — but the
field-scope fix is validated directly against all 50,000 catalog products,
not extrapolated from 200.

### Change 2 — add `fabric` to `MATERIAL_VOCAB`: TESTED AND REJECTED

`fabric` is the single largest residual true-vocabulary-gap term (1247/1131
products even after the field-scope fix) and looked zero-ambiguity to add —
it's literally in `evaluator.local_evaluator.MATERIALS`. Adding it broke
`public_0071` (intent_override): hit at turn 5 rank 1 → total miss.

**Root cause, traced turn-by-turn against the unmodified agent:** this is
*not* an override-vs-paraphrase confusion — `detect_override()` only ever
treats a new value as conflicting when a contradiction cue is present *and*
the value differs from what's filled, so a synonym resolving to the *same*
canonical value can never trigger a false override. What actually happened:
adding `fabric` gave more pool candidates a recognized `material` value,
which shifted `material`'s entropy×coverage score just enough to flip
which attribute got asked at turn 3 for this session's specific pool
(`material` instead of `feature`). That reordering meant the scripted
override at turn 4 landed on a different pending slot than before, so the
one disclosure that identified the target ("90% Cotton, 10% Others") never
got asked for in time. Confirmed by isolating each change independently
(`git stash` one file at a time + re-run): field-scope alone reproduces the
original turn sequence exactly and has zero regressions; `fabric` is the
sole cause of the flip. `MATERIAL_VOCAB` reverted; the field-scope fix
shipped without it. Don't re-add `fabric` (or any single new controlled-vocab
term) without re-running the full 200 + checking hit-to-miss zero — this is
a concrete demonstration that even a single, well-justified vocabulary
addition can cascade through the clarification-order scoring in
non-obvious, session-specific ways.

### Turn-1 attribute yield rates (`scripts/diagnose_attribute_yield.py`)

Separately measured, per askable attribute, how often asking it FIRST (at
turn 1, given whatever the opening message already discloses) gets a real
disclosure from `customer_reply()` vs. a dead "I don't have an additional
preference" reply, across all 200 public sessions:

| attribute | yield | attribute | yield |
|---|---|---|---|
| `other` | 100% | `style` | 8.5% |
| `feature` | 96% | `size` | 4.5% |
| `material` | 72.5% | `use_case` | 2% |
| `color` | 25.5% | `brand`/`budget`/`category` | 0% |

Holds consistently across all four scenarios. Two concrete implications,
not yet implemented:

- **`budget` belongs in `UNASKABLE_SLOTS` alongside `brand`/`category`.**
  Same standard of evidence already used for the other two (0/200, same
  root cause already documented under "Known dead ends" —
  `intent_card()`'s price candidate is structurally sliced off the
  disclosable list). Currently still live in
  `select_clarification_attribute`'s candidate pool for no benefit. Trivial,
  zero-risk, not yet applied.
- **`select_clarification_attribute` has no notion of yield rate** — it
  scores purely on pool entropy×coverage + profile bonus, so it can lead
  with `style` (8.5%) or `use_case` (2%, as dead as `budget`) over
  `feature`/`material`. A yield-rate-aware term (same additive-term pattern
  as `preference_tag_bonus`) is a plausible lever for both MRR and MTTC
  together, since a wasted low-yield ask delays every subsequent turn's
  constraint-rerank signal by one turn for nothing. Proposed but NOT
  implemented or tested — needs the same sweep → split-half →
  zero-hit-to-miss-regression gate as everything else here before being
  trusted, and iteration 5's `fabric` result is a fresh reminder that
  changes touching this scoring function can have non-obvious,
  session-specific cascading effects.
- `other` bypasses `classify_constraint()` entirely (100% yield) but isn't a
  tracked `SLOT_NAME` — if `select_clarification_attribute` ever returned
  it, `state.pending_ask = "other"` would `KeyError` on
  `state.slots[state.pending_ask]` in `update_state`'s Step D.5 (caught by
  the evaluator's `try/except`, degrading to an empty response for that
  turn rather than crashing the harness, but still a silently dead turn).
  Real but smaller upside than it looks (only ~4 points above what
  `feature` already gets), and unimplemented — needs `memory.py` changes to
  track it outside the slot dict before it's safe to return.

## Iteration 6 — budget/brand/category: exclude from primary ask pool, keep as last-resort fallback — ADOPTED — 2026-08-30

Direct follow-up to a stated concern: iteration 5 measured brand/category/
budget at 0% turn-1 yield on the public 200 and recommended excluding
`budget` the same way `brand`/`category` already were. But a hard,
permanent exclusion assumes the public 200's 0% generalizes perfectly to
the private 800, and that assumption deserves its own evidence rather than
being taken on faith.

**Checked how airtight each exclusion actually is, rather than treating
all three the same:**
- `brand`/`category` are provably, not just empirically, impossible:
  `classify_constraint()`'s source has exactly seven possible return
  values (`budget/material/color/size/style/use_case/feature`) — there is
  no code path that can ever return `"brand"` or `"category"`, for any
  input, on any product. This holds for the private 800 with certainty
  equal to the "same template" assurance itself.
- `budget` is structurally *likely* to fail (intent_card() appends the
  price candidate last to a list sliced to `[:2]`/`[2:4]`), but not
  impossible. Measured directly against the full 50k catalog: **234/50,000
  products (0.468%)** have few enough other candidates that a price
  constraint survives the slice and would classify as `budget` if that
  product were ever sampled as a target. 0/200 public sessions hitting this
  is fully consistent with a true rate of ~0.47% (expected ~0.9
  occurrences) — it is not evidence the private 800 will also see zero.

**Design, in `agent/agent.py`:** replaced `UNASKABLE_SLOTS` with
`PRIMARY_UNASKABLE_SLOTS = {"brand", "category", "budget"}`. `respond()`
calls `select_clarification_attribute` with that exclusion first as
before; if it returns `None` (every other slot genuinely exhausted:
filled, no-preference, or already asked-and-answered) with turns
remaining, it retries once with no exclusions at all, falling back to
brand/category/budget rather than asking nothing. Per the existing
pool-narrowing finding (a non-ask still produces a wasted, uninformative
"not quite right yet" reply), there is no downside to trying — the
fallback only ever engages once nothing else is left, so it costs at most
one turn in the case where the private 800 behaves exactly like the public
200, and recovers real signal in the ~0.5% budget case (and, per the
`brand`/`category` proof above, provides free insurance against the small
residual chance the "same template" assurance isn't byte-for-byte exact).

A/B against iteration 5 (`python -m scripts.run_memory_agent`):

| | HitRate@10 | MRR | MTTC | TechnicalScore | hit→miss |
|---|---|---|---|---|---|
| iteration 5 (control) | 0.915 | 0.632494 | 3.645 | 0.794348 | — |
| **iteration 6 (adopted)** | **0.920** | **0.631383** | **3.635** | **0.796715** | **0** |

Zero hit-to-miss regressions; one clean miss→hit (`public_0087`,
browsing — matches the scenario breakdown exactly, browsing
0.9375→0.95, +1/80); two pre-existing hits shuffled rank by ±1 turn. No
scenario regressed (boundary 0.700, buying 0.9375, intent_override 0.8667
all unchanged). All 14 tests still pass.

## Suggested priority for the next session

Iteration 4 disproved the claim that all deterministic constraint-aware
reranking was exhausted; iteration 5 disproved the claim that the only
generalization risk was "the private set might phrase things differently"
(it doesn't); iteration 6 replaced a faith-based permanent exclusion with
one backed by full-catalog evidence, plus a zero-cost fallback tier. The
agent sits at HR@10 0.920 / MRR 0.631383 / MTTC 3.635 / TS 0.796715.

What actually remains, in order:

1. **Yield-rate-aware clarification scoring** (iteration 5's attribute-yield
   table). Plausible joint MRR/MTTC win; needs the full validation gate
   before adopting, and expect non-obvious interactions given both the
   `fabric` result and iteration 6's own fallback tier (re-check the
   fallback path still behaves once this scoring changes).
2. **The 6 known retrieval failures** (gold never entered the 200-pool in the
   pre-iteration ceiling diagnostic; half are jeans with a generic "Jeans"
   category segment) cap
   any reranker at HR@10 0.97. A second retrieval route (a strict
   category-segment pool merged with the OR pool) is the only way past them
   — ~+0.015 TS max and it risks reordering everything else. Low priority,
   but it's the one untried *retrieval* idea.
3. Compare window 40 with 100/200 on any genuinely new validation set; public
   paired bootstrap could not distinguish them, so 40 is the conservative
   default rather than proof that larger windows cannot generalize better.
4. Otherwise redirect effort to the report and demo. Previously rejected
   experiments remain documented and disabled; Iteration 4 is the adopted
   offline deterministic ranking mechanism, iteration 5's field-scope
   widening is the adopted generalization fix, and iteration 6's fallback
   tier is the adopted private-set hedge.

## Useful scripts

| Script | Purpose |
|---|---|
| `scripts/run_memory_agent.py` | Score this agent on the 200 public sessions |
| `scripts/tune_field_weights.py` | Sweep BM25 per-field weights |
| `scripts/tune_policy_constants.py` | Coordinate sweep of `pool_threshold`/`budget_tolerance`/`preference_tag_bonus` |
| `scripts/diagnose_recall_gap.py` | Measure recall-in-pool vs. recall-in-top-10 (the reranker-opportunity diagnostic) |
| `scripts/validate_weight_split.py` | Split-half overfitting check for the field-weight winner |
| `scripts/diagnose_pool_size.py` | Per-turn candidate-pool size + repeated-`ask_attribute` streaks (the stuck-loop / `pool_threshold` instrument) |
| `scripts/tune_pool_narrowing.py` | A/B sweep of the score-cutoff and conjunction narrowing passes (config 1 is the inert-when-disabled regression check) |
| `scripts/tune_field_weights2.py` | iter 2: finer field-weight search around the sweep winner + split-half (concluded: no change) |
| `scripts/tune_pref_bonus.py` | iter 2: extended `preference_tag_bonus` sweep 0.0–3.0 + split-half (adopted 1.5) |
| `scripts/validate_policy_split.py` | iter 2: split-half check for policy constants (`preference_tag_bonus` by default; takes values as argv) |
| `scripts/ab_strip_boilerplate.py` | iter 2 Task 2: A/B the `strip_boilerplate_query` flag (rejected) |
| `scripts/trace_boundary_misses.py` | iter 2 Task 3: full turn-by-turn transcript of every boundary session |
| `scripts/ab_extraction_fixes.py` | iter 2 Task 3: A/B the jeans→denim and 1-char-size extraction fixes (both rejected) |
| `scripts/ab_embed_rerank.py` | iter 2 Task 4: A/B the MiniLM embedding reranker across window sizes (rejected) |
| `scripts/diagnose_ceiling.py` | iter 3 Phase 1: retrieval-vs-ranking split of every miss + oracle HR/MRR, at bonus 0.2 and 1.5 |
| `scripts/ab_noninformative_filter.py` | iter 3 Phase 2: A/B the structural non-informative-turn query filter, paired diff + bootstrap CI (rejected, −0.159) |
| `scripts/tune_slot_boost.py` | iter 3 step 1: sweep `slot_boost_k` (repeat confirmed slot terms in the MATCH expr), paired diff + bootstrap CI (rejected, monotone negative) |
| `scripts/ab_constraint_rerank.py` | iter 4: control vs exact windows 20/40/100/200, scenario/split/paired/bootstrap/runtime report (adopted window 40) |
| `scripts/diagnose_vocab_coverage.py` | iter 5: measures agent/vocab.py coverage across the FULL 50k catalog (not just 200 sessions), production vs. moderate vs. wide field scope, cross-checked against evaluator's own MATERIAL_RE/COLOR_RE |
| `scripts/diagnose_attribute_yield.py` | iter 5: turn-1 yield rate per `ask_attribute` against `evaluator.local_evaluator.customer_reply()`, overall and per scenario |

All reuse `evaluator.local_evaluator` helpers so the simulated-customer
protocol stays identical to the real scoring path; none of them modify
`evaluator/` or the public labels.
