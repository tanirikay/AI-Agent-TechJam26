# AI-Agent-TechJam26

A conversational shopping agent for the TechJam Conversational E-Commerce
Search Challenge. Given a shopper's message and up to 10 turns, it finds a
hidden target product by tracking what's been said, asking the question that
narrows the field the most, and reranking candidates against the exact facts
the shopper has disclosed.

**Result on the 200-session public benchmark: HitRate@10 0.925, MRR 0.6345,
MTTC 3.53, TechnicalScore 0.8022** — up from a 0.125 HitRate@10 / 0.107
TechnicalScore weak-baseline starting point.

## Repo layout

This whole repository is the submission — there is no separate bundle folder.
The required entry point is the single `agent.py` at the repo root.

```
agent.py                            <- the submission entry point. Exports
                                       Agent, matching
                                       docs/agent_api_contract.json exactly.
                                       Frozen to the adopted configuration --
                                       no experimental constructor switches.

src/                                <- the five runtime modules agent.py
                                       imports: memory, vocab, retrieval,
                                       constraint_rerank, price_rerank.
                                       Everything agent.py needs at runtime
                                       lives here; nothing else is required.

agent_dev/                          <- the SAME agent logic, but with every
                                       experimental constructor switch tried
                                       during development left in (slot
                                       boosting, pool narrowing, embedding
                                       rerank, semantic fallback, ...) and
                                       still configurable. Used by scripts/
                                       and tests/ to reproduce the tuning
                                       and ablation history -- not imported
                                       by agent.py, and not required to run
                                       the submission. See CLAUDE.md for why
                                       each switch is where it landed.

scripts/                            <- evaluation, tuning, and diagnostic
                                       tooling (imports agent_dev/), plus
                                       runnable live demos (scripts/demo_*.py
                                       and scripts/run_memory_agent.py import
                                       the actual root agent.py instead, since
                                       those represent "run the submission").

data/, docs/, evaluator/, tests/    <- participant kit reference material
                                       (data/, docs/, evaluator/) plus the
                                       test suite. Never edit evaluator/ or
                                       docs/, never edit data/public_set.jsonl.

starter/agent.py                    <- the original weak baseline, kept
                                       unmodified for reference/comparison.
```

## Setup and installation

Requires Python 3.10+ (developed and verified against 3.13.6). The agent
itself is standard library + SQLite only — no third-party dependencies for
the adopted, shipped configuration. (`agent_dev/`'s optional, disabled-by-
default semantic-fallback experiment needs `sentence-transformers`/`torch`
if ever explicitly enabled — see `CLAUDE.md` iteration 7 — but it's never
imported by default, and never imported at all by the root `agent.py`.)

1. Download `catalog.jsonl.gz` from the [participant-kit release](https://github.com/TechJam2026/techjam-conversational-search/releases/tag/participant-kit)
   and verify it against the published `SHA256SUMS`.
2. Decompress into `data/catalog.jsonl` (50,000 rows expected). This file is
   gitignored — regenerate it locally, don't commit it.
3. No `pip install` needed to run the submitted agent.
4. The agent needs a SQLite build with the FTS5 extension, which CPython's
   bundled `sqlite3` module ships with by default on all common platforms —
   verify with:

   ```bash
   python -c "import sqlite3; sqlite3.connect(':memory:').execute('CREATE VIRTUAL TABLE t USING fts5(x)'); print('FTS5 OK')"
   ```

## Steps to reproduce results

```bash
# The actual submitted agent (root agent.py) -- this is "your results".
python -m scripts.run_memory_agent
# Writes results_memory_agent.json. Expect HitRate@10 0.925, MRR 0.6345,
# MTTC 3.53, TechnicalScore 0.8022 (deterministic, byte-identical run to run).

# The untouched weak starter, for reference/comparison only.
python -m evaluator.local_evaluator
# Writes results.json. Expect HitRate@10 0.125, MRR 0.068, MTTC 9.81
# (matches docs/baseline_results.json).
```

Or use `agent.py` directly:

```python
from agent import Agent

agent = Agent()  # looks for data/catalog.jsonl; or Agent(catalog_path=...)
agent.reset(session_id, user_profile)
response = agent.respond(session_id, user_message, turn, top_k=10)
```

Run the test suite:

```bash
python -m unittest discover -s tests -q
```

See a live multi-turn session instead of just the aggregate score:

```bash
python -m scripts.demo_conversation          # real session, pauses per turn
python -m scripts.demo_conversation --auto    # same, no pauses
python -m scripts.demo_negation_layer         # the always-on negation layer
python -m scripts.demo_defensive_fallback     # the brand/category/budget fallback tier
```

## Demo: one multi-turn session

Real output from `python -m scripts.demo_conversation`, not staged — public
session `public_0108`, target `Hanes Women's Stretch Jersey Bike Shorts`
($13.00), unmodified `agent.py`:

| Turn | Customer says | Agent asks (`ask_attribute`) | Target rank |
|---|---|---|---|
| 1 | "I'm looking for Pants & Shorts Shorts. A key requirement is: cotton." | `style` | not in top 10 |
| 2 | "I don't have an additional preference for style." | `feature` | not in top 10 |
| 3 | "For that, what matters is: Imported; Elastic closure." | *(hit — no further question)* | **1** |

By turn 3 every fact the customer had disclosed — the category phrase,
"cotton", "Imported", "Elastic closure" — appeared word-for-word in the
target's own listing text, which is what pushed it to rank 1. Run
`python -m scripts.demo_conversation` yourself to see this live, including
the exact-match check printed at the moment of the hit.

## Disclosures

- **Network**: none. No code path in `agent.py` or `src/` makes a network
  call, reads an API key, or depends on an external service — fully offline.
  (`agent_dev/`'s disabled-by-default semantic-fallback experiment would need
  network access to fetch a model on first use if ever explicitly enabled,
  but it isn't part of the submitted `agent.py` and isn't imported by it.)
- **Model / cost**: no LLM or ML model is used by the submitted agent;
  retrieval and reranking are deterministic. `usage` in every response is
  genuinely `{0, 0}`, not a stub — there is no token cost.
- **Latency**: index construction (once, at `Agent()` construction) takes
  roughly 10-15 seconds for the full 50,000-product catalog; each
  `respond()` call after that completes in well under 100ms.

## How it works

`agent.py::Agent` combines:

- **Retrieval** (`src/retrieval.py`) — BM25 over a SQLite FTS5 index across
  six weighted fields (title, categories, features, details, store,
  description), queried with the full accumulated conversation, not just the
  latest message.
- **Memory** (`src/memory.py`) — nine tracked slots (category, material,
  color, size, style, brand, budget, feature, use_case), a separate raw-
  constraint memory storing exactly what the shopper said, intent-override
  and no-preference detection, and information-gain-based selection of which
  attribute to ask about next.
- **Deterministic reranking** (`src/price_rerank.py`, `src/constraint_rerank.py`,
  and negation exclusion in `agent.py` itself) — budget/price demotion, an
  always-on negation layer ("not wool" demotes wool-matching candidates), and
  an exact-clause reranker that checks every disclosed fact word-for-word
  against each candidate's own listing text.

Full architecture detail, every design decision, and the complete iteration
history (what was tried, what worked, what was rejected and why) is in
[`CLAUDE.md`](CLAUDE.md) — it's a development log, not judge-facing, but it's
the most accurate record of how this system actually got built. Its prose
predates this repo's restructure and still refers to paths as `agent/*.py`
throughout; that package is now `agent_dev/*.py` plus `src/*.py` — see the
note at the top of `CLAUDE.md`.

## Scoring

```
TechnicalScore = 0.50 × HitRate@10 + 0.30 × MRR + 0.20 × Efficiency
Efficiency     = clip((11 - MTTC) / 10, 0, 1)
```

Reported per scenario (buying 40% / browsing 40% / intent_override 15% /
boundary 5%) — treat the breakdown as a regression gate, not just the
aggregate.

## Limitations and what we'd improve with more time

- **A handful of products never enter the retrieval pool at all** — mostly
  generically-categorized items (e.g. jeans under a bare "Jeans" category) —
  which no amount of reranking downstream can recover, since retrieval never
  surfaces them. A second, category-targeted retrieval route is the untried
  fix.
- **Boundary scenario (customer has no strong preference) is our weakest** —
  70% hit rate, but on only 10 sessions, so treat that as a noisy estimate,
  not a settled result.
- **Everything above is validated against 200 public sessions; final grading
  uses 800 we've never seen.** We built specific, evidence-backed hedges for
  known risks there (a fallback tier for otherwise-unaskable attributes, a
  deterministic negation-exclusion layer) rather than assuming our tuning
  generalizes, but we can't fully confirm it until it's graded.
- **The exact-clause reranker's window size (40 of the top 200 candidates) is
  a statistically-defensible choice, not a proven ceiling** — wider windows
  scored slightly higher on our test set, but the improvement over 40 wasn't
  statistically distinguishable from noise on paired bootstrap resampling. It's
  a single config value, not a redesign, so it's cheap to revisit with more
  validation data.
- **A real semantic/embedding-based reranker does not help here** — we built
  and tested one; it made results worse at every setting, because this task
  rewards exact constraint satisfaction (every disclosed fact is a literal
  quote from the target's own listing) over general semantic similarity. A
  narrower, single-clause-scoped version exists but isn't validated against
  real paraphrased data, since none exists locally — see `CLAUDE.md`
  iteration 7.
- Given more time, our priority order would be: yield-rate-aware clarification
  scoring (asking about attributes shoppers actually answer, which we've
  measured varies from ~96% to under 5% by attribute), the second retrieval
  route above, and validating the reranking-window choice against a genuinely
  new dataset rather than the same 200 sessions we've already tuned against.

## Source

- Participant repo: https://github.com/TechJam2026/techjam-conversational-search
- Participant kit release: https://github.com/TechJam2026/techjam-conversational-search/releases/tag/participant-kit
- Data source: https://amazon-reviews-2023.github.io/ (Amazon Reviews 2023, McAuley Lab, UCSD — see `DATA_ATTRIBUTION.md`)
