# AI-Agent-TechJam26

Team repo for the TechJam Conversational E-Commerce Search Challenge. Build an
agent that finds a hidden target product within 10 turns by asking useful
follow-up questions, tracking session state, and reranking candidates.

## Repo layout — dev environment vs. submission bundle

This repo has two zones. Don't blur them — the submission bundle is graded in
an environment that only ever sees what's inside `submission/`.

```
data/, docs/, evaluator/, tests/   <- participant kit reference material.
                                       Read-only. Never edit evaluator/ or
                                       docs/, never edit data/public_set.jsonl.
                                       Not part of the submission.

starter/agent.py                    <- editable weak baseline. Rewrite this,
                                       or replace it entirely.

submission/                         <- what actually ships. See
                                       submission/README.md for the contract.
```

Nothing under `data/`, `docs/`, `evaluator/`, or `tests/` should ever be
copied into `submission/`. Those are organizer-provided artifacts for local
development and scoring, not your code.

## Environment setup

Requires Python 3.10+ (verified locally against 3.13.6). Standard library
only — no dependencies to install for the starter kit itself.

1. Download `catalog.jsonl.gz` from the [participant-kit release](https://github.com/TechJam2026/techjam-conversational-search/releases/tag/participant-kit)
   and verify it against the published `SHA256SUMS`.
2. Decompress into `data/catalog.jsonl` (50,000 rows expected). This file is
   gitignored — regenerate it locally, don't commit it.
3. Run the local evaluator:
   ```
   python -m evaluator.local_evaluator
   ```
   Writes `results.json` with aggregate + per-scenario metrics. The weak BM25
   baseline should reproduce exactly: `hit_rate_at_10 0.125`, `mrr 0.068034`,
   `mttc 9.81` (see `docs/baseline_results.json`).

## Scoring

```
TechnicalScore = 0.50 × HitRate@10 + 0.30 × MRR + 0.20 × Efficiency
Efficiency     = clip((11 - MTTC) / 10, 0, 1)
```

Reported per scenario (buying 40% / browsing 40% / intent_override 15% /
boundary 5%) — treat the breakdown as a regression gate, not just the
aggregate.

## Source

- Participant repo: https://github.com/TechJam2026/techjam-conversational-search
- Participant kit release: https://github.com/TechJam2026/techjam-conversational-search/releases/tag/participant-kit
- Data source: https://amazon-reviews-2023.github.io/ (Amazon Reviews 2023, McAuley Lab, UCSD — see `DATA_ATTRIBUTION.md`)
