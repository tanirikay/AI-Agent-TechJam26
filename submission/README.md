# Submission bundle

This folder is a scaffold — it becomes the actual zip/bundle handed to the
organizer. Nothing in here should assume the rest of this repo exists; the
grading harness only ever sees what's in `submission/`.

## Target layout

```
submission/
  agent.py           <- exports `Agent`, matching starter/agent.py's contract
  requirements.txt   <- only if you add dependencies beyond stdlib
  README.md           <- method, model choice, limitations, network/offline
                          disclosure, latency + token cost estimate
  src/                <- your own local helper modules (memory store,
                          follow-up policy, retrieval, reranking, ...)
```

## Copy into this folder

- Your `Agent` implementation and any local modules it imports.
- A `requirements.txt` only if you actually add a dependency.
- A short report: method, model choice, limitations, whether it needs network
  access (and the offline fallback if final judging disables it), latency,
  token usage, estimated model cost.

## Never copy into this folder

Straight from `docs/submission_rules.md` — these are grounds for an invalid
run, not just style:

- `evaluator/` — the harness re-scores with its own frozen copy; a bundled
  copy is dead weight at best, suspicious at worst.
- `data/catalog.jsonl`, `data/catalog.jsonl.gz`, `data/public_set.jsonl` —
  organizer-provided, frozen, already present in the grading environment.
  Never redistribute the catalog.
- `docs/` — reference material, not your code.
- `tests/` — validates the evaluator itself, not part of your submission.
- Anything under an `organizer/` path, if you ever encounter one — this repo
  never actually ships that directory (the upstream README links to it, but
  it isn't distributed to participants), so there is nothing here to
  accidentally include. If a teammate ever pastes organizer-only content into
  a chat or file, it does not belong in this repo at all, submission folder
  or otherwise.
- API keys, secrets, or credentials of any kind.
- Anything that modifies evaluator files or requires privileged host access.

## Before zipping

- [ ] `python -m evaluator.local_evaluator` run from a copy of this folder
      alone (plus the organizer-provided `data/` + `evaluator/`) reproduces
      your expected local score — proves the bundle is self-contained.
- [ ] `requirements.txt` installs cleanly in a fresh environment.
- [ ] README states network dependency and offline fallback explicitly.
- [ ] No API keys committed.
