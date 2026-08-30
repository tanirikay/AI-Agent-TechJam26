"""Optional embedding-similarity reranker (Task 4).

Sits ON TOP OF the BM25 recall pool exactly like the disabled
`Agent._slot_match_rerank` hook -- it only reorders an already-retrieved
pool, never changes recall. Disabled by default (Agent(embed_rerank_window=None)).

Rationale: scripts/diagnose_recall_gap.py showed the target is in the
~200-candidate BM25 pool ~89% of the time but in the scored top 10 only
~50% of the time. A keyword-match rerank was already tried and rejected
(CLAUDE.md) because a binary slot match carries no graded similarity signal.
This uses a real sentence embedding (all-MiniLM-L6-v2, CPU, ~90MB, runs
offline once the model is cached) so a candidate that is *semantically*
close to the accumulated query text -- not just sharing a keyword -- is what
gets promoted.

Dependency note: this module imports sentence-transformers/torch lazily, so
the default agent (embed_rerank_window=None) never imports them and stays
stdlib + sqlite only. Only relevant if this hook is ever enabled.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np

_MODEL_NAME = "all-MiniLM-L6-v2"
_CACHE_DIR = Path(os.environ.get("TECHJAM_EMBED_CACHE", ".embed_cache"))

# process-global singleton so a sweep that builds many Agents over one shared
# catalog index pays the ~2min encode + load once, not per Agent.
_SHARED: "EmbeddingReranker | None" = None


def get_shared_reranker(products: dict[str, dict]) -> "EmbeddingReranker":
    global _SHARED
    if _SHARED is None:
        _SHARED = EmbeddingReranker()
        _SHARED.build_or_load(products)
    return _SHARED


_TEXT_CHAR_CAP = 500  # MiniLM truncates at 256 wordpieces (~1000 chars) anyway;
                      # capping the raw string keeps tokenization of 50k long
                      # feature blobs from dominating the one-time encode.


def _product_text(product: dict) -> str:
    title = str(product.get("title") or "")
    features = product.get("features")
    if isinstance(features, list):
        feat = " ".join(str(f) for f in features)
    else:
        feat = str(features or "")
    return f"{title} {feat}".strip()[:_TEXT_CHAR_CAP]


class EmbeddingReranker:
    def __init__(self, model_name: str = _MODEL_NAME, cache_dir: Path = _CACHE_DIR) -> None:
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self._model = None
        self.asin_order: list[str] = []
        self.matrix: np.ndarray | None = None  # (N, d) L2-normalized float32
        self._row_of: dict[str, int] = {}

    # -- lazy model load (imports torch only when first needed) --
    def _get_model(self):
        if self._model is None:
            try:
                import torch

                torch.set_num_threads(max(1, (os.cpu_count() or 4)))
            except Exception:
                pass
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _catalog_fingerprint(self, products: dict[str, dict]) -> str:
        h = hashlib.sha256()
        h.update(str(len(products)).encode())
        for asin in sorted(products)[:2000]:
            h.update(asin.encode())
        h.update(self.model_name.encode())
        return h.hexdigest()[:16]

    def build_or_load(self, products: dict[str, dict]) -> None:
        fp = self._catalog_fingerprint(products)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        mat_path = self.cache_dir / f"prod_{fp}.npy"
        order_path = self.cache_dir / f"prod_{fp}.txt"

        if mat_path.exists() and order_path.exists():
            self.asin_order = order_path.read_text(encoding="utf-8").split("\n")
            self.matrix = np.load(mat_path)
        else:
            self.asin_order = sorted(products)
            texts = [_product_text(products[a]) for a in self.asin_order]
            model = self._get_model()
            emb = model.encode(
                texts, batch_size=128, convert_to_numpy=True,
                normalize_embeddings=True, show_progress_bar=False,
            ).astype(np.float32)
            self.matrix = emb
            np.save(mat_path, emb)
            order_path.write_text("\n".join(self.asin_order), encoding="utf-8")

        self._row_of = {a: i for i, a in enumerate(self.asin_order)}

    def similarity(self, asins: list[str], text: str) -> dict[str, float]:
        """Cosine similarity between `text` and each listed product's cached
        embedding. Products missing from the cache are simply omitted from
        the result (caller treats that as "no signal", not zero).

        Distinct from `rerank()`: this scores arbitrary text (e.g. one raw
        disclosed clause) against a candidate set, without reordering
        anything itself -- used by the constraint-rerank semantic fallback
        to score a single lost clause, not the whole accumulated
        conversation, which is precisely what made the whole-pool
        `rerank()` hook lose badly (see CLAUDE.md iter 2 Task 4).
        """
        if self.matrix is None or not text.strip():
            return {}
        model = self._get_model()
        q = model.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0].astype(np.float32)
        result: dict[str, float] = {}
        for asin in asins:
            row = self._row_of.get(asin)
            if row is not None:
                result[asin] = float(self.matrix[row] @ q)
        return result

    def rerank(self, pool_asins: list[str], query_text: str, window: int | None) -> list[str]:
        """Reorder the first `window` of pool_asins by cosine similarity to
        query_text; leave the tail untouched and appended. window=None
        reranks the whole pool.
        """
        if self.matrix is None or not pool_asins or not query_text.strip():
            return pool_asins
        head = pool_asins if window is None else pool_asins[:window]
        tail = [] if window is None else pool_asins[window:]

        rows = [self._row_of.get(a, -1) for a in head]
        known = [(a, r) for a, r in zip(head, rows) if r >= 0]
        if not known:
            return pool_asins

        model = self._get_model()
        q = model.encode([query_text], convert_to_numpy=True, normalize_embeddings=True)[0].astype(np.float32)
        idx = np.array([r for _, r in known])
        sims = self.matrix[idx] @ q
        order = np.argsort(-sims)
        reranked = [known[i][0] for i in order]
        # any head asin not in the embedding matrix keeps original order, at the end of the head
        missing = [a for a, r in zip(head, rows) if r < 0]
        return reranked + missing + tail
