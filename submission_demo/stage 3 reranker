from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from src.hybrid_search import SearchResult


@dataclass(frozen=True)
class RankedResult:
    parent_asin: str
    stage2_score: float
    ai_score: float
    final_score: float


class Stage3Ranker:
    """Handles LLM candidate re-ranking and multi-score fusion with offline fallback."""

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")

    def rank(
        self,
        query: str,
        candidates: List[SearchResult],
        catalog_lookup: Dict[str, Dict[str, Any]],
        mode: str,
        top_k: int = 10,
    ) -> Tuple[List[RankedResult], Dict[str, int]]:
        if not candidates:
            return [], {"prompt_tokens": 0, "completion_tokens": 0}

        candidate_items = [
            catalog_lookup[c.parent_asin]
            for c in candidates
            if c.parent_asin in catalog_lookup
        ]

        # Stage 3 LLM call with offline protection
        ai_scores, usage = self._llm_rerank(query, candidate_items)

        # Score blending weights based on Buying vs. Browsing mode
        stage2_weight, ai_weight = (
            (0.65, 0.35) if mode == "buying" else (0.35, 0.65)
        )

        ranked_results: List[RankedResult] = []
        for candidate in candidates:
            pasin = candidate.parent_asin
            s2_score = candidate.score
            ai_score = ai_scores.get(pasin, 0.5)

            final_score = (stage2_weight * s2_score) + (ai_weight * ai_score)
            ranked_results.append(
                RankedResult(
                    parent_asin=pasin,
                    stage2_score=s2_score,
                    ai_score=ai_score,
                    final_score=final_score,
                )
            )

        ranked_results.sort(key=lambda item: (-item.final_score, item.parent_asin))
        return ranked_results[:top_k], usage

    def _llm_rerank(
        self, query: str, items: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, float], Dict[str, int]]:
        """Calls OpenAI API via stdlib urllib. Falls back cleanly if offline or key is missing."""
        default_usage = {"prompt_tokens": 0, "completion_tokens": 0}
        
        if not self.api_key or not items:
            return {str(item["parent_asin"]): 0.5 for item in items}, default_usage

        payload_items = [
            {
                "parent_asin": str(item["parent_asin"]),
                "title": str(item.get("title", ""))[:120],
                "description": str(item.get("description", ""))[:150],
            }
            for item in items
        ]

        prompt = (
            f"User Query: '{query}'\n\n"
            f"Items:\n{json.dumps(payload_items)}\n\n"
            "Score the relevance of each item to the user query between 0.0 and 1.0. "
            "Return ONLY a JSON object mapping parent_asin to score. "
            'Example: {"B0001": 0.85, "B0002": 0.12}'
        )

        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            data=json.dumps(
                {
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0,
                    "response_format": {"type": "json_object"},
                }
            ).encode("utf-8"),
        )

        try:
            with urllib.request.urlopen(req, timeout=3) as response:
                result = json.loads(response.read().decode("utf-8"))
                content = result["choices"][0]["message"]["content"]
                parsed_scores = json.loads(content)
                
                usage_data = result.get("usage", {})
                usage = {
                    "prompt_tokens": int(usage_data.get("prompt_tokens", 0)),
                    "completion_tokens": int(usage_data.get("completion_tokens", 0)),
                }
                
                return {str(k): float(v) for k, v in parsed_scores.items()}, usage
        except Exception:
            # Fallback for network timeouts or disabled external APIs
            return {str(item["parent_asin"]): 0.5 for item in items}, default_usage
