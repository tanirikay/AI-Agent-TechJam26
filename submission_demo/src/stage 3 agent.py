from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.hybrid_search import HybridSearch
from src.reranker import Stage3Ranker

ALLOWED_ATTRIBUTES = {
    "category", "material", "color", "size", "style", "brand",
    "budget", "feature", "use_case", "other",
}

BUYING_SIGNALS = re.compile(
    r"\b(buy|purchase|order|need|size|under|below|budget|exact|must have)\b",
    re.IGNORECASE,
)


class Agent:
    """Multi-turn agent conforming strictly to the TechJam Contract Schema."""

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.catalog_path = Path(catalog_path)
        self.search_engine = HybridSearch(self.catalog_path)
        self.ranker = Stage3Ranker()
        self.profiles: Dict[str, Dict[str, Any]] = {}
        self.turn_history: Dict[str, List[str]] = {}
        
        self.catalog_lookup: Dict[str, Dict[str, Any]] = {}
        self._load_catalog_lookup()

    def _load_catalog_lookup(self) -> None:
        if not self.catalog_path.exists():
            return
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                item = json.loads(line)
                self.catalog_lookup[str(item["parent_asin"])] = item

    def reset(self, session_id: str, user_profile: Dict[str, Any]) -> None:
        self.profiles[session_id] = user_profile or {}
        self.turn_history[session_id] = []

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int = 10,
    ) -> Dict[str, Any]:
        if session_id not in self.profiles:
            raise RuntimeError("reset must be called before respond")

        # Track turn history to handle multi-turn intent overrides
        self.turn_history[session_id].append(user_message)
        context_query = " ".join(self.turn_history[session_id])

        profile = self.profiles[session_id]

        # Stage 1: Mode & Clarification selection
        mode = self._detect_mode(profile, user_message)
        ask_attr = self._select_clarification_attribute(turn, mode, user_message)

        # Stage 2: Candidate Retrieval
        shortlist = self.search_engine.retrieve(
            context_query,
            mode,
            candidate_count=30,
        )

        # Stage 3: Score Fusion and Re-ranking
        final_rankings, usage = self.ranker.rank(
            query=context_query,
            candidates=shortlist,
            catalog_lookup=self.catalog_lookup,
            mode=mode,
            top_k=top_k,
        )

        return {
            "message": "Here are the best matching items based on your request.",
            "ask_attribute": ask_attr,
            "recommendations": [
                {
                    "parent_asin": result.parent_asin,
                    "score": round(result.final_score, 6),
                }
                for result in final_rankings
            ],
            "usage": usage,
        }

    @staticmethod
    def _detect_mode(profile: Dict[str, Any], message: str) -> str:
        declared = str(profile.get("scenario_type", "")).lower()
        if declared in {"buying", "browsing"}:
            return declared
        return "buying" if BUYING_SIGNALS.search(message) else "browsing"

    @staticmethod
    def _select_clarification_attribute(
        turn: int, mode: str, message: str
    ) -> Optional[str]:
        # Return an allowed attribute on early browsing turns; null when specific
        if turn > 4 or mode == "buying":
            return None

        message_lower = message.lower()
        if "material" not in message_lower:
            return "material"
        if "color" not in message_lower:
            return "color"
        if "size" not in message_lower:
            return "size"

        return None
