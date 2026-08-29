from __future__ import annotations

"""Stage 2 submission entry point. Stage 3 re-ranking is intentionally absent."""

from pathlib import Path
import re

from src.hybrid_search import HybridSearch


BUYING_SIGNALS = re.compile(
    r"\b(buy|purchase|order|need|size|under|below|budget|exact|must have)\b",
    re.IGNORECASE,
)


class Agent:
    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.search = HybridSearch(catalog_path)
        self.profiles: dict[str, dict] = {}

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.profiles[session_id] = user_profile or {}

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        if session_id not in self.profiles:
            raise RuntimeError("reset must be called before respond")

        profile = self.profiles[session_id]
        mode = self._mode(profile, user_message)
        shortlist = self.search.retrieve(
            user_message,
            mode,
            candidate_count=30,
        )

        return {
            "message": "I found matches using hybrid catalog search.",
            "ask_attribute": None,
            "recommendations": [
                {
                    "parent_asin": result.parent_asin,
                    "score": round(result.score, 6),
                }
                for result in shortlist[:top_k]
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }

    @staticmethod
    def _mode(profile: dict, message: str) -> str:
        declared = str(profile.get("scenario_type", "")).lower()
        if declared in {"buying", "browsing"}:
            return declared
        return "buying" if BUYING_SIGNALS.search(message) else "browsing"