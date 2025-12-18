"""
Router Agent - Deterministic query classification to track:
TRACK_DATA (SQL/Pandas), TRACK_DOC (RAG), or TRACK_WEB (Web Search).
"""
import logging
import re
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


# Track constants
TRACK_DATA = "TRACK_DATA"
TRACK_DOC = "TRACK_DOC"
TRACK_WEB = "TRACK_WEB"


# Keyword patterns for deterministic routing
DATA_KEYWORDS = [
    r"\b(sum|total|aggregate|count|average|avg|mean)\b",
    r"\b(growth|variance|change|delta|difference)\b",
    r"\b(fy\d{2}|q\d|9mfy|3mfy)\b",
    r"\b(revenue|sales|income|profit|cost|expense)\b",
    r"\b(from\s+\d{4}\s+to|between\s+\d{4})\b",
    r"\b(what was|how much|calculate|compute)\b",
    r"\b(gmv|arpu|cac|ltv|churn|retention)\b",
    r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}\b",
    r"\b(ledger|balance sheet|income statement|cashflow|p&l)\b",
    r"\b(per order|per unit|unit economics)\b",
    r"\b(how many sheets|list.*sheets|sheet names)\b",  # Metadata queries
]

DOC_KEYWORDS = [
    r"\b(clause|policy|terms|conditions|agreement)\b",
    r"\b(definition|defined as|means)\b",
    r"\b(contract|legal|compliance|regulation)\b",
    r"\b(notes to accounts|accounting policy)\b",
    r"\b(cancellation|termination|liability)\b",
    r"\b(indemnity|warranty|guarantee)\b",
]

WEB_KEYWORDS = [
    r"\b(current|latest|today|2024|2025)\b.*\b(rate|tax|regulation)\b",
    r"\b(budget\s+\d{4}|union budget)\b",
    r"\b(rbi|sebi|gst|income tax)\b.*\b(current|latest|new)\b",
    r"\b(benchmark|industry|market)\b.*\b(rate|standard)\b",
    r"\b(what is the current|latest news)\b",
]


class RouterAgent:
    """
    Deterministic router with LLM fallback.
    Routes queries to appropriate track based on keyword patterns.
    """

    def __init__(self, llm_wrapper=None):
        self._llm = llm_wrapper
        self._route_cache: Dict[str, str] = {}

    def _match_patterns(self, query: str, patterns: List[str]) -> int:
        """Count pattern matches for a query."""
        matches = 0
        query_lower = query.lower()
        for pattern in patterns:
            if re.search(pattern, query_lower, re.IGNORECASE):
                matches += 1
        return matches

    def route(self, query: str, client_context: Optional[str] = None) -> Dict[str, Any]:
        """
        Route query to appropriate track.
        
        Args:
            query: User's question
            client_context: Optional context about available data
            
        Returns:
            Dict with track, confidence, and reasoning
        """
        # Check cache
        cache_key = query.lower().strip()
        if cache_key in self._route_cache:
            cached = self._route_cache[cache_key]
            return {"track": cached, "confidence": 0.95, "method": "cache"}

        # Deterministic pattern matching
        data_score = self._match_patterns(query, DATA_KEYWORDS)
        doc_score = self._match_patterns(query, DOC_KEYWORDS)
        web_score = self._match_patterns(query, WEB_KEYWORDS)

        logger.debug(f"Route scores - DATA: {data_score}, DOC: {doc_score}, WEB: {web_score}")

        # Determine track
        max_score = max(data_score, doc_score, web_score)
        
        if max_score == 0:
            # No clear pattern - use LLM if available
            if self._llm:
                return self._route_with_llm(query, client_context)
            # Default to data track
            track = TRACK_DATA
            confidence = 0.5
            method = "default"
        elif data_score == max_score and data_score > doc_score and data_score > web_score:
            track = TRACK_DATA
            confidence = min(0.6 + (data_score * 0.1), 0.95)
            method = "pattern"
        elif doc_score == max_score and doc_score > data_score:
            track = TRACK_DOC
            confidence = min(0.6 + (doc_score * 0.1), 0.95)
            method = "pattern"
        elif web_score == max_score and web_score > data_score:
            track = TRACK_WEB
            confidence = min(0.6 + (web_score * 0.1), 0.95)
            method = "pattern"
        else:
            # Tie-breaker: prefer DATA track for analytical queries
            track = TRACK_DATA
            confidence = 0.6
            method = "pattern_tiebreak"

        # Cache result
        self._route_cache[cache_key] = track

        return {
            "track": track,
            "confidence": confidence,
            "method": method,
            "scores": {
                "data": data_score,
                "doc": doc_score,
                "web": web_score
            }
        }

    def _route_with_llm(self, query: str, client_context: Optional[str] = None) -> Dict[str, Any]:
        """Use LLM for routing when patterns don't match."""
        try:
            from app.core.prompts import get_router_prompt

            prompt = get_router_prompt(query, client_context or "")
            response = self._llm.invoke(prompt)

            # Extract track from response
            response_upper = response.upper().strip()
            
            if "TRACK_DATA" in response_upper:
                track = TRACK_DATA
            elif "TRACK_DOC" in response_upper:
                track = TRACK_DOC
            elif "TRACK_WEB" in response_upper:
                track = TRACK_WEB
            else:
                # Default to data
                track = TRACK_DATA

            self._route_cache[query.lower().strip()] = track
            
            return {
                "track": track,
                "confidence": 0.8,
                "method": "llm",
                "raw_response": response[:100]
            }

        except Exception as e:
            logger.error(f"LLM routing failed: {e}")
            return {
                "track": TRACK_DATA,
                "confidence": 0.5,
                "method": "fallback",
                "error": str(e)
            }

    def clear_cache(self):
        """Clear routing cache."""
        self._route_cache.clear()


# Singleton instance
_router: Optional[RouterAgent] = None


def get_router_agent(llm_wrapper=None) -> RouterAgent:
    """Get or create singleton router agent."""
    global _router
    if _router is None:
        _router = RouterAgent(llm_wrapper)
    elif llm_wrapper and _router._llm is None:
        _router._llm = llm_wrapper
    return _router


__all__ = ["RouterAgent", "get_router_agent", "TRACK_DATA", "TRACK_DOC", "TRACK_WEB"]
