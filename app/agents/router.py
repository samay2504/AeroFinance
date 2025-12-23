"""
Router Agent - Semantic query classification using NLP.
Uses spaCy for similarity matching and LLM for intent classification.
Routes to: TRACK_DATA (SQL/Pandas), TRACK_DOC (RAG), or TRACK_WEB (Web Search).
"""
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


# Track constants
TRACK_DATA = "TRACK_DATA"
TRACK_DOC = "TRACK_DOC"
TRACK_WEB = "TRACK_WEB"

# Try to load spaCy for semantic similarity
_nlp = None
_SPACY_AVAILABLE = False

try:
    import spacy
    # Try to load a model with word vectors
    try:
        _nlp = spacy.load("en_core_web_md")  # Medium model has word vectors
        _SPACY_AVAILABLE = True
        logger.info("Loaded spaCy model: en_core_web_md")
    except OSError:
        try:
            _nlp = spacy.load("en_core_web_sm")  # Small model (limited vectors)
            _SPACY_AVAILABLE = True
            logger.info("Loaded spaCy model: en_core_web_sm (limited vectors)")
        except OSError:
            logger.warning("No spaCy model found. Run: python -m spacy download en_core_web_md")
except ImportError:
    logger.info("spaCy not available. Using pattern-based routing only.")


# Intent exemplars for semantic similarity matching
DATA_INTENT_EXEMPLARS = [
    "how many sheets are there",
    "what are the sheet names",
    "show me the data",
    "calculate the total",
    "what is the sum",
    "list all records",
    "find the average",
    "what is nifty",
    "show volume gainers",
    "top stocks by price",
    "count the rows",
    "growth rate",
    "compare values",
]

DOC_INTENT_EXEMPLARS = [
    "what does clause 5 say",
    "find the cancellation policy",
    "accounting policy for inventory",
    "terms and conditions",
    "legal definition",
    "contract agreement",
    "compliance requirement",
]

WEB_INTENT_EXEMPLARS = [
    "current repo rate",
    "latest GST rules",
    "today's market news",
    "current tax rate",
    "real-time stock price",
    "2024 budget announcement",
]


class RouterAgent:
    """
    Semantic router with LLM fallback.
    Uses spaCy similarity for intent matching, LLM for complex cases.
    No hardcoded keyword matching - fully dynamic.
    """

    def __init__(self, llm_wrapper=None):
        self._llm = llm_wrapper
        self._route_cache: Dict[str, str] = {}
        self._has_loaded_data = False
        self._loaded_datasets_info = ""
        
        # Pre-compute intent exemplar docs for spaCy similarity
        self._data_docs = []
        self._doc_docs = []
        self._web_docs = []
        
        if _SPACY_AVAILABLE and _nlp:
            self._data_docs = [_nlp(text) for text in DATA_INTENT_EXEMPLARS]
            self._doc_docs = [_nlp(text) for text in DOC_INTENT_EXEMPLARS]
            self._web_docs = [_nlp(text) for text in WEB_INTENT_EXEMPLARS]
            logger.debug("Pre-computed spaCy docs for intent exemplars")

    def set_data_context(self, has_data: bool, datasets_info: str = ""):
        """Set whether data is currently loaded and what datasets."""
        self._has_loaded_data = has_data
        self._loaded_datasets_info = datasets_info

    def _compute_intent_similarity(self, query: str) -> Dict[str, float]:
        """Compute semantic similarity to each intent using spaCy."""
        if not _SPACY_AVAILABLE or not _nlp:
            return {"data": 0.0, "doc": 0.0, "web": 0.0}
        
        query_doc = _nlp(query.lower())
        
        # Compute max similarity to each intent category
        data_sim = max((query_doc.similarity(doc) for doc in self._data_docs), default=0.0) if self._data_docs else 0.0
        doc_sim = max((query_doc.similarity(doc) for doc in self._doc_docs), default=0.0) if self._doc_docs else 0.0
        web_sim = max((query_doc.similarity(doc) for doc in self._web_docs), default=0.0) if self._web_docs else 0.0
        
        return {"data": data_sim, "doc": doc_sim, "web": web_sim}

    def route(self, query: str, client_context: Optional[str] = None, has_loaded_data: bool = None) -> Dict[str, Any]:
        """
        Route query to appropriate track using semantic understanding.
        
        Args:
            query: User's question
            client_context: Optional context about available data
            has_loaded_data: Override for whether data is loaded
            
        Returns:
            Dict with track, confidence, and reasoning
        """
        # Use provided flag or instance flag
        data_loaded = has_loaded_data if has_loaded_data is not None else self._has_loaded_data
        
        # Check cache
        cache_key = f"{query.lower().strip()}:{data_loaded}"
        if cache_key in self._route_cache:
            cached = self._route_cache[cache_key]
            return {"track": cached, "confidence": 0.95, "method": "cache"}

        # Method 1: Semantic similarity with spaCy
        similarities = self._compute_intent_similarity(query)
        logger.debug(f"Intent similarities: {similarities}")
        
        # Apply data boost when data is loaded
        if data_loaded:
            similarities["data"] += 0.2  # Significant boost
        
        # Determine track from similarities
        max_sim = max(similarities.values())
        
        if max_sim > 0.5:  # Reasonable similarity threshold
            if similarities["data"] >= max_sim:
                track = TRACK_DATA
                confidence = min(0.6 + similarities["data"] * 0.3, 0.95)
                method = "spacy_similarity"
            elif similarities["doc"] >= max_sim:
                track = TRACK_DOC
                confidence = min(0.6 + similarities["doc"] * 0.3, 0.95)
                method = "spacy_similarity"
            elif similarities["web"] >= max_sim:
                track = TRACK_WEB
                confidence = min(0.6 + similarities["web"] * 0.3, 0.95)
                method = "spacy_similarity"
            else:
                track = TRACK_DATA if data_loaded else TRACK_DOC
                confidence = 0.5
                method = "default"
        else:
            # Low similarity - use LLM if available
            if self._llm:
                context = client_context or self._loaded_datasets_info or ""
                if data_loaded and not context:
                    context = "User has data loaded and ready for analysis."
                return self._route_with_llm(query, context)
            
            # Default based on data availability
            track = TRACK_DATA if data_loaded else TRACK_DOC
            confidence = 0.5
            method = "default_no_match"

        # Cache result
        self._route_cache[cache_key] = track

        return {
            "track": track,
            "confidence": confidence,
            "method": method,
            "similarities": similarities
        }

    def _route_with_llm(self, query: str, client_context: Optional[str] = None) -> Dict[str, Any]:
        """Use LLM for routing when semantic similarity is low."""
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
                # Default to data if we have data loaded
                track = TRACK_DATA if self._has_loaded_data else TRACK_DOC

            cache_key = f"{query.lower().strip()}:{self._has_loaded_data}"
            self._route_cache[cache_key] = track
            
            return {
                "track": track,
                "confidence": 0.8,
                "method": "llm",
                "raw_response": response[:100]
            }

        except Exception as e:
            logger.error(f"LLM routing failed: {e}")
            return {
                "track": TRACK_DATA if self._has_loaded_data else TRACK_DOC,
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
