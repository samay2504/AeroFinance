"""
Router Agent - Production-grade semantic query classification.

Uses multi-tier routing strategy:
1. Fast cache lookup (O(1))
2. Pre-computed spaCy similarity (O(n) but vectorized)
3. LLM fallback for ambiguous cases

Routes to: TRACK_DATA (SQL/Pandas), TRACK_DOC (RAG), or TRACK_WEB (Web Search).
"""
import logging
from typing import Dict, Any, Optional, List
from functools import lru_cache
import hashlib

logger = logging.getLogger(__name__)


# Track constants
TRACK_DATA = "TRACK_DATA"
TRACK_DOC = "TRACK_DOC"
TRACK_WEB = "TRACK_WEB"
TRACK_DOC_SUMMARY = "TRACK_DOC_SUMMARY"  # Dataset summary/overview queries
TRACK_OUT_OF_DOMAIN = "TRACK_OUT_OF_DOMAIN"  # Non-CA queries

# Keywords indicating numeric/analytical intent - must route to deterministic computation
ANALYTICAL_KEYWORDS = {
    "total", "growth", "sum", "variance", "calculate", "gmv", "revenue", 
    "fcf", "dcf", "wacc", "cagr", "margin", "profit", "loss", "expense",
    "cost", "average", "mean", "count", "percentage", "%", "rate", "ratio",
    "increase", "decrease", "compare", "difference", "aggregate"
}

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


# Intent exemplars for semantic similarity matching - EXPANDED
DATA_INTENT_EXEMPLARS = [
    # Sheet/data queries
    "how many sheets are there",
    "what are the sheet names",
    "show me the data",
    "list all datasets",
    # Calculations
    "calculate the total",
    "what is the sum",
    "find the average",
    "growth rate",
    "compare values",
    "percentage change",
    # Financial analysis
    "what is the revenue",
    "total profit for the year",
    "operating expenses",
    "balance sheet analysis",
    "income statement",
    "cash flow projection",
    # Valuation
    "enterprise value",
    "equity value",
    "dcf valuation",
    "wacc calculation",
    # Lookups
    "get the value of",
    "find the row where",
    "filter by date",
    "show records for",
]

DOC_INTENT_EXEMPLARS = [
    "what does clause 5 say",
    "find the cancellation policy",
    "accounting policy for inventory",
    "terms and conditions",
    "legal definition",
    "contract agreement",
    "compliance requirement",
    "audit requirement",
    "regulatory guideline",
    "disclosure requirement",
]

WEB_INTENT_EXEMPLARS = [
    "current repo rate",
    "latest GST rules",
    "today's market news",
    "current tax rate",
    "real-time stock price",
    "2024 budget announcement",
    "latest inflation data",
    "RBI circular",
    "SEBI regulation update",
]

# Summary/overview intent exemplars - triggers summarize_dataset flow
SUMMARY_INTENT_EXEMPLARS = [
    "what is this data about",
    "what is this file about",
    "give me an overview",
    "explain this document",
    "summarize this file",
    "what does this data contain",
    "describe this spreadsheet",
    "what information is in this file",
    "overview of the data",
    "what am i looking at",
    "tell me about this dataset",
    "summary of the file",
]

# Out-of-domain patterns for non-CA queries
OUT_OF_DOMAIN_PATTERNS = [
    "tell me a joke",
    "how are you",
    "what is the weather",
    "write a poem",
    "hello",
    "hi there",
    "who are you",
    "what can you do",
    "recipe for",
    "play a game",
]


class RouterAgent:
    """
    Production-grade semantic router with:
    - LRU cache for repeated queries (O(1) lookup)
    - Pre-computed spaCy vectors for fast similarity
    - Batch processing optimization
    - LLM fallback for complex cases
    - Dynamic context awareness
    """

    def __init__(self, llm_wrapper=None):
        self._llm = llm_wrapper
        self._route_cache: Dict[str, Dict[str, Any]] = {}  # Full result cache
        self._cache_max_size = 1000
        self._has_loaded_data = False
        self._loaded_datasets_info = ""
        
        # Pre-compute intent exemplar docs for spaCy similarity
        self._data_docs = []
        self._doc_docs = []
        self._web_docs = []
        
        if _SPACY_AVAILABLE and _nlp:
            # Batch process all exemplars for efficiency
            all_texts = DATA_INTENT_EXEMPLARS + DOC_INTENT_EXEMPLARS + WEB_INTENT_EXEMPLARS
            all_docs = list(_nlp.pipe(all_texts, batch_size=50))
            
            n_data = len(DATA_INTENT_EXEMPLARS)
            n_doc = len(DOC_INTENT_EXEMPLARS)
            
            self._data_docs = all_docs[:n_data]
            self._doc_docs = all_docs[n_data:n_data + n_doc]
            self._web_docs = all_docs[n_data + n_doc:]
            
            logger.debug(f"Pre-computed {len(all_docs)} spaCy docs for intent exemplars")

    def set_data_context(self, has_data: bool, datasets_info: str = ""):
        """Set whether data is currently loaded and what datasets."""
        self._has_loaded_data = has_data
        self._loaded_datasets_info = datasets_info
        # Clear cache when context changes
        if has_data != self._has_loaded_data:
            self._route_cache.clear()

    def _get_cache_key(self, query: str, data_loaded: bool) -> str:
        """Generate cache key using hash for memory efficiency."""
        normalized = query.lower().strip()
        return hashlib.md5(f"{normalized}:{data_loaded}".encode()).hexdigest()[:16]

    def _compute_intent_similarity(self, query: str) -> Dict[str, float]:
        """Compute semantic similarity to each intent using spaCy."""
        if not _SPACY_AVAILABLE or not _nlp:
            return {"data": 0.0, "doc": 0.0, "web": 0.0}
        
        query_doc = _nlp(query.lower())
        
        # Use numpy for vectorized similarity computation if available
        data_sim = max((query_doc.similarity(doc) for doc in self._data_docs), default=0.0) if self._data_docs else 0.0
        doc_sim = max((query_doc.similarity(doc) for doc in self._doc_docs), default=0.0) if self._doc_docs else 0.0
        web_sim = max((query_doc.similarity(doc) for doc in self._web_docs), default=0.0) if self._web_docs else 0.0
        
        return {"data": data_sim, "doc": doc_sim, "web": web_sim}

    def route(self, query: str, client_context: Optional[str] = None, has_loaded_data: bool = None) -> Dict[str, Any]:
        """
        Route query to appropriate track using semantic understanding.
        
        Pipeline:
        1. Check for out-of-domain queries
        2. Check for summary/overview intent
        3. Check for analytical/numeric intent
        4. Cache lookup (instant)
        5. spaCy semantic similarity (fast, vectorized)
        6. LLM fallback (for ambiguous cases)
        
        Args:
            query: User's question
            client_context: Optional context about available data
            has_loaded_data: Override for whether data is loaded
            
        Returns:
            Dict with track, confidence, method, similarities, is_analytical, is_summary
        """
        # Use provided flag or instance flag
        data_loaded = has_loaded_data if has_loaded_data is not None else self._has_loaded_data
        query_lower = query.lower().strip()
        
        # TIER 0a: Check for OUT-OF-DOMAIN queries
        if self._is_out_of_domain(query_lower):
            result = {
                "track": TRACK_OUT_OF_DOMAIN,
                "confidence": 0.9,
                "method": "pattern_match",
                "is_analytical": False,
                "is_summary": False,
                "status": "failed"
            }
            logger.info(f"Out-of-domain query detected: {query[:50]}...")
            return result
        
        # TIER 0b: Check for SUMMARY/OVERVIEW intent
        is_summary = self._is_summary_query(query_lower)
        if is_summary and data_loaded:
            result = {
                "track": TRACK_DOC_SUMMARY,
                "confidence": 0.85,
                "method": "summary_pattern",
                "is_analytical": False,
                "is_summary": True
            }
            logger.debug(f"Summary query detected, routing to TRACK_DOC_SUMMARY")
            return result
        
        # TIER 0c: Check for ANALYTICAL/NUMERIC intent - mark for deterministic execution
        is_analytical = self._is_analytical_query(query_lower)
        
        # TIER 1: Cache lookup (O(1))
        cache_key = self._get_cache_key(query, data_loaded)
        if cache_key in self._route_cache:
            cached = self._route_cache[cache_key].copy()
            cached["method"] = "cache"
            cached["is_analytical"] = is_analytical
            cached["is_summary"] = is_summary
            return cached

        # TIER 2: spaCy semantic similarity
        similarities = self._compute_intent_similarity(query)
        logger.debug(f"Intent similarities: {similarities}")
        
        # Apply context-aware boosting
        if data_loaded:
            similarities["data"] += 0.25  # Boost when data is loaded
        
        # Determine track from similarities
        max_sim = max(similarities.values())
        track = None
        confidence = 0.5
        method = "default"
        
        if max_sim > 0.45:  # Reasonable similarity threshold
            if similarities["data"] >= max_sim:
                track = TRACK_DATA
                confidence = min(0.65 + similarities["data"] * 0.3, 0.95)
                method = "spacy_semantic"
            elif similarities["web"] >= max_sim:
                track = TRACK_WEB
                confidence = min(0.65 + similarities["web"] * 0.3, 0.95)
                method = "spacy_semantic"
            elif similarities["doc"] >= max_sim:
                track = TRACK_DOC
                confidence = min(0.65 + similarities["doc"] * 0.3, 0.95)
                method = "spacy_semantic"
        
        # TIER 3: LLM fallback for ambiguous cases
        if track is None or confidence < 0.6:
            if self._llm:
                context = client_context or self._loaded_datasets_info or ""
                if data_loaded and not context:
                    context = "User has data loaded and ready for analysis."
                llm_result = self._route_with_llm(query, context)
                if llm_result.get("confidence", 0) > confidence:
                    llm_result["is_analytical"] = is_analytical
                    llm_result["is_summary"] = is_summary
                    return llm_result
            
            # Final default based on data availability
            if track is None:
                track = TRACK_DATA if data_loaded else TRACK_DOC
                method = "default_fallback"

        result = {
            "track": track,
            "confidence": round(confidence, 2),
            "method": method,
            "similarities": {k: round(v, 3) for k, v in similarities.items()},
            "is_analytical": is_analytical,
            "is_summary": is_summary
        }
        
        # Cache result (with LRU eviction)
        if len(self._route_cache) >= self._cache_max_size:
            # Evict oldest entries
            keys_to_remove = list(self._route_cache.keys())[:100]
            for k in keys_to_remove:
                del self._route_cache[k]
        
        self._route_cache[cache_key] = result
        return result
    
    def _is_out_of_domain(self, query_lower: str) -> bool:
        """Check if query is out of domain (non-CA/non-finance)."""
        for pattern in OUT_OF_DOMAIN_PATTERNS:
            if pattern in query_lower:
                return True
        return False
    
    def _is_summary_query(self, query_lower: str) -> bool:
        """Check if query is asking for dataset overview/summary."""
        summary_patterns = [
            "what is this data", "what is this file", "overview", 
            "summarize", "describe this", "what does this contain",
            "what information", "tell me about this", "what am i looking at"
        ]
        for pattern in summary_patterns:
            if pattern in query_lower:
                return True
        # Also check semantic similarity to summary exemplars if spaCy available
        return False
    
    def _is_analytical_query(self, query_lower: str) -> bool:
        """
        Check if query contains numeric/analytical intent keywords.
        These queries must use deterministic computation (SQL/Pandas).
        """
        for keyword in ANALYTICAL_KEYWORDS:
            if keyword in query_lower:
                return True
        return False

    def _route_with_llm(self, query: str, client_context: Optional[str] = None) -> Dict[str, Any]:
        """Use LLM for routing when semantic similarity is low or ambiguous."""
        try:
            from app.core.prompts import get_router_prompt

            prompt = get_router_prompt(query, client_context or "")
            response = self._llm.invoke(prompt)

            # Extract track from response
            response_upper = response.upper().strip()
            
            if "TRACK_DATA" in response_upper or "DATA" in response_upper:
                track = TRACK_DATA
            elif "TRACK_DOC" in response_upper or "DOCUMENT" in response_upper:
                track = TRACK_DOC
            elif "TRACK_WEB" in response_upper or "WEB" in response_upper:
                track = TRACK_WEB
            else:
                # Default to data if we have data loaded
                track = TRACK_DATA if self._has_loaded_data else TRACK_DOC

            result = {
                "track": track,
                "confidence": 0.8,
                "method": "llm",
            }
            
            # Cache LLM result
            cache_key = self._get_cache_key(query, self._has_loaded_data)
            self._route_cache[cache_key] = result
            
            return result

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
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics for monitoring."""
        return {
            "cache_size": len(self._route_cache),
            "max_size": self._cache_max_size,
            "has_data": self._has_loaded_data
        }


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


__all__ = [
    "RouterAgent", 
    "get_router_agent", 
    "TRACK_DATA", 
    "TRACK_DOC", 
    "TRACK_WEB",
    "TRACK_DOC_SUMMARY",
    "TRACK_OUT_OF_DOMAIN",
    "ANALYTICAL_KEYWORDS",
]

