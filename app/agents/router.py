"""
Router Agent - Production-grade semantic query classification.

Uses multi-tier routing strategy:
1. Fast cache lookup (O(1))
2. Pre-computed spaCy similarity (O(n) but vectorized)
3. LLM fallback for ambiguous cases with structured intent classification

Routes to: TRACK_DATA (SQL/Pandas), TRACK_DOC (RAG), TRACK_WEB (Web Search),
           TRACK_DOC_SUMMARY (Dataset Overview), TRACK_OUT_OF_DOMAIN (Unrelated)
"""
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import hashlib
import json

try:
    from app.core.dll_fix import apply_dll_fix
    apply_dll_fix()
except ImportError:
    pass

logger = logging.getLogger(__name__)

# ============================================================================
# TRACK DEFINITIONS - Unified Track System
# ============================================================================
TRACK_DATA = "TRACK_DATA"
TRACK_DOC = "TRACK_DOC"
TRACK_WEB = "TRACK_WEB"
TRACK_DOC_SUMMARY = "TRACK_DOC_SUMMARY"
TRACK_OUT_OF_DOMAIN = "TRACK_OUT_OF_DOMAIN"


@dataclass
class TrackConfig:
    """Configuration for each routing track."""
    name: str
    description: str
    priority: int  # Lower = higher priority (for tie-breaking)
    base_threshold: float  # Minimum similarity to consider
    boost_with_data: float  # Boost when user has data loaded
    requires_data: bool  # Whether this track needs data to be loaded


# Unified track configuration - all tracks defined consistently
TRACK_CONFIGS = {
    TRACK_DATA: TrackConfig(
        name="DATA",
        description="Query/analyze loaded data (Excel, CSV, calculations, entity search)",
        priority=1,
        base_threshold=0.55,
        boost_with_data=0.30,
        requires_data=True
    ),
    TRACK_DOC_SUMMARY: TrackConfig(
        name="SUMMARY",
        description="Get overview/summary/description of loaded data",
        priority=2,
        base_threshold=0.70,
        boost_with_data=0.15,
        requires_data=True
    ),
    TRACK_DOC: TrackConfig(
        name="DOCUMENT",
        description="Search unstructured documents (PDFs, contracts, policies)",
        priority=3,
        base_threshold=0.60,
        boost_with_data=-0.10,
        requires_data=False
    ),
    TRACK_WEB: TrackConfig(
        name="WEB",
        description="Real-time/current information from internet",
        priority=4,
        base_threshold=0.60,
        boost_with_data=-0.15,
        requires_data=False
    ),
    TRACK_OUT_OF_DOMAIN: TrackConfig(
        name="OUT_OF_DOMAIN",
        description="Query unrelated to financial/CA domain",
        priority=5,
        base_threshold=0.90,
        boost_with_data=-0.50,  # Strong negative boost - don't OOD when data loaded
        requires_data=False
    ),
}


# ============================================================================
# INTENT CLASSIFICATION - LLM-based structured classification
# ============================================================================
class UserIntent(Enum):
    """Semantic user intent types for intelligent routing."""
    CALCULATION = "calculation"      # Sum, average, total, compute
    ENTITY_SEARCH = "entity_search"  # Find mentions of X, is there any Y
    DATA_QUERY = "data_query"        # Filter, show records, list values
    SUMMARY = "summary"              # Describe, overview, what is this about
    DOCUMENT_SEARCH = "document_search"  # Find in documents, clause, policy
    REAL_TIME = "real_time"          # Current, today, latest, live
    OUT_OF_DOMAIN = "out_of_domain"  # Jokes, recipes, unrelated


# Intent to Track mapping
INTENT_TO_TRACK = {
    UserIntent.CALCULATION: TRACK_DATA,
    UserIntent.ENTITY_SEARCH: TRACK_DATA,
    UserIntent.DATA_QUERY: TRACK_DATA,
    UserIntent.SUMMARY: TRACK_DOC_SUMMARY,
    UserIntent.DOCUMENT_SEARCH: TRACK_DOC,
    UserIntent.REAL_TIME: TRACK_WEB,
    UserIntent.OUT_OF_DOMAIN: TRACK_OUT_OF_DOMAIN,
}

_nlp = None
_SPACY_AVAILABLE = False

try:
    import spacy
    try:
        _nlp = spacy.load("en_core_web_md")
        _SPACY_AVAILABLE = True
        logger.info("Loaded spaCy model: en_core_web_md")
    except OSError:
        try:
            _nlp = spacy.load("en_core_web_sm")
            _SPACY_AVAILABLE = True
            logger.info("Loaded spaCy model: en_core_web_sm")
        except OSError:
            logger.warning("No spaCy model found. Run: python -m spacy download en_core_web_md")
except ImportError:
    logger.info("spaCy not available.")


DATA_INTENT_EXEMPLARS = [
    "how many sheets are there", "what are the sheet names", "show me the data", "list all datasets",
    "calculate the total", "what is the sum", "find the average", "growth rate", "compare values",
    "percentage change", "what is the revenue", "total profit for the year", "operating expenses",
    "balance sheet analysis", "income statement", "cash flow projection", "enterprise value",
    "equity value", "dcf valuation", "wacc calculation", "get the value of", "find the row where",
    "filter by date", "show records for", "sum of values", "total calculation", "metric analysis",
    "calculate from our data", "analyze our dataset", "compute from uploaded file",
    # Content/entity search queries
    "is there any mention of", "find mentions of", "search for", "look for", "does it mention",
    "is there a reference to", "what companies are mentioned", "find company name", "company names in data",
    "is there any company", "what entities are in", "extract names from", "list all names",
    "find all occurrences of", "search the data for", "locate in the file", "find in dataset",
    # Metadata queries about uploaded data
    "what columns are there", "what fields exist", "what information is stored", "what data is available",
    "what categories are in", "what types of data", "what is included in", "what does the file contain",
    "is there data about", "do we have information on", "check if there is", "verify if exists"
]

DOC_INTENT_EXEMPLARS = [
    "what does clause 5 say", "find the cancellation policy", "accounting policy for inventory",
    "terms and conditions", "legal definition", "contract agreement", "compliance requirement",
    "audit requirement", "regulatory guideline", "disclosure requirement", "policy statement",
    "clause interpretation", "legal requirement", "contractual term", "document section",
    "governance policy", "risk disclosure", "regulatory compliance", "audit finding"
]

WEB_INTENT_EXEMPLARS = [
    "current repo rate", "latest GST rules", "today's market news", "current tax rate",
    "real-time stock price", "2024 budget announcement", "latest inflation data", "RBI circular",
    "SEBI regulation update", "market data", "real-time information", "current news",
    "latest announcement", "market quote", "regulatory update", "economic data", "industry news",
    "today's gold price", "this week's market update", "current interest rate", "live forex rates",
    "breaking market news", "latest economic announcement", "today's commodity prices",
    "current currency exchange", "real-time oil price", "today's stock index", "latest budget update",
    "recent policy change", "current regulatory news", "today's market analysis",
    "this week's economic data", "latest central bank decision", "real-time market quote",
    "current inflation report", "today's unemployment data", "latest gdp announcement",
    "breaking news headlines", "today's market performance", "current stock performance",
    "latest market trends", "real-time forex trading", "today's precious metals price",
    "current cryptocurrency price", "latest market update", "real-time weather forecast"
]

SUMMARY_INTENT_EXEMPLARS = [
    "give me a summary", "give me the summary", "give me summary", "show me a summary",
    "show me the summary", "provide a summary", "provide summary", "what is the summary",
    "tell me the summary", "give me an overview", "give me the overview", "show me an overview",
    "provide an overview", "what is the overview", "describe this data", "describe the data",
    "describe this file", "describe the file", "explain this data", "explain the data",
    "explain this file", "explain the file", "what is this data about", "what is the data about",
    "what is this file about", "what is the file about", "what is this about",
    "tell me about this data", "tell me about the data", "tell me about this file",
    "tell me about the file", "summarize this", "summarize the data", "summarize this data",
    "summarize the file", "summarize this file", "what information is here",
    "what information is in this", "what does this contain", "what does this data contain",
    "what am i looking at", "what are we looking at", "describe this dataset",
    "describe our dataset", "tell me about our data", "what is in this data",
    # Content discovery queries that should also trigger summary/exploration
    "what kind of data is this", "what type of file is this", "what is the structure",
    "what are the main topics", "what subjects are covered", "key information in this"
]

# Content/Entity SEARCH queries - These ask to FIND/SEARCH for specific content, NOT summarize
# These should route to TRACK_DATA for entity extraction, not TRACK_DOC_SUMMARY
CONTENT_SEARCH_EXEMPLARS = [
    "is there any mention of", "is there mention of", "does it mention", "does this mention",
    "any mention of", "find mentions of", "search for mentions of", "look for mentions of",
    "is there a reference to", "any reference to", "find references to",
    "is there any company", "is there a company name", "any company name", "company names mentioned",
    "are there any names", "find all names", "extract names", "list all names",
    "is there any person", "find person names", "who is mentioned", "names in the data",
    "does it contain", "does this contain", "check if there is", "verify if there is",
    "search the data for", "look in the data for", "find in the data", "locate in the data",
    "is there any instance of", "any occurrence of", "find occurrences of",
    "what companies are mentioned", "which companies appear", "list companies in",
    "extract entities from", "find entities in", "entity extraction",
    "are there any keywords", "find keywords", "extract keywords"
]

OUT_OF_DOMAIN_EXEMPLARS = [
    "tell me a joke please", "write poetry for me", "recipe for chicken pasta", 
    "how to play chess game", "tell me a funny story", "write creative fiction novel",
    "homework assignment help", "medical diagnosis advice", "legal lawsuit guidance",
    "how to cook dinner meal", "music composition tutorial", "gardening tips plants",
    "pet training advice dogs", "travel vacation planning", "movie recommendation",
    "book review suggestion", "sports betting predictions", "cryptocurrency tips"
]

ANALYTICAL_INTENT_EXEMPLARS = [
    "calculate the total", "what is the sum", "find the average", "growth rate",
    "compare values", "percentage change", "what is the revenue", "total profit",
    "operating expenses", "profit margin", "cost analysis", "revenue breakdown",
    "expense report", "financial summary", "balance sheet", "income statement",
    "dcf valuation", "wacc calculation", "equity value", "enterprise value",
    "find the maximum", "find the minimum", "calculate variance", "compute standard deviation",
    "growth percentage", "year over year", "quarterly change", "trend analysis",
    "what is the total", "how much revenue", "how much profit", "expense details",
    "cost breakdown", "sales analysis", "margin calculation", "ratio analysis"
]


# ============================================================================
# KEYWORD-BASED INTENT DETECTION (Fast path)
# ============================================================================
INTENT_KEYWORDS = {
    UserIntent.CALCULATION: [
        'calculate', 'compute', 'sum', 'total', 'average', 'mean', 'count',
        'percentage', 'growth', 'rate', 'ratio', 'margin', 'variance'
    ],
    UserIntent.ENTITY_SEARCH: [
        'is there any mention', 'any mention of', 'does it mention', 'does this mention',
        'is there a reference', 'find mentions', 'is there any company', 'company name',
        'are there any', 'does it contain', 'does this contain', 'check if there is',
        'search for', 'look for', 'find in', 'extract'
    ],
    UserIntent.DATA_QUERY: [
        'show me', 'list', 'filter', 'get the value', 'find the row',
        'what are the', 'how many', 'which', 'where is'
    ],
    UserIntent.SUMMARY: [
        'summary', 'summarize', 'overview', 'describe', 'explain', 'about',
        'tell me about', 'what is this', 'what does this'
    ],
    UserIntent.DOCUMENT_SEARCH: [
        'clause', 'policy', 'contract', 'agreement', 'regulation', 'compliance',
        'legal', 'audit requirement', 'document section'
    ],
    UserIntent.REAL_TIME: [
        'current', 'today', 'now', 'live', 'real-time', 'latest', 'recent',
        'this week', 'breaking', 'news', 'market quote'
    ],
    UserIntent.OUT_OF_DOMAIN: [
        'joke', 'poem', 'recipe', 'cook', 'game', 'story', 'fiction',
        'medical advice', 'vacation', 'movie', 'music'
    ]
}


class RouterAgent:
    """
    Intelligent query router with multi-tier classification.
    
    Routing Strategy:
    1. Cache lookup (O(1)) - for repeated queries
    2. Keyword-based fast path - for obvious intents
    3. Semantic similarity (spaCy) - for nuanced classification
    4. LLM fallback - for truly ambiguous cases with structured output
    
    All tracks are scored uniformly and the highest score wins.
    """

    def __init__(self, llm_wrapper=None):
        self._llm = llm_wrapper
        self._route_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_max_size = 1000
        self._has_loaded_data = False
        self._loaded_datasets_info = ""
        
        # Semantic similarity vectors (pre-computed for performance)
        self._data_docs = []
        self._doc_docs = []
        self._web_docs = []
        self._summary_docs = []
        self._out_of_domain_docs = []
        self._analytical_docs = []
        self._content_search_docs = []  # New: for entity/content search queries
        
        self._real_time_keywords = {
            'current', 'today', 'now', 'live', 'real-time', 'latest', 'recent',
            'this week', 'this month', 'breaking', 'new', 'update', 'announcement',
            'quote', 'index', 'news', 'forecast'
        }
        
        self._historical_keywords = {
            'historical', 'past', 'year ago', 'last year', 'previous', 'old',
            'archive', 'backtesting', 'analysis', 'trend', '5 years', '10 years',
            'last 5 years', 'last 10 years'
        }
        
        if _SPACY_AVAILABLE and _nlp:
            all_texts = (
                DATA_INTENT_EXEMPLARS + 
                DOC_INTENT_EXEMPLARS + 
                WEB_INTENT_EXEMPLARS + 
                SUMMARY_INTENT_EXEMPLARS +
                OUT_OF_DOMAIN_EXEMPLARS +
                ANALYTICAL_INTENT_EXEMPLARS +
                CONTENT_SEARCH_EXEMPLARS  # New exemplars for content search
            )
            all_docs = list(_nlp.pipe(all_texts, batch_size=50))
            
            n_data = len(DATA_INTENT_EXEMPLARS)
            n_doc = len(DOC_INTENT_EXEMPLARS)
            n_web = len(WEB_INTENT_EXEMPLARS)
            n_summary = len(SUMMARY_INTENT_EXEMPLARS)
            n_ood = len(OUT_OF_DOMAIN_EXEMPLARS)
            n_analytical = len(ANALYTICAL_INTENT_EXEMPLARS)
            n_content_search = len(CONTENT_SEARCH_EXEMPLARS)
            
            self._data_docs = all_docs[:n_data]
            self._doc_docs = all_docs[n_data:n_data + n_doc]
            self._web_docs = all_docs[n_data + n_doc:n_data + n_doc + n_web]
            self._summary_docs = all_docs[n_data + n_doc + n_web:n_data + n_doc + n_web + n_summary]
            self._out_of_domain_docs = all_docs[n_data + n_doc + n_web + n_summary:n_data + n_doc + n_web + n_summary + n_ood]
            self._analytical_docs = all_docs[n_data + n_doc + n_web + n_summary + n_ood:n_data + n_doc + n_web + n_summary + n_ood + n_analytical]
            self._content_search_docs = all_docs[n_data + n_doc + n_web + n_summary + n_ood + n_analytical:]
            
            logger.debug(f"Pre-computed {len(all_docs)} spaCy docs for intent classification (incl. {n_content_search} content search)")

    def set_data_context(self, has_data: bool, datasets_info: str = ""):
        self._has_loaded_data = has_data
        self._loaded_datasets_info = datasets_info
        if has_data != self._has_loaded_data:
            self._route_cache.clear()

    def _get_cache_key(self, query: str, data_loaded: bool) -> str:
        normalized = query.lower().strip()
        return hashlib.md5(f"{normalized}:{data_loaded}".encode()).hexdigest()[:16]

    def _is_real_time_query(self, query: str) -> bool:
        query_lower = query.lower()
        query_doc = _nlp(query_lower) if _SPACY_AVAILABLE and _nlp else None
        
        real_time_count = sum(1 for keyword in self._real_time_keywords if keyword in query_lower)
        historical_count = sum(1 for keyword in self._historical_keywords if keyword in query_lower)
        
        if historical_count > 0:
            return False
        
        summary_keywords = {'summary', 'summarize', 'overview', 'describe', 'explain', 'tell me about'}
        if any(kw in query_lower for kw in summary_keywords):
            return False
        
        data_keywords = {'our data', 'your data', 'my data', 'the data', 'dataset', 'uploaded', 'loaded', 'file'}
        if any(kw in query_lower for kw in data_keywords) and 'current' not in query_lower and 'today' not in query_lower:
            return False
        
        if 'show' in query_lower or 'give' in query_lower or 'tell' in query_lower or 'get' in query_lower:
            if 'current' not in query_lower and 'today' not in query_lower and 'latest' not in query_lower:
                return False
        
        if real_time_count >= 1:
            return True
        
        if 'today' in query_lower:
            return True
        
        return False


    def _get_dominant_category(self, similarities: Dict[str, float]) -> Optional[str]:
        categories = ['data', 'doc', 'web', 'summary']
        valid_sims = {cat: sim for cat, sim in similarities.items() if cat in categories}
        
        if not valid_sims:
            return None
        
        max_category = max(valid_sims, key=valid_sims.get)
        max_sim = valid_sims[max_category]
        
        return max_category if max_sim > 0.45 else None

    def _compute_semantic_similarity(self, query: str) -> Dict[str, float]:
        if not _SPACY_AVAILABLE or not _nlp:
            return {"data": 0.0, "doc": 0.0, "web": 0.0, "summary": 0.0, "ood": 0.0, "analytical": 0.0, "content_search": 0.0}
        
        query_doc = _nlp(query.lower())
        
        return {
            "data": max((query_doc.similarity(doc) for doc in self._data_docs), default=0.0) if self._data_docs else 0.0,
            "doc": max((query_doc.similarity(doc) for doc in self._doc_docs), default=0.0) if self._doc_docs else 0.0,
            "web": max((query_doc.similarity(doc) for doc in self._web_docs), default=0.0) if self._web_docs else 0.0,
            "summary": max((query_doc.similarity(doc) for doc in self._summary_docs), default=0.0) if self._summary_docs else 0.0,
            "ood": max((query_doc.similarity(doc) for doc in self._out_of_domain_docs), default=0.0) if self._out_of_domain_docs else 0.0,
            "analytical": max((query_doc.similarity(doc) for doc in self._analytical_docs), default=0.0) if self._analytical_docs else 0.0,
            "content_search": max((query_doc.similarity(doc) for doc in self._content_search_docs), default=0.0) if self._content_search_docs else 0.0
        }

    def route(self, query: str, client_context: Optional[str] = None, has_loaded_data: bool = None) -> Dict[str, Any]:
        data_loaded = has_loaded_data if has_loaded_data is not None else self._has_loaded_data
        query_lower = query.lower().strip()
        
        cache_key = self._get_cache_key(query, data_loaded)
        if cache_key in self._route_cache:
            return self._route_cache[cache_key].copy()
        
        similarities = self._compute_semantic_similarity(query)
        is_real_time = self._is_real_time_query(query)
        
        analytical_metrics = [
            'revenue', 'profit', 'loss', 'expense', 'cost', 'margin', 
            'growth rate', 'growth', 'rate', 'wacc', 'ebitda', 'fcf', 'fcff', 'dcf',
            'equity', 'debt', 'ratio', 'value', 'worth', 'percentage',
            'total', 'average', 'median', 'variance'
        ]
        analytical_actions = ['calculate', 'compute', 'find', 'determine', 'derive']
        
        has_metric = any(metric in query_lower for metric in analytical_metrics)
        has_summary_keyword = " summary" in query_lower or "summary " in query_lower or "summarize" in query_lower or " overview" in query_lower or "overview " in query_lower
        
        if is_real_time and similarities["web"] > 0.65:
            result = {
                "track": TRACK_WEB,
                "confidence": round(min(0.75 + similarities["web"] * 0.2, 0.98), 2),
                "method": "semantic_web_real_time",
                "similarities": {k: round(v, 3) for k, v in similarities.items()},
                "is_analytical": False,
                "is_summary": False,
                "is_real_time": True,
                "is_content_search": False
            }
            if len(self._route_cache) >= self._cache_max_size:
                keys_to_remove = list(self._route_cache.keys())[:100]
                for k in keys_to_remove:
                    del self._route_cache[k]
            self._route_cache[cache_key] = result
            return result
        
        # ==================================================================
        # CONTENT SEARCH DETECTION (Must check BEFORE summary)
        # "Is there any mention of X?" is a SEARCH query, not a summary query
        # BUT: Explicit summary requests should NOT be treated as content search
        # ==================================================================
        
        # First check if this is an explicit SUMMARY request (should NOT be content search)
        explicit_summary_keywords = ['summary', 'summarize', 'overview', 'describe', 'explain', 'about']
        is_explicit_summary = any(kw in query_lower for kw in explicit_summary_keywords)
        
        content_search_indicators = [
            'is there any mention', 'any mention of', 'does it mention', 'does this mention',
            'is there a reference', 'any reference to', 'find mentions', 'search for mentions',
            'is there any company', 'company name', 'company names', 'names mentioned',
            'are there any', 'does it contain', 'does this contain',
            'check if there is', 'verify if', 'look for', 'search for', 'find in'
        ]
        # Only mark as content search if NOT an explicit summary request
        is_content_search = (
            not is_explicit_summary and 
            any(indicator in query_lower for indicator in content_search_indicators)
        )
        
        # Also check semantic similarity to content search exemplars (only if not explicit summary)
        if not is_content_search and not is_explicit_summary and similarities.get("content_search", 0) > 0.75:
            is_content_search = True
        
        # Content search queries should route to DATA track (entity extraction), not summary
        if is_content_search and data_loaded:
            result = {
                "track": TRACK_DATA,
                "confidence": round(min(0.85 + similarities.get("content_search", 0.5) * 0.1, 0.95), 2),
                "method": "content_search_data",
                "similarities": {k: round(v, 3) for k, v in similarities.items()},
                "is_analytical": False,
                "is_summary": False,
                "is_real_time": False,
                "is_content_search": True
            }
            if len(self._route_cache) >= self._cache_max_size:
                keys_to_remove = list(self._route_cache.keys())[:100]
                for k in keys_to_remove:
                    del self._route_cache[k]
            self._route_cache[cache_key] = result
            logger.info(f"Content search detected: '{query[:50]}...' -> TRACK_DATA")
            return result
        
        # ==================================================================
        # SUMMARY DETECTION (only if NOT content search)
        # ==================================================================
        is_summary_eligible = (
            similarities["summary"] > 0.70 and 
            data_loaded and 
            not is_real_time and
            not is_content_search and  # Don't treat content search as summary
            (is_explicit_summary or similarities["summary"] > 0.85)  # Use explicit_summary flag
        )
        
        if is_summary_eligible:
            result = {
                "track": TRACK_DOC_SUMMARY,
                "confidence": round(min(0.8 + similarities["summary"] * 0.15, 0.98), 2),
                "method": "semantic_summary",
                "similarities": {k: round(v, 3) for k, v in similarities.items()},
                "is_analytical": False,
                "is_summary": True,
                "is_real_time": False,
                "is_content_search": False
            }
            if len(self._route_cache) >= self._cache_max_size:
                keys_to_remove = list(self._route_cache.keys())[:100]
                for k in keys_to_remove:
                    del self._route_cache[k]
            self._route_cache[cache_key] = result
            return result
        
        # Smart OOD detection: When data is loaded, be MORE lenient - user is likely asking about their data
        # Only route to OOD if the query is CLEARLY unrelated (high OOD score AND low data/summary scores)
        data_related_keywords = {'data', 'file', 'sheet', 'column', 'row', 'mention', 'name', 'company', 
                                  'information', 'contain', 'find', 'search', 'look', 'check', 'any', 'there'}
        query_words = set(query_lower.split())
        has_data_context = bool(query_words & data_related_keywords)
        
        # Stricter OOD: must be very high OOD AND very low everything else AND no data-related keywords
        is_truly_ood = (
            similarities["ood"] > 0.92 and  # Very high OOD threshold
            similarities["data"] < 0.40 and
            similarities["summary"] < 0.40 and
            similarities["analytical"] < 0.40 and
            similarities.get("content_search", 0) < 0.40 and  # Also check content search
            not has_data_context and  # No data-related words in query
            not data_loaded  # Don't mark as OOD if user has data loaded
        )
        
        if is_truly_ood:
            result = {
                "track": TRACK_OUT_OF_DOMAIN,
                "confidence": round(min(0.8 + similarities["ood"] * 0.15, 0.98), 2),
                "method": "semantic_ood",
                "similarities": {k: round(v, 3) for k, v in similarities.items()},
                "is_analytical": False,
                "is_summary": False,
                "is_real_time": False,
                "is_content_search": False
            }
            if len(self._route_cache) >= self._cache_max_size:
                keys_to_remove = list(self._route_cache.keys())[:100]
                for k in keys_to_remove:
                    del self._route_cache[k]
            self._route_cache[cache_key] = result
            return result
        
        is_analytical = similarities["analytical"] > 0.60
        
        has_historical = any(kw in query_lower for kw in self._historical_keywords)
        
        # Strong boost for DATA track when user has data loaded - they're likely asking about their data
        if data_loaded and not is_real_time and not has_summary_keyword:
            similarities["data"] += 0.35  # Stronger boost
        
        if has_historical:
            similarities["data"] += 0.30
            similarities["web"] -= 0.15
        
        if is_real_time:
            similarities["web"] += 0.25
            similarities["data"] -= 0.15
        
        max_sim = max(similarities["data"], similarities["doc"], similarities["web"])
        track = None
        confidence = 0.5
        method = "default"
        
        if max_sim > 0.45:
            if is_real_time and similarities["web"] > 0.65:
                track = TRACK_WEB
                confidence = min(0.70 + similarities["web"] * 0.25, 0.95)
                method = "semantic_web_adaptive"
            elif similarities["data"] >= max_sim:
                track = TRACK_DATA
                confidence = min(0.65 + similarities["data"] * 0.3, 0.95)
                method = "semantic_data"
            elif similarities["web"] >= max_sim:
                track = TRACK_WEB
                confidence = min(0.65 + similarities["web"] * 0.3, 0.95)
                method = "semantic_web"
            elif similarities["doc"] >= max_sim:
                track = TRACK_DOC
                confidence = min(0.65 + similarities["doc"] * 0.3, 0.95)
                method = "semantic_doc"
        
        # ======================================================================
        # LLM FALLBACK - For ambiguous cases, use intelligent intent classifier
        # ======================================================================
        if track is None or confidence < 0.6:
            if self._llm:
                context = client_context or self._loaded_datasets_info or ""
                if data_loaded and not context:
                    context = "User has data loaded and ready for analysis."
                
                # Use the new intelligent LLM intent classifier
                llm_result = self._route_with_llm(query, context, has_data=data_loaded)
                if llm_result.get("confidence", 0) > confidence:
                    llm_result["is_analytical"] = is_analytical
                    llm_result["is_summary"] = is_summary_eligible
                    llm_result["is_real_time"] = is_real_time
                    llm_result["is_content_search"] = is_content_search
                    if len(self._route_cache) >= self._cache_max_size:
                        keys_to_remove = list(self._route_cache.keys())[:100]
                        for k in keys_to_remove:
                            del self._route_cache[k]
                    self._route_cache[cache_key] = llm_result
                    return llm_result
            
            if track is None:
                track = TRACK_WEB if is_real_time else (TRACK_DATA if data_loaded else TRACK_DOC)
                method = "semantic_fallback"
        
        result = {
            "track": track,
            "confidence": round(confidence, 2),
            "method": method,
            "similarities": {k: round(v, 3) for k, v in similarities.items()},
            "is_analytical": is_analytical,
            "is_summary": is_summary_eligible,
            "is_real_time": is_real_time,
            "is_content_search": is_content_search
        }
        
        if len(self._route_cache) >= self._cache_max_size:
            keys_to_remove = list(self._route_cache.keys())[:100]
            for k in keys_to_remove:
                del self._route_cache[k]
        
        self._route_cache[cache_key] = result
        return result

    def _classify_intent_with_keywords(self, query: str) -> Optional[UserIntent]:
        """
        Fast-path keyword-based intent detection.
        Returns intent if clear match, None if ambiguous.
        """
        query_lower = query.lower()
        
        # Check each intent's keywords
        intent_scores = {}
        for intent, keywords in INTENT_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in query_lower)
            if score > 0:
                intent_scores[intent] = score
        
        if not intent_scores:
            return None
        
        # Get the intent with highest keyword matches
        best_intent = max(intent_scores, key=intent_scores.get)
        best_score = intent_scores[best_intent]
        
        # Only return if clear winner (2+ keyword matches or only one intent matched)
        if best_score >= 2 or len(intent_scores) == 1:
            return best_intent
        
        return None

    def _route_with_llm(self, query: str, client_context: Optional[str] = None, has_data: bool = False) -> Dict[str, Any]:
        """
        Intelligent LLM-based intent classification with structured output.
        
        Uses a carefully crafted prompt to understand user intent semantically
        and maps it to the appropriate track.
        """
        try:
            # Build the structured intent classification prompt
            prompt = f"""You are an expert intent classifier for an AI Chartered Accountant system.
Your task is to understand the user's TRUE INTENT and classify it into one of these categories.

AVAILABLE INTENT TYPES:
1. CALCULATION - User wants to compute/calculate something from data (sum, total, average, growth, ratio)
   Examples: "What is total revenue?", "Calculate growth rate", "Find the average"

2. ENTITY_SEARCH - User wants to FIND/SEARCH for specific content or check if something exists
   Examples: "Is there any mention of a company name?", "Find mentions of revenue", "Does it contain..."
   KEY: Answers are typically YES/NO or LIST of found items

3. DATA_QUERY - User wants to retrieve, filter, or list data values
   Examples: "Show me all records", "List column names", "Filter by date"

4. SUMMARY - User wants an OVERVIEW or DESCRIPTION of what the data contains
   Examples: "Give me a summary", "What is this data about?", "Describe this file"
   KEY: User wants to UNDERSTAND what they're looking at, not search for specific content

5. DOCUMENT_SEARCH - User wants to find info in documents (PDFs, contracts, policies)
   Examples: "What does clause 5 say?", "Find the cancellation policy"

6. REAL_TIME - User wants CURRENT/LIVE information from the internet
   Examples: "Current repo rate", "Today's stock price", "Latest news"

7. OUT_OF_DOMAIN - Query is completely unrelated to financial/business data
   Examples: "Tell me a joke", "Recipe for pasta", "How to play chess"

CONTEXT:
- User has data loaded: {has_data}
- Additional context: {client_context or 'None'}

USER QUERY: "{query}"

CRITICAL DISTINCTIONS:
- "Is there any mention of X?" = ENTITY_SEARCH (searching for content)
- "What is this about?" = SUMMARY (wanting overview)
- "What is the total X?" = CALCULATION (computing value)

Respond with ONLY a JSON object (no markdown, no explanation):
{{"intent": "CALCULATION|ENTITY_SEARCH|DATA_QUERY|SUMMARY|DOCUMENT_SEARCH|REAL_TIME|OUT_OF_DOMAIN", "confidence": 0.0-1.0, "reasoning": "brief explanation"}}"""

            response = self._llm.invoke(prompt)
            
            # Parse JSON response
            response_text = response.strip()
            # Remove markdown code blocks if present
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
            response_text = response_text.strip()
            
            try:
                result = json.loads(response_text)
            except json.JSONDecodeError:
                # Try to extract JSON from response
                import re
                json_match = re.search(r'\{[^{}]+\}', response_text)
                if json_match:
                    result = json.loads(json_match.group())
                else:
                    raise ValueError("Could not parse LLM response as JSON")
            
            intent_str = result.get("intent", "").upper()
            llm_confidence = float(result.get("confidence", 0.7))
            reasoning = result.get("reasoning", "")
            
            # Map intent string to UserIntent enum
            intent_map = {
                "CALCULATION": UserIntent.CALCULATION,
                "ENTITY_SEARCH": UserIntent.ENTITY_SEARCH,
                "DATA_QUERY": UserIntent.DATA_QUERY,
                "SUMMARY": UserIntent.SUMMARY,
                "DOCUMENT_SEARCH": UserIntent.DOCUMENT_SEARCH,
                "REAL_TIME": UserIntent.REAL_TIME,
                "OUT_OF_DOMAIN": UserIntent.OUT_OF_DOMAIN,
            }
            
            intent = intent_map.get(intent_str, UserIntent.DATA_QUERY if has_data else UserIntent.DOCUMENT_SEARCH)
            
            # Map intent to track
            track = INTENT_TO_TRACK.get(intent, TRACK_DATA if has_data else TRACK_DOC)
            
            logger.info(f"LLM intent classification: '{query[:40]}...' -> {intent.value} -> {track} (confidence: {llm_confidence:.2f})")
            
            return {
                "track": track,
                "confidence": llm_confidence,
                "method": "llm_intent_classifier",
                "intent": intent.value,
                "reasoning": reasoning
            }

        except Exception as e:
            logger.error(f"LLM intent classification failed: {e}")
            # Fallback to data track if user has data, else doc
            return {
                "track": TRACK_DATA if has_data else TRACK_DOC,
                "confidence": 0.5,
                "method": "llm_fallback",
                "error": str(e)
            }

    def clear_cache(self):
        self._route_cache.clear()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        return {
            "cache_size": len(self._route_cache),
            "max_size": self._cache_max_size,
            "has_data": self._has_loaded_data
        }


_router: Optional[RouterAgent] = None


def get_router_agent(llm_wrapper=None) -> RouterAgent:
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
]

