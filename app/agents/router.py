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
import hashlib

try:
    from app.core.dll_fix import apply_dll_fix
    apply_dll_fix()
except ImportError:
    pass

logger = logging.getLogger(__name__)

TRACK_DATA = "TRACK_DATA"
TRACK_DOC = "TRACK_DOC"
TRACK_WEB = "TRACK_WEB"
TRACK_DOC_SUMMARY = "TRACK_DOC_SUMMARY"
TRACK_OUT_OF_DOMAIN = "TRACK_OUT_OF_DOMAIN"

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
    "calculate from our data", "analyze our dataset", "compute from uploaded file"
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
    "describe our dataset", "tell me about our data", "what is in this data"
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



class RouterAgent:

    def __init__(self, llm_wrapper=None):
        self._llm = llm_wrapper
        self._route_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_max_size = 1000
        self._has_loaded_data = False
        self._loaded_datasets_info = ""
        
        self._data_docs = []
        self._doc_docs = []
        self._web_docs = []
        self._summary_docs = []
        self._out_of_domain_docs = []
        self._analytical_docs = []
        
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
                ANALYTICAL_INTENT_EXEMPLARS
            )
            all_docs = list(_nlp.pipe(all_texts, batch_size=50))
            
            n_data = len(DATA_INTENT_EXEMPLARS)
            n_doc = len(DOC_INTENT_EXEMPLARS)
            n_web = len(WEB_INTENT_EXEMPLARS)
            n_summary = len(SUMMARY_INTENT_EXEMPLARS)
            n_ood = len(OUT_OF_DOMAIN_EXEMPLARS)
            
            self._data_docs = all_docs[:n_data]
            self._doc_docs = all_docs[n_data:n_data + n_doc]
            self._web_docs = all_docs[n_data + n_doc:n_data + n_doc + n_web]
            self._summary_docs = all_docs[n_data + n_doc + n_web:n_data + n_doc + n_web + n_summary]
            self._out_of_domain_docs = all_docs[n_data + n_doc + n_web + n_summary:n_data + n_doc + n_web + n_summary + n_ood]
            self._analytical_docs = all_docs[n_data + n_doc + n_web + n_summary + n_ood:]
            
            logger.debug(f"Pre-computed {len(all_docs)} spaCy docs for intent classification")

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
            return {"data": 0.0, "doc": 0.0, "web": 0.0, "summary": 0.0, "ood": 0.0, "analytical": 0.0}
        
        query_doc = _nlp(query.lower())
        
        return {
            "data": max((query_doc.similarity(doc) for doc in self._data_docs), default=0.0) if self._data_docs else 0.0,
            "doc": max((query_doc.similarity(doc) for doc in self._doc_docs), default=0.0) if self._doc_docs else 0.0,
            "web": max((query_doc.similarity(doc) for doc in self._web_docs), default=0.0) if self._web_docs else 0.0,
            "summary": max((query_doc.similarity(doc) for doc in self._summary_docs), default=0.0) if self._summary_docs else 0.0,
            "ood": max((query_doc.similarity(doc) for doc in self._out_of_domain_docs), default=0.0) if self._out_of_domain_docs else 0.0,
            "analytical": max((query_doc.similarity(doc) for doc in self._analytical_docs), default=0.0) if self._analytical_docs else 0.0
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
                "is_real_time": True
            }
            if len(self._route_cache) >= self._cache_max_size:
                keys_to_remove = list(self._route_cache.keys())[:100]
                for k in keys_to_remove:
                    del self._route_cache[k]
            self._route_cache[cache_key] = result
            return result
        
        is_summary_eligible = (
            similarities["summary"] > 0.70 and 
            data_loaded and 
            not is_real_time and
            (has_summary_keyword or similarities["summary"] > 0.95)
        )
        
        if is_summary_eligible:
            result = {
                "track": TRACK_DOC_SUMMARY,
                "confidence": round(min(0.8 + similarities["summary"] * 0.15, 0.98), 2),
                "method": "semantic_summary",
                "similarities": {k: round(v, 3) for k, v in similarities.items()},
                "is_analytical": False,
                "is_summary": True,
                "is_real_time": False
            }
            if len(self._route_cache) >= self._cache_max_size:
                keys_to_remove = list(self._route_cache.keys())[:100]
                for k in keys_to_remove:
                    del self._route_cache[k]
            self._route_cache[cache_key] = result
            return result
        
        if similarities["ood"] > 0.85:
            result = {
                "track": TRACK_OUT_OF_DOMAIN,
                "confidence": round(min(0.8 + similarities["ood"] * 0.15, 0.98), 2),
                "method": "semantic_ood",
                "similarities": {k: round(v, 3) for k, v in similarities.items()},
                "is_analytical": False,
                "is_summary": False,
                "is_real_time": False
            }
            if len(self._route_cache) >= self._cache_max_size:
                keys_to_remove = list(self._route_cache.keys())[:100]
                for k in keys_to_remove:
                    del self._route_cache[k]
            self._route_cache[cache_key] = result
            return result
        
        is_analytical = similarities["analytical"] > 0.60
        
        has_historical = any(kw in query_lower for kw in self._historical_keywords)
        
        if self._has_loaded_data and not is_real_time and not has_summary_keyword:
            similarities["data"] += 0.25
        
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
        
        if track is None or confidence < 0.6:
            if self._llm:
                context = client_context or self._loaded_datasets_info or ""
                if data_loaded and not context:
                    context = "User has data loaded and ready for analysis."
                llm_result = self._route_with_llm(query, context)
                if llm_result.get("confidence", 0) > confidence:
                    llm_result["is_analytical"] = is_analytical
                    llm_result["is_summary"] = is_summary_eligible
                    llm_result["is_real_time"] = is_real_time
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
            "is_real_time": is_real_time
        }
        
        if len(self._route_cache) >= self._cache_max_size:
            keys_to_remove = list(self._route_cache.keys())[:100]
            for k in keys_to_remove:
                del self._route_cache[k]
        
        self._route_cache[cache_key] = result
        return result

    def _route_with_llm(self, query: str, client_context: Optional[str] = None) -> Dict[str, Any]:
        try:
            from app.core.prompts import get_router_prompt
            prompt = get_router_prompt(query, client_context or "")
            response = self._llm.invoke(prompt)
            response_upper = response.upper().strip()
            
            if "TRACK_DATA" in response_upper or "DATA" in response_upper:
                track = TRACK_DATA
            elif "TRACK_DOC" in response_upper or "DOCUMENT" in response_upper:
                track = TRACK_DOC
            elif "TRACK_WEB" in response_upper or "WEB" in response_upper:
                track = TRACK_WEB
            else:
                track = TRACK_DATA if self._has_loaded_data else TRACK_DOC

            result = {
                "track": track,
                "confidence": 0.8,
                "method": "llm",
            }
            
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

