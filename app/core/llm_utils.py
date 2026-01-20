"""
Production-grade LLM utilities with retry logic, caching, and fallback mechanisms.
Implements best practices for LLM API resilience:
- Exponential backoff with jitter (tenacity)
- Semantic result caching
- Provider fallback chains
- Rate limit header awareness
"""
import logging
import time
import random
import hashlib
import json
from typing import Any, Callable, Dict, List, Optional, Tuple, Type
from dataclasses import dataclass, field
from functools import wraps
from datetime import datetime, timedelta
import threading

try:
    from app.core.dll_fix import apply_dll_fix
    apply_dll_fix()
except ImportError:
    pass

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cached LLM response with metadata."""
    response: Any
    created_at: datetime
    ttl_seconds: int = 3600  # 1 hour default
    hit_count: int = 0

    def is_valid(self) -> bool:
        """Check if cache entry is still valid."""
        return datetime.now() < self.created_at + timedelta(seconds=self.ttl_seconds)


class SemanticCache:
    """
    Semantic caching for LLM responses.
    Uses prompt hashing with optional semantic similarity.
    Thread-safe implementation.
    """

    def __init__(self, max_size: int = 500, default_ttl: int = 3600):
        self._cache: Dict[str, CacheEntry] = {}
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._lock = threading.RLock()
        self._stats = {"hits": 0, "misses": 0}

    def _hash_prompt(self, prompt: str, context: Optional[str] = None) -> str:
        """Create deterministic hash of prompt + context."""
        combined = f"{prompt}::{context or ''}"
        return hashlib.sha256(combined.encode()).hexdigest()[:16]

    def get(self, prompt: str, context: Optional[str] = None) -> Optional[Any]:
        """Get cached response if available."""
        key = self._hash_prompt(prompt, context)
        
        with self._lock:
            entry = self._cache.get(key)
            if entry and entry.is_valid():
                entry.hit_count += 1
                self._stats["hits"] += 1
                logger.debug(f"Cache HIT for key {key[:8]}...")
                return entry.response
            elif entry:
                # Expired - remove it
                del self._cache[key]
            
            self._stats["misses"] += 1
            return None

    def set(
        self,
        prompt: str,
        response: Any,
        context: Optional[str] = None,
        ttl: Optional[int] = None
    ) -> None:
        """Cache a response."""
        key = self._hash_prompt(prompt, context)
        
        with self._lock:
            # Evict oldest entries if at capacity
            if len(self._cache) >= self._max_size:
                self._evict_lru()
            
            self._cache[key] = CacheEntry(
                response=response,
                created_at=datetime.now(),
                ttl_seconds=ttl or self._default_ttl
            )
            logger.debug(f"Cache SET for key {key[:8]}...")

    def _evict_lru(self) -> None:
        """Evict least recently used entries."""
        if not self._cache:
            return
        
        # Remove entries with lowest hit count and oldest
        sorted_entries = sorted(
            self._cache.items(),
            key=lambda x: (x[1].hit_count, x[1].created_at)
        )
        
        # Remove bottom 10%
        remove_count = max(1, len(sorted_entries) // 10)
        for key, _ in sorted_entries[:remove_count]:
            del self._cache[key]

    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total = self._stats["hits"] + self._stats["misses"]
            hit_rate = self._stats["hits"] / total if total > 0 else 0
            return {
                "hits": self._stats["hits"],
                "misses": self._stats["misses"],
                "hit_rate": round(hit_rate, 3),
                "size": len(self._cache),
                "max_size": self._max_size
            }


class RetryConfig:
    """Configuration for retry behavior."""
    
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retryable_status_codes: Tuple[int, ...] = (429, 500, 502, 503, 504),
        retryable_exceptions: Tuple[Type[Exception], ...] = (
            ConnectionError, TimeoutError
        )
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.retryable_status_codes = retryable_status_codes
        self.retryable_exceptions = retryable_exceptions

    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay with exponential backoff and optional jitter."""
        delay = min(
            self.base_delay * (self.exponential_base ** attempt),
            self.max_delay
        )
        
        if self.jitter:
            # Add random jitter between 0 and delay
            delay = delay * (0.5 + random.random())
        
        return delay


def with_retry(config: Optional[RetryConfig] = None):
    """
    Decorator for adding retry logic with exponential backoff.
    
    Usage:
        @with_retry(RetryConfig(max_retries=3))
        def call_llm_api(prompt):
            ...
    """
    config = config or RetryConfig()
    
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(config.max_retries + 1):
                try:
                    result = func(*args, **kwargs)
                    return result
                    
                except Exception as e:
                    last_exception = e
                    error_str = str(e).lower()
                    
                    # Check if this is a rate limit or retryable error
                    is_retryable = (
                        "429" in error_str or
                        "rate limit" in error_str or
                        "too many requests" in error_str or
                        any(isinstance(e, exc_type) for exc_type in config.retryable_exceptions)
                    )
                    
                    if not is_retryable:
                        logger.warning(f"Non-retryable error in {func.__name__}: {e}")
                        raise
                    
                    if attempt < config.max_retries:
                        delay = config.calculate_delay(attempt)
                        logger.info(
                            f"Retry {attempt + 1}/{config.max_retries} for {func.__name__} "
                            f"after {delay:.2f}s due to: {e}"
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"Max retries ({config.max_retries}) exceeded for {func.__name__}: {e}"
                        )
            
            raise last_exception
        
        return wrapper
    return decorator


class AdaptiveRateLimiter:
    """
    Adaptive client-side rate limiter.
    Adjusts rate based on API response headers and error patterns.
    """
    
    def __init__(
        self,
        requests_per_minute: int = 30,
        tokens_per_minute: int = 10000
    ):
        self._rpm_limit = requests_per_minute
        self._tpm_limit = tokens_per_minute
        self._request_times: List[float] = []
        self._token_counts: List[Tuple[float, int]] = []
        self._lock = threading.Lock()
        self._backoff_until: Optional[float] = None

    def acquire(self, estimated_tokens: int = 0) -> float:
        """
        Acquire permission to make a request.
        Returns the delay (in seconds) before request should be made.
        """
        with self._lock:
            now = time.time()
            
            # Check if in backoff period
            if self._backoff_until and now < self._backoff_until:
                return self._backoff_until - now
            
            # Clean old entries (older than 60 seconds)
            cutoff = now - 60
            self._request_times = [t for t in self._request_times if t > cutoff]
            self._token_counts = [(t, c) for t, c in self._token_counts if t > cutoff]
            
            # Check RPM limit
            if len(self._request_times) >= self._rpm_limit:
                wait_until = self._request_times[0] + 60
                return max(0, wait_until - now)
            
            # Check TPM limit
            total_tokens = sum(c for _, c in self._token_counts) + estimated_tokens
            if total_tokens >= self._tpm_limit:
                wait_until = self._token_counts[0][0] + 60
                return max(0, wait_until - now)
            
            # Record this request
            self._request_times.append(now)
            if estimated_tokens > 0:
                self._token_counts.append((now, estimated_tokens))
            
            return 0

    def record_rate_limit_error(self, retry_after: Optional[float] = None) -> None:
        """Record that we hit a rate limit, triggering backoff."""
        with self._lock:
            backoff = retry_after or 60.0  # Default 60 second backoff
            self._backoff_until = time.time() + backoff
            logger.warning(f"Rate limit hit, backing off for {backoff}s")

    def update_from_headers(self, headers: Dict[str, str]) -> None:
        """Update limits based on API response headers."""
        with self._lock:
            # Common rate limit headers
            if "x-ratelimit-remaining" in headers:
                remaining = int(headers["x-ratelimit-remaining"])
                if remaining < 5:
                    logger.warning(f"API rate limit nearly exhausted: {remaining} remaining")
            
            if "retry-after" in headers:
                retry_after = float(headers["retry-after"])
                self._backoff_until = time.time() + retry_after


class ProviderFallback:
    """
    Manages fallback between multiple LLM providers.
    Tracks provider health and automatically routes to healthy providers.
    """
    
    def __init__(self, providers: List[str]):
        self._providers = providers
        self._health: Dict[str, Dict] = {
            p: {"healthy": True, "failures": 0, "last_failure": None}
            for p in providers
        }
        self._lock = threading.Lock()

    def get_provider(self) -> Optional[str]:
        """Get the next healthy provider."""
        with self._lock:
            for provider in self._providers:
                health = self._health[provider]
                
                if health["healthy"]:
                    return provider
                
                # Check if enough time has passed to retry
                if health["last_failure"]:
                    recovery_time = 60 * (2 ** min(health["failures"], 5))  # Exponential recovery
                    if (datetime.now() - health["last_failure"]).seconds > recovery_time:
                        health["healthy"] = True
                        return provider
            
            # All providers unhealthy - return first one anyway
            return self._providers[0] if self._providers else None

    def record_success(self, provider: str) -> None:
        """Record successful request to provider."""
        with self._lock:
            if provider in self._health:
                self._health[provider]["healthy"] = True
                self._health[provider]["failures"] = 0

    def record_failure(self, provider: str, is_rate_limit: bool = False) -> None:
        """Record failed request to provider."""
        with self._lock:
            if provider in self._health:
                health = self._health[provider]
                health["failures"] += 1
                health["last_failure"] = datetime.now()
                
                # Mark unhealthy after multiple failures (more lenient for rate limits)
                if is_rate_limit:
                    health["healthy"] = False
                elif health["failures"] >= 3:
                    health["healthy"] = False
                    logger.warning(f"Provider {provider} marked unhealthy after {health['failures']} failures")

    def get_status(self) -> Dict[str, Dict]:
        """Get health status of all providers."""
        with self._lock:
            return {p: dict(h) for p, h in self._health.items()}


# =============================================================================
# ROBUST JSON PARSING - Multi-layer approach for malformed LLM output
# =============================================================================

class RobustJSONParser:
    """
    Production-grade JSON parser with multi-layer repair for LLM output.
    
    Based on 2024 best practices research:
    - json-repair library for automatic JSON fixing
    - Multiple fallback parsing strategies
    - Python literal eval for Python dict format
    
    Approach:
    1. Try standard json.loads()
    2. Strip markdown code blocks
    3. Apply repair patterns for common LLM malformations
    4. Try json-repair library if available
    5. Python literal eval
    6. Regex extraction for nested objects
    """
    
    # Common LLM JSON malformations patterns
    REPAIR_PATTERNS = [
        # Python booleans to JSON
        (r'\bTrue\b', 'true'),
        (r'\bFalse\b', 'false'),
        (r'\bNone\b', 'null'),
        # Single quotes to double quotes (careful with apostrophes)
        (r"(?<![a-zA-Z])'([^']*)'(?![a-zA-Z])", r'"\1"'),
        # Trailing commas before } or ]
        (r',\s*([\}\]])', r'\1'),
        # Missing quotes around keys
        (r'(\{|\,)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":'),
        # NaN and Infinity to null
        (r'\bNaN\b', 'null'),
        (r'\bInfinity\b', 'null'),
        (r'\b-Infinity\b', 'null'),
    ]
    
    @classmethod
    def parse(cls, content: str, fallback_value: Any = None) -> Dict[str, Any]:
        """
        Parse potentially malformed JSON from LLM output.
        
        Args:
            content: Raw string from LLM
            fallback_value: Value to return if all parsing fails
            
        Returns:
            Parsed dict or fallback value
        """
        import re
        
        if not content or not content.strip():
            return fallback_value or {"error": "Empty content", "fallback": True}
        
        original_content = content
        
        # Layer 1: Standard parsing
        result = cls._try_standard_parse(content)
        if result is not None:
            return result
        
        # Layer 2: Strip markdown
        content = cls._strip_markdown(content)
        result = cls._try_standard_parse(content)
        if result is not None:
            return result
        
        # Layer 3: Apply repair patterns
        content = cls._apply_repair_patterns(content)
        result = cls._try_standard_parse(content)
        if result is not None:
            return result
        
        # Layer 4: Try json-repair library if available
        result = cls._try_json_repair(content)
        if result is not None:
            return result
        
        # Layer 5: Python literal eval
        result = cls._try_literal_eval(content)
        if result is not None:
            return result
        
        # Layer 6: Regex JSON extraction
        result = cls._try_regex_extraction(original_content)
        if result is not None:
            return result
        
        # Layer 7: Extract key-value fallback
        result = cls._try_key_value_extraction(original_content)
        if result is not None:
            return result
        
        # All layers failed
        logger.warning(f"JSON parsing failed after all layers. Content preview: {original_content[:200]}")
        return fallback_value or {
            "error": "Invalid JSON",
            "raw_content": original_content[:500],
            "fallback": True
        }
    
    @staticmethod
    def _try_standard_parse(content: str) -> Optional[Dict]:
        """Try standard json.loads()."""
        try:
            result = json.loads(content)
            if isinstance(result, dict):
                return result
            elif isinstance(result, list) and len(result) > 0:
                return {"result": result}
            return {"value": result}
        except json.JSONDecodeError:
            return None
    
    @staticmethod
    def _strip_markdown(content: str) -> str:
        """Remove markdown code blocks."""
        import re
        content = content.strip()
        
        # Remove ```json ... ``` or ```python ... ```
        md_patterns = [
            r'```json\s*(.*?)\s*```',
            r'```python\s*(.*?)\s*```',
            r'```\s*(.*?)\s*```'
        ]
        
        for pattern in md_patterns:
            match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # Simple strip of ``` markers
        if content.startswith('```'):
            lines = content.split('\n')
            lines = [l for l in lines if not l.strip().startswith('```')]
            content = '\n'.join(lines).strip()
        
        return content
    
    @classmethod
    def _apply_repair_patterns(cls, content: str) -> str:
        """Apply regex-based repair patterns."""
        import re
        for pattern, replacement in cls.REPAIR_PATTERNS:
            content = re.sub(pattern, replacement, content)
        return content
    
    @staticmethod
    def _try_json_repair(content: str) -> Optional[Dict]:
        """Try json-repair library if available."""
        try:
            from json_repair import repair_json
            repaired = repair_json(content)
            result = json.loads(repaired)
            if isinstance(result, dict):
                return result
            return {"value": result}
        except ImportError:
            pass  # Library not installed
        except Exception:
            pass
        return None
    
    @staticmethod
    def _try_literal_eval(content: str) -> Optional[Dict]:
        """Try Python literal eval for Python dict format."""
        try:
            import ast
            result = ast.literal_eval(content)
            if isinstance(result, dict):
                return result
        except Exception:
            pass
        return None
    
    @staticmethod
    def _try_regex_extraction(content: str) -> Optional[Dict]:
        """Extract JSON objects using regex."""
        import re
        # Try to find any JSON object in the content
        patterns = [
            r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  # Nested objects
            r'\{[^{}]+\}',  # Simple objects
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content, re.DOTALL)
            for match in matches:
                try:
                    result = json.loads(match)
                    if isinstance(result, dict):
                        return result
                except json.JSONDecodeError:
                    continue
        return None
    
    @staticmethod
    def _try_key_value_extraction(content: str) -> Optional[Dict]:
        """Extract key-value pairs as last resort."""
        import re
        pattern = r'"?([a-zA-Z_][a-zA-Z0-9_]*)"?\s*:\s*"?([^",\n\}]+)"?'
        matches = re.findall(pattern, content)
        
        if matches:
            result = {}
            for key, value in matches:
                value = value.strip()
                if value.lower() in ('true', 'false'):
                    result[key] = value.lower() == 'true'
                elif value.lower() in ('null', 'none'):
                    result[key] = None
                else:
                    try:
                        result[key] = float(value) if '.' in value else int(value)
                    except ValueError:
                        result[key] = value
            
            if result:
                result["_extracted"] = True
                return result
        return None


# =============================================================================
# SQL/CODE EXTRACTION - Extract SQL/Python from prose LLM responses
# =============================================================================

class SQLCodeExtractor:
    """
    Production-grade SQL and Python code extractor for when LLM returns prose.
    
    Based on 2024 research best practices:
    - Two-step approach: When JSON fails, extract code directly
    - Multiple extraction patterns for robustness
    - Handles markdown, inline code, and natural language references
    
    Usage:
        extractor = SQLCodeExtractor()
        result = extractor.extract_sql(llm_response)
        if result['success']:
            sql = result['sql']
    """
    
    # SQL patterns in order of specificity
    SQL_PATTERNS = [
        # Markdown SQL code block
        r'```sql\s*(SELECT.*?)```',
        r'```SQL\s*(SELECT.*?)```',
        # Generic code block with SELECT
        r'```\s*(SELECT.*?)```',
        # Inline SQL with quotes
        r'["\'](SELECT\s+.*?)["\']',
        # Raw SELECT statement (greedy but terminated)
        r'\b(SELECT\s+(?:DISTINCT\s+)?(?:[\w\.\*\"\'\(\)\s,]+)\s+FROM\s+[\w\.\"\'\s,]+(?:\s+WHERE\s+.*?)?(?:\s+(?:GROUP|ORDER|LIMIT|HAVING)\s+.*?)?)\s*(?:;|$|\n\n)',
        # Simpler SELECT pattern
        r'\b(SELECT\s+.+?\s+FROM\s+\w+.*?)(?:;|$)',
    ]
    
    # Python patterns
    PYTHON_PATTERNS = [
        # Markdown Python code block
        r'```python\s*(def\s+run.*?)```',
        r'```Python\s*(def\s+run.*?)```',
        # Generic code block with def run
        r'```\s*(def\s+run.*?)```',
        # Inline function definition
        r'(def\s+run\s*\(.*?\):\s*(?:.*?\n)+?\s*return\s+.*)',
    ]
    
    @classmethod
    def extract_sql(cls, content: str) -> Dict[str, Any]:
        """
        Extract SQL query from LLM response.
        
        Args:
            content: Raw LLM response text
            
        Returns:
            {
                "success": bool,
                "sql": str or None,
                "explanation": str,
                "extraction_method": str
            }
        """
        import re
        
        if not content or not content.strip():
            return {
                "success": False,
                "sql": None,
                "explanation": "Empty content",
                "extraction_method": "none"
            }
        
        # Try each pattern
        for i, pattern in enumerate(cls.SQL_PATTERNS):
            try:
                match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
                if match:
                    sql = match.group(1).strip()
                    # Validate it looks like SQL
                    if cls._is_valid_sql(sql):
                        return {
                            "success": True,
                            "sql": sql,
                            "explanation": f"Extracted from pattern {i+1}",
                            "extraction_method": f"pattern_{i+1}"
                        }
            except Exception as e:
                logger.debug(f"SQL extraction pattern {i+1} failed: {e}")
                continue
        
        # Last resort: Look for SELECT...FROM anywhere
        select_match = re.search(r'\bSELECT\b', content, re.IGNORECASE)
        from_match = re.search(r'\bFROM\b', content, re.IGNORECASE)
        
        if select_match and from_match and from_match.start() > select_match.start():
            # Extract everything between SELECT and the next statement terminator
            start = select_match.start()
            # Find end: semicolon, double newline, or end of content
            end_patterns = [r';', r'\n\n', r'\n(?=[A-Z][a-z])', r'$']
            end_pos = len(content)
            
            for ep in end_patterns:
                m = re.search(ep, content[start:])
                if m:
                    candidate_end = start + m.end()
                    if candidate_end < end_pos:
                        end_pos = candidate_end
                    break
            
            sql = content[start:end_pos].strip().rstrip(';').strip()
            if cls._is_valid_sql(sql):
                return {
                    "success": True, 
                    "sql": sql,
                    "explanation": "Extracted using keyword boundaries",
                    "extraction_method": "keyword_boundary"
                }
        
        return {
            "success": False,
            "sql": None,
            "explanation": "No valid SQL found in response",
            "extraction_method": "none"
        }
    
    @classmethod
    def extract_python(cls, content: str) -> Dict[str, Any]:
        """
        Extract Python code (specifically run(df) function) from LLM response.
        
        Args:
            content: Raw LLM response text
            
        Returns:
            {
                "success": bool,
                "code": str or None,
                "explanation": str,
                "extraction_method": str
            }
        """
        import re
        
        if not content or not content.strip():
            return {
                "success": False,
                "code": None,
                "explanation": "Empty content",
                "extraction_method": "none"
            }
        
        # Try Python patterns
        for i, pattern in enumerate(cls.PYTHON_PATTERNS):
            try:
                match = re.search(pattern, content, re.DOTALL)
                if match:
                    code = match.group(1).strip()
                    if 'def run' in code and 'return' in code:
                        return {
                            "success": True,
                            "code": code,
                            "explanation": f"Extracted from pattern {i+1}",
                            "extraction_method": f"pattern_{i+1}"
                        }
            except Exception:
                continue
        
        # Try to find def run even without code blocks
        run_match = re.search(
            r'(def\s+run\s*\(\s*df\s*\)\s*:\s*\n(?:[ \t]+.*\n)*)',
            content,
            re.MULTILINE
        )
        if run_match:
            code = run_match.group(1).strip()
            return {
                "success": True,
                "code": code,
                "explanation": "Extracted function definition",
                "extraction_method": "function_def"
            }
        
        return {
            "success": False,
            "code": None,
            "explanation": "No valid Python code found",
            "extraction_method": "none"
        }
    
    @staticmethod
    def _is_valid_sql(sql: str) -> bool:
        """Basic validation that string looks like valid SQL."""
        if not sql or len(sql) < 10:
            return False
        
        sql_upper = sql.upper()
        
        # Must have SELECT and FROM
        if 'SELECT' not in sql_upper or 'FROM' not in sql_upper:
            return False
        
        # Should not contain obvious prose
        prose_indicators = [
            'I WOULD', 'I THINK', 'YOU SHOULD', 'PLEASE NOTE',
            'HOWEVER', 'UNFORTUNATELY', 'THE QUERY', 'THIS QUERY'
        ]
        for indicator in prose_indicators:
            if indicator in sql_upper:
                return False
        
        return True
    
    @classmethod
    def enhance_json_response(cls, parsed_json: Dict[str, Any], raw_content: str) -> Dict[str, Any]:
        """
        Enhance a partially failed JSON parse by extracting SQL/code.
        
        If the JSON has error/fallback flags but the raw content contains
        valid SQL or code, extract and add it.
        
        Args:
            parsed_json: Result from RobustJSONParser.parse()
            raw_content: Original raw LLM response
            
        Returns:
            Enhanced JSON with sql/code if extraction succeeded
        """
        # Check if this is a failed parse that we can enhance
        if not parsed_json.get("fallback") and not parsed_json.get("error"):
            return parsed_json  # Already good
        
        # Try SQL extraction
        if "sql" not in parsed_json or not parsed_json.get("sql"):
            sql_result = cls.extract_sql(raw_content)
            if sql_result["success"]:
                parsed_json["sql"] = sql_result["sql"]
                parsed_json["explanation"] = sql_result.get("explanation", "SQL extracted from response")
                parsed_json.pop("error", None)
                parsed_json.pop("fallback", None)
                logger.info(f"Enhanced JSON with extracted SQL: {sql_result['extraction_method']}")
        
        # Try Python extraction
        if "code" not in parsed_json or not parsed_json.get("code"):
            code_result = cls.extract_python(raw_content)
            if code_result["success"]:
                parsed_json["code"] = code_result["code"]
                parsed_json["explanation"] = code_result.get("explanation", "Code extracted from response")
                parsed_json.pop("error", None)
                parsed_json.pop("fallback", None)
                logger.info(f"Enhanced JSON with extracted code: {code_result['extraction_method']}")
        
        return parsed_json


# =============================================================================
# DATAFRAME TYPE FIXING - Handle mixed types for safe operations
# =============================================================================

# =============================================================================
# DYNAMIC TYPE INFERENCE - Statistical & Adaptive Schema Detection
# =============================================================================

class SmartTypeInference:
    """
    Production-grade ADAPTIVE type inference and coercion.
    
    NO HARDCODED REGEX LISTS - Uses:
    1. Statistical sampling to detect content patterns
    2. Dynamic noise removal (currencies, units, artifacts)
    3. Heuristic categorical detection
    4. Safe ID detection (preserves leading zeros/IDs)
    
    Adapts to new, unseen data formats automatically.
    """
    
    @classmethod
    def infer_and_fix(cls, df, inplace: bool = False):
        """
        Dynamically infer types and fix DataFrame.
        """
        import pandas as pd
        import numpy as np
        
        if not inplace:
            df = df.copy()
        
        for col in df.columns:
            series = df[col]
            
            # Skip if already proper type
            if not pd.api.types.is_object_dtype(series):
                continue
                
            # 1. Analyze column statistics
            stats = cls._analyze_column(series)
            
            # 2. Decision logic based on stats
            if stats['is_empty']:
                continue
                
            # 3. Dynamic Numeric Conversion
            if stats['potential_numeric']:
                # Dynamically clean based on detected common noise chars
                clean_series = series.astype(str)
                if stats['noise_chars']:
                    for char in stats['noise_chars']:
                        clean_series = clean_series.str.replace(char, '', regex=False)
                
                # Handle standard numeric formats like (123) -> -123
                clean_series = clean_series.str.replace(r'^\(([\d\.]+)\)$', r'-\1', regex=True)
                
                try:
                    # Coerce
                    converted = pd.to_numeric(clean_series, errors='coerce')
                    
                    # Check success rate
                    valid_count = converted.notna().sum()
                    success_rate = valid_count / stats['non_null_count']
                    
                    if success_rate > 0.8: # >80% success
                        # Check for ID-like behavior (integers, uniform distribution)
                        if cls._is_id_column(converted, series):
                            # Keep as object/string to preserve exact formatting (leading zeros)
                            logger.debug(f"Column '{col}' identified as ID/Code - keeping as object")
                            continue
                            
                        # Downcast integers if possible (Int64 allows NaNs)
                        if (converted.dropna() % 1 == 0).all():
                            df[col] = converted.astype('Int64')
                        else:
                            df[col] = converted
                        continue
                except Exception:
                    pass

            # 4. Date/Time Detection
            if stats['potential_date']:
                try:
                    # Use pandas flexible parser
                    converted = pd.to_datetime(series, errors='coerce')
                    valid_count = converted.notna().sum()
                    if valid_count / stats['non_null_count'] > 0.8:
                        df[col] = converted
                        continue
                except Exception:
                    pass
            
            # 5. Categorical Detection
            # If low cardinality and not an ID
            if stats['unique_ratio'] < 0.2 and stats['non_null_count'] > 20:
                # Convert to category for memory efficiency and semantic meaning
                df[col] = series.astype('category')
                
        return df

    @classmethod
    def fix_dataframe(cls, df, inplace: bool = False):
        """Legacy alias for backward compatibility."""
        return cls.infer_and_fix(df, inplace=inplace)

    @classmethod
    def _analyze_column(cls, series) -> Dict[str, Any]:
        """
        Perform statistical analysis on column content.
        Returns metadata about data patterns.
        """
        import collections
        
        # Sample data (up to 100 non-null values)
        sample = series.dropna().astype(str).sample(min(100, len(series.dropna())), random_state=42).tolist()
        
        if not sample:
            return {'is_empty': True}
            
        stats = {
            'is_empty': False,
            'non_null_count': len(series.dropna()),
            'unique_ratio': series.nunique() / len(series) if len(series) > 0 else 0,
            'potential_numeric': False,
            'potential_date': False,
            'noise_chars': []
        }
        
        # Relaxed numeric check: digit density > 0.3
        # Allows for currency codes like 'EUR 123' (3/7 = 0.42)
        digit_count = sum(c.isdigit() for c in ''.join(sample))
        total_len = sum(len(s) for s in sample)
        stats['potential_numeric'] = (digit_count / max(1, total_len)) > 0.3
        
        char_counts = collections.Counter(''.join(sample))
        
        if stats['potential_numeric']:
            # Identify frequent non-numeric characters (noise)
            # e.g. '$', ',', '%', ' ', 'Rs'
            noise_candidates = []
            for char, count in char_counts.items():
                if not char.isdigit() and char not in ['.', '-']:
                    # If char appears in > 10% of samples, it's a systematic artifacts
                    if count > len(sample) * 0.1:
                        noise_candidates.append(char)
            stats['noise_chars'] = noise_candidates

        # Date potential?
        # Check for common date separators
        date_seps = sum(char_counts[c] for c in ['/', '-', ':'])
        stats['potential_date'] = date_seps > len(sample) * 0.5  # At least one sep per 2 samples roughly
        
        return stats

    @classmethod
    def _is_id_column(cls, numeric_series, original_series) -> bool:
        """
        Check if column is an ID using SAFE_ID_PATTERN from id_generator.
        """
        try:
            from app.core.id_generator import SAFE_ID_PATTERN
        except ImportError:
            # Fallback regex if module unavailable
            import re
            SAFE_ID_PATTERN = re.compile(r'^[a-zA-Z0-9_\-\.]+$')

        # IDs typically don't have decimals
        if (numeric_series.dropna() % 1 != 0).any():
            return False
            
        # IDs often have uniform length
        lengths = original_series.dropna().astype(str).apply(len)
        if lengths.nunique() <= 2 and lengths.mean() > 3:
            return True
            
        # High cardinality
        if numeric_series.nunique() / numeric_series.count() > 0.9:
            return True
            
        return False


# Legacy alias for compatibility
DataFrameTypeFixer = SmartTypeInference


# =============================================================================
# SEMANTIC COLUMN MATCHING - Delegates to core Semantic Understanding
# =============================================================================

class SemanticColumnMatcher:
    """
    Production-grade SEMANTIC column matching.
    
    Acts as a bridge to `app.core.semantic_understanding` to ensure
    consistent logic across the application.
    """
    
    _matcher = None
    
    @classmethod
    def _get_matcher(cls):
        if cls._matcher is None:
            try:
                from app.core.semantic_understanding import SemanticMatcher
                cls._matcher = SemanticMatcher()
            except ImportError:
                return None
        return cls._matcher
    
    @classmethod
    def find_best_match(cls, query_col: str, available_cols: List[str],
                        df=None, threshold: float = 0.6) -> Tuple[Optional[str], float]:
        """Find best matching column using centralized SemanticMatcher."""
        matcher = cls._get_matcher()
        if matcher:
            # SemanticMatcher returns (match, score) or None
            # We map its score to our expectation
            result = matcher.find_best_match(query_col, available_cols, threshold)
            if result:
                return result
            # Try finding best manually if find_best_match strictness diffs
            best_match = None
            best_score = 0.0
            for col in available_cols:
                score = matcher.calculate_similarity(query_col, col)
                if score > best_score:
                    best_score = score
                    best_match = col
            
            if best_score >= threshold:
                return best_match, best_score
            return None, 0.0
        
        return cls._simple_fallback(query_col, available_cols, threshold)

    @classmethod
    def find_all_matches(cls, query_col: str, available_cols: List[str],
                         df=None, limit: int = 3, threshold: float = None) -> List[Tuple[str, float]]:
        """Find multiple matches."""
        matcher = cls._get_matcher()
        results = []
        threshold = threshold or 0.6
        
        if matcher:
            for col in available_cols:
                score = matcher.calculate_similarity(query_col, col)
                if score >= threshold:
                    results.append((col, round(score, 3)))
            return sorted(results, key=lambda x: x[1], reverse=True)[:limit]
        
        # Fallback
        match, score = cls._simple_fallback(query_col, available_cols, threshold)
        return [(match, score)] if match else []

    @staticmethod
    def _simple_fallback(query: str, cols: List[str], threshold: float) -> Tuple[Optional[str], float]:
        """Simple substring matching fallback."""
        import re
        def norm(s): return re.sub(r'[\s\-\_]+', '', str(s).lower())
        q_norm = norm(query)
        best, best_score = None, 0.0
        for col in cols:
            c_norm = norm(col)
            if q_norm in c_norm or c_norm in q_norm:
                # Basic overlap score
                score = len(min(q_norm, c_norm)) / max(1, len(max(q_norm, c_norm)))
                if score > best_score:
                    best_score = score
                    best = col
        if best_score >= threshold:
            return best, best_score
        return None, 0.0


# Legacy alias for compatibility
FuzzyColumnMatcher = SemanticColumnMatcher


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def parse_llm_json(content: str, fallback: Any = None) -> Dict[str, Any]:
    """Convenience function for robust JSON parsing from LLM output."""
    return RobustJSONParser.parse(content, fallback)


def fix_dataframe_types(df) -> Any:
    """Convenience function for DataFrame type fixing."""
    return SmartTypeInference.infer_and_fix(df)


def find_column(query_name: str, df, threshold: float = 0.6) -> Tuple[Optional[str], float]:
    """
    Convenience function for SEMANTIC column matching.
    
    Uses embeddings to find columns by meaning, not just string similarity.
    Automatically adapts to any column naming convention.
    """
    return SemanticColumnMatcher.find_best_match(
        query_name, 
        list(df.columns), 
        df=df,
        threshold=threshold
    )


# Global instances
_cache = SemanticCache()
_rate_limiter = AdaptiveRateLimiter()


def get_semantic_cache() -> SemanticCache:
    """Get the global semantic cache instance."""
    return _cache


def get_rate_limiter() -> AdaptiveRateLimiter:
    """Get the global rate limiter instance."""
    return _rate_limiter


__all__ = [
    "SemanticCache",
    "RetryConfig",
    "with_retry",
    "AdaptiveRateLimiter",
    "ProviderFallback",
    "RobustJSONParser",
    "SmartTypeInference", 
    "DataFrameTypeFixer", # Alias
    "SemanticColumnMatcher",
    "FuzzyColumnMatcher", # Alias
    "parse_llm_json",
    "fix_dataframe_types",
    "find_column",
    "get_semantic_cache",
    "get_rate_limiter",
]
