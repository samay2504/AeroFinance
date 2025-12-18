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
    "get_semantic_cache",
    "get_rate_limiter",
]
