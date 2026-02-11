"""
Metrics Collector - Production-grade observability for the agent system.

Provides:
- Self-healing execution stats (retry counts, success rates)
- Router performance metrics (L0/L1/L2 hit rates, latency)
- Schema filtering compression ratios
- LLM call tracking (tokens, latency, provider usage)
- Prometheus-compatible exposition (optional)

Thread-safe singleton implementation.
"""

import time
import threading
import logging
from typing import Any, Dict, List, Optional
from collections import defaultdict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ExecutionMetric:
    """Single execution measurement."""
    name: str
    value: float
    timestamp: float = field(default_factory=time.time)
    tags: Dict[str, str] = field(default_factory=dict)


class MetricsCollector:
    """
    Thread-safe metrics collector for production observability.
    
    Tracks:
    - Counters (monotonically increasing values)
    - Gauges (point-in-time values)
    - Histograms (distribution of values with percentiles)
    """

    def __init__(self):
        self._lock = threading.RLock()  # RLock: allows re-entrant locking (snapshot → get_histogram_stats)
        self._counters: Dict[str, int] = defaultdict(int)
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, List[float]] = defaultdict(list)
        self._MAX_HISTOGRAM_SIZE = 10000

    # ═══ COUNTERS ═══
    def increment(self, name: str, value: int = 1, **tags) -> None:
        """Increment a counter."""
        key = self._make_key(name, tags)
        with self._lock:
            self._counters[key] += value

    def get_counter(self, name: str, **tags) -> int:
        """Get current counter value."""
        key = self._make_key(name, tags)
        with self._lock:
            return self._counters.get(key, 0)

    # ═══ GAUGES ═══
    def set_gauge(self, name: str, value: float, **tags) -> None:
        """Set a gauge value."""
        key = self._make_key(name, tags)
        with self._lock:
            self._gauges[key] = value

    def get_gauge(self, name: str, **tags) -> float:
        """Get current gauge value."""
        key = self._make_key(name, tags)
        with self._lock:
            return self._gauges.get(key, 0.0)

    # ═══ HISTOGRAMS ═══
    def observe(self, name: str, value: float, **tags) -> None:
        """Record an observation in a histogram."""
        key = self._make_key(name, tags)
        with self._lock:
            hist = self._histograms[key]
            hist.append(value)
            # Prevent unbounded growth
            if len(hist) > self._MAX_HISTOGRAM_SIZE:
                hist[:] = hist[-self._MAX_HISTOGRAM_SIZE:]

    def get_histogram_stats(self, name: str, **tags) -> Dict[str, float]:
        """Get histogram statistics (count, min, max, mean, p50, p95, p99)."""
        key = self._make_key(name, tags)
        with self._lock:
            values = self._histograms.get(key, [])
            if not values:
                return {"count": 0}
            
            sorted_values = sorted(values)
            n = len(sorted_values)
            return {
                "count": n,
                "min": sorted_values[0],
                "max": sorted_values[-1],
                "mean": round(sum(sorted_values) / n, 4),
                "p50": sorted_values[int(n * 0.50)],
                "p95": sorted_values[min(int(n * 0.95), n - 1)],
                "p99": sorted_values[min(int(n * 0.99), n - 1)],
            }

    # ═══ CONVENIENCE: Self-Healing Metrics ═══
    def record_self_healing_attempt(
        self,
        method: str,
        attempt: int,
        success: bool,
        error_type: Optional[str] = None,
        latency_ms: float = 0,
    ):
        """Record a self-healing execution attempt."""
        self.increment("self_healing.attempts", method=method)
        if success:
            self.increment("self_healing.successes", method=method)
        else:
            self.increment("self_healing.failures", method=method)
            if error_type:
                self.increment("self_healing.error_types", error_type=error_type)
        
        self.set_gauge("self_healing.last_attempt_number", attempt, method=method)
        self.observe("self_healing.latency_ms", latency_ms, method=method)

    def get_self_healing_stats(self) -> Dict[str, Any]:
        """Get self-healing execution statistics."""
        stats = {}
        with self._lock:
            for key, count in self._counters.items():
                if key.startswith("self_healing."):
                    stats[key] = count
            for key, value in self._gauges.items():
                if key.startswith("self_healing."):
                    stats[key] = value
        
        # Calculate success rate
        attempts = stats.get("self_healing.attempts", 0)
        successes = stats.get("self_healing.successes", 0)
        if attempts > 0:
            stats["self_healing.success_rate"] = round(successes / attempts, 3)
        
        return stats

    # ═══ CONVENIENCE: Router Metrics ═══
    def record_route(self, track: str, method: str, latency_ms: float, confidence: float):
        """Record a routing decision."""
        self.increment("router.queries")
        self.increment("router.track_distribution", track=track)
        self.increment("router.method_distribution", method=method)
        self.observe("router.latency_ms", latency_ms)
        self.observe("router.confidence", confidence)

    # ═══ CONVENIENCE: LLM Metrics ═══
    def record_llm_call(
        self, provider: str, latency_ms: float, tokens: int = 0, success: bool = True
    ):
        """Record an LLM API call."""
        self.increment("llm.calls", provider=provider)
        if not success:
            self.increment("llm.errors", provider=provider)
        self.observe("llm.latency_ms", latency_ms, provider=provider)
        if tokens > 0:
            self.observe("llm.tokens", tokens, provider=provider)

    # ═══ CONVENIENCE: Schema Filtering Metrics ═══
    def record_schema_compression(
        self, total_columns: int, filtered_columns: int, compression_ratio: float
    ):
        """Record schema filtering results."""
        self.increment("schema.filter_calls")
        self.observe("schema.total_columns", total_columns)
        self.observe("schema.filtered_columns", filtered_columns)
        self.observe("schema.compression_ratio", compression_ratio)

    # ═══ EXPORT ═══
    def snapshot(self) -> Dict[str, Any]:
        """Get a complete snapshot of all metrics."""
        with self._lock:
            return {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "histograms": {
                    k: self.get_histogram_stats(k) for k in self._histograms
                },
            }

    def reset(self):
        """Reset all metrics (for testing)."""
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()

    @staticmethod
    def _make_key(name: str, tags: Dict[str, str]) -> str:
        """Create a composite key from name and tags."""
        if not tags:
            return name
        tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{name}{{{tag_str}}}"


# ═══ Singleton ═══
_metrics: Optional[MetricsCollector] = None
_metrics_lock = threading.Lock()


def get_metrics() -> MetricsCollector:
    """Get the global MetricsCollector singleton."""
    global _metrics
    if _metrics is None:
        with _metrics_lock:
            if _metrics is None:
                _metrics = MetricsCollector()
    return _metrics


__all__ = ["MetricsCollector", "get_metrics"]
