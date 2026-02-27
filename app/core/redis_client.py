"""
Production-grade Redis / Valkey client singleton.

Cloud-friendly features:
  • Valkey support — fully API-compatible Redis fork (AWS ElastiCache Valkey,
    Valkey OSS).  Use VALKEY_URL=valkey://... (or valkeys:// for TLS).
    The valkey[s]:// scheme is transparently normalised to redis[s]:// so
    redis-py works unchanged on both servers.

  • URL priority (first non-empty wins):
      REDIS_URL → CACHE_REDIS_URL → UPSTASH_REDIS_URL
      → VALKEY_URL → VALKEY_CLOUD_URL
      → KV_URL  (Vercel KV / Railway Redis)

  • Circuit breaker — connection failures open the breaker for
    _CB_RETRY_SECONDS (default 60 s) then allow ONE probe attempt.
    A transient blip (pod restart, network hiccup) heals automatically.
    No permanent "unavailable" latching that requires process restart.

  • Graceful degradation — all callers receive None when the circuit is
    open; bloom filter / vision cache / HotPromptCache fall back to
    disk / in-memory transparently.

  • Process-level pool — 50-connection pool shared across all modules.

  • Thread-safe — double-checked locking on init; pool itself is thread-safe.

Usage:
    from app.core.redis_client import get_redis

    r = get_redis()
    if r:                           # None when unavailable / circuit open
        r.set("key", "val", ex=60)

    # Backend-aware observability:
    from app.core.redis_client import get_backend_type
    print(get_backend_type())  # "redis" | "valkey" | "none"
"""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)

# ─── Circuit-breaker parameters ─────────────────────────────────────────────
_CB_RETRY_SECONDS: int = 60          # Probe interval after failure

# ─── Singleton state ─────────────────────────────────────────────────────────
_redis_instance: Optional["redis.Redis"] = None  # type: ignore[name-defined]
_redis_lock = threading.Lock()
_cb_open_until: float = 0.0          # epoch-time after which we may retry
_no_url: bool = False                # True when no URL is configured at all
_backend_type: str = "none"          # "redis" | "valkey" | "none"


# ─── Valkey / Redis scheme normalisation ────────────────────────────────────

def _normalise_url(url: str) -> tuple[str, str]:
    """
    Convert valkey[s]:// → redis[s]:// so redis-py works unchanged.
    Returns (normalised_url, detected_backend_type).
    """
    if url.startswith("valkeys://"):
        return "rediss://" + url[len("valkeys://"):], "valkey"
    if url.startswith("valkey://"):
        return "redis://" + url[len("valkey://"):], "valkey"
    return url, "redis"


def _resolve_url() -> Optional[str]:
    """
    Return the first configured cache URL, or None.

    Resolution priority (first non-empty wins):
      1. REDIS_URL             — Upstash / any Redis cloud
      2. CACHE_REDIS_URL       — alternate Redis key
      3. UPSTASH_REDIS_URL     — explicit Upstash key
      4. VALKEY_URL            — cloud Valkey (ElastiCache Valkey, Valkey Cloud)
      5. VALKEY_CLOUD_URL      — explicit Valkey Cloud endpoint
      6. KV_URL                — Vercel KV / Railway Redis
      7. VALKEY_DOCKER_URL     — local Docker Valkey (last-resort fallback)
    """
    return (
        os.getenv("REDIS_URL")
        or os.getenv("CACHE_REDIS_URL")
        or os.getenv("UPSTASH_REDIS_URL")
        or os.getenv("VALKEY_URL")          # cloud Valkey
        or os.getenv("VALKEY_CLOUD_URL")
        or os.getenv("KV_URL")              # Vercel KV / Railway Redis
        or os.getenv("VALKEY_DOCKER_URL")   # local Docker Valkey — sub-fallback
    )


# ─── Public API ──────────────────────────────────────────────────────────────

def get_redis() -> Optional["redis.Redis"]:  # type: ignore[name-defined]
    """
    Return the process-level Redis/Valkey client, or None.

    • Returns cached instance immediately when healthy.
    • Respects circuit-breaker: returns None while breaker is open; retries
      once after _CB_RETRY_SECONDS.
    • Lazy-initialised on first call.
    """
    global _redis_instance, _cb_open_until, _no_url, _backend_type

    # Fast path — healthy instance
    if _redis_instance is not None:
        return _redis_instance

    # Fast path — no URL ever configured
    if _no_url:
        return None

    # Fast path — circuit breaker open (don't hammer a down server)
    if _cb_open_until and time.monotonic() < _cb_open_until:
        return None

    with _redis_lock:
        # Double-checked under lock
        if _redis_instance is not None:
            return _redis_instance
        if _no_url:
            return None
        if _cb_open_until and time.monotonic() < _cb_open_until:
            return None

        # ── Resolve URL ──────────────────────────────────────────────────
        raw_url = _resolve_url()
        if not raw_url:
            logger.info(
                "Redis/Valkey not configured (REDIS_URL / VALKEY_URL not set) — "
                "falling back to in-memory / disk for bloom & vision cache"
            )
            _no_url = True
            return None

        normalised_url, detected_backend = _normalise_url(raw_url)

        # ── Build client ─────────────────────────────────────────────────
        try:
            import redis as _redis_lib

            client = _redis_lib.from_url(
                normalised_url,
                decode_responses=True,          # keys/values as str
                socket_timeout=2,               # fail fast on unreachable host
                socket_connect_timeout=2,
                retry_on_timeout=False,
                max_connections=50,             # shared pool
                health_check_interval=30,
            )
            client.ping()                       # validate connectivity now

            _redis_instance = client
            _backend_type = detected_backend
            _cb_open_until = 0.0                # reset breaker
            logger.info(
                f"{detected_backend.capitalize()} connected: {_redact_url(raw_url)}"
            )
            return _redis_instance

        except ImportError:
            logger.warning("redis library not installed — pip install redis")
            _no_url = True          # won't recover without reinstall
            return None
        except Exception as exc:
            # ── Automatic localhost fallback ──────────────────────────────
            # When the primary URL fails (e.g. Upstash DNS unreachable),
            # try local Docker Redis before opening the circuit breaker.
            _LOCAL_FALLBACK = "redis://localhost:6379/0"
            if normalised_url != _LOCAL_FALLBACK:
                try:
                    import redis as _redis_lib
                    local_client = _redis_lib.from_url(
                        _LOCAL_FALLBACK,
                        decode_responses=True,
                        socket_timeout=2,
                        socket_connect_timeout=2,
                        retry_on_timeout=False,
                        max_connections=50,
                        health_check_interval=30,
                    )
                    local_client.ping()
                    _redis_instance = local_client
                    _backend_type = "redis"
                    _cb_open_until = 0.0
                    logger.info(
                        f"Primary Redis unreachable ({exc}); "
                        f"connected to local Docker Redis ({_LOCAL_FALLBACK})"
                    )
                    return _redis_instance
                except Exception:
                    pass  # local also unavailable — open breaker below

            _cb_open_until = time.monotonic() + _CB_RETRY_SECONDS
            logger.warning(
                f"Redis/Valkey unavailable ({exc}) — "
                f"circuit breaker open for {_CB_RETRY_SECONDS}s, "
                "falling back to in-memory / disk"
            )
            return None


def get_backend_type() -> str:
    """
    Return the active cache backend type.

    Returns:
        "redis"  — connected to a Redis server
        "valkey" — connected to a Valkey server
        "none"   — no cache backend available
    """
    return _backend_type


def is_redis_available() -> bool:
    """Return True only when the backend is connected and responds to PING."""
    r = get_redis()
    if r is None:
        return False
    try:
        return bool(r.ping())
    except Exception:
        return False


def reset_redis_client() -> None:
    """
    Force re-initialisation on next get_redis() call.
    Useful in tests, connection-recovery scripts, or config reloads.
    """
    global _redis_instance, _cb_open_until, _no_url, _backend_type
    with _redis_lock:
        _redis_instance = None
        _cb_open_until = 0.0
        _no_url = False
        _backend_type = "none"


# ─── Convenience helper ──────────────────────────────────────────────────────

def execute(command: str, *args, default=None):
    """
    Execute a single Redis command by name, returning `default` on any failure.

    Example:
        val = execute("get", "my:key")
        execute("set", "my:key", "val", "ex", 300)

    Never raises — safe to call without checking availability first.
    """
    r = get_redis()
    if r is None:
        return default
    try:
        return getattr(r, command)(*args)
    except Exception as exc:
        logger.debug(f"redis.{command}({args!r}) failed: {exc}")
        return default


# ─── Internal utilities ──────────────────────────────────────────────────────

def _redact_url(url: str) -> str:
    """Replace password in URL with *** for safe logging."""
    try:
        from urllib.parse import urlparse, urlunparse
        p = urlparse(url)
        if p.password:
            netloc = p.netloc.replace(f":{p.password}@", ":***@")
            return urlunparse(p._replace(netloc=netloc))
    except Exception:
        pass
    return url


__all__ = [
    "get_redis",
    "get_backend_type",
    "is_redis_available",
    "reset_redis_client",
    "execute",
]
