"""
LLM Wrapper - Orchestrates LLM calls with caching, retry logic, structured output, and metrics.
Integrates with llm_provider for multi-provider orchestration.
"""
import logging
import time
from typing import Any, Dict, Optional, Union
import json
import re
import hashlib

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from langchain_core.prompts import PromptTemplate

logger = logging.getLogger(__name__)

RETRYABLE_EXCEPTIONS = (ConnectionError, TimeoutError, Exception)


class LLMWrapper:
    """High-level wrapper around LLM providers with caching and structured output."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._failed_providers: set = set()
        self._call_count = 0
        self._total_latency_ms = 0
        self._cache_hits = 0

        # Initialize LLM provider
        try:
            from .llm_provider import LLMProvider
            self._llm_provider = LLMProvider(config)
            self.llm = self._llm_provider.llm
            self.provider_name = self._llm_provider.current_provider
        except Exception as e:
            logger.error(f"LLM provider initialization failed: {e}")
            self._llm_provider = None
            self.llm = None
            self.provider_name = "fallback"

        # Initialize cache (Redis or in-memory)
        self._cache: Dict[str, Any] = {}
        self._cache_enabled = config.get("cache_enabled", True)
        self._redis_client = None
        
        try:
            if config.get("redis_enabled"):
                import redis
                redis_url = config.get("redis_url", "redis://localhost:6379/0")
                self._redis_client = redis.from_url(redis_url)
                self._redis_client.ping()
                logger.info("Redis cache connected")
        except Exception as e:
            logger.warning(f"Redis unavailable, using in-memory cache: {e}")

    def _cache_key(self, prefix: str, **kwargs) -> str:
        """Generate cache key from parameters."""
        key_data = json.dumps(kwargs, sort_keys=True, default=str)
        hash_val = hashlib.md5(key_data.encode()).hexdigest()[:16]
        return f"{prefix}:{hash_val}"

    def _get_cached(self, key: str) -> Optional[str]:
        """Get from cache."""
        if not self._cache_enabled:
            return None
        if self._redis_client:
            try:
                val = self._redis_client.get(key)
                return val.decode() if val else None
            except Exception:
                pass
        return self._cache.get(key)

    def _set_cached(self, key: str, value: str, ttl: int = 1800):
        """Set in cache."""
        if not self._cache_enabled:
            return
        if self._redis_client:
            try:
                self._redis_client.setex(key, ttl, value)
                return
            except Exception:
                pass
        self._cache[key] = value

    def _attempt_provider_fallback(self) -> bool:
        """Attempt to switch to fallback provider."""
        if not self._llm_provider:
            return False

        current = self.provider_name
        if current:
            base = str(current).split("_")[0]
            self._failed_providers.add(base)

        provider_preference = self.config.get(
            "provider_preference",
            ["groq", "google_genai", "ollama", "openrouter", "openai", "fallback"]
        )

        for provider_name in provider_preference:
            if provider_name in self._failed_providers:
                continue
            if current and provider_name in str(current):
                continue

            try:
                from .llm_provider import LLMProvider
                fallback_config = self.config.copy()
                fallback_config["provider_preference"] = [provider_name]
                fallback_config["strict_provider_list"] = True
                fallback_config["provider_blacklist"] = list(self._failed_providers)

                new_provider = LLMProvider(fallback_config)
                if new_provider.llm and new_provider.current_provider != "fallback":
                    self._llm_provider = new_provider
                    self.llm = new_provider.llm
                    self.provider_name = new_provider.current_provider
                    logger.info(f"✅ Switched to: {self.provider_name}")
                    return True
                else:
                    self._failed_providers.add(provider_name)
            except Exception as e:
                logger.warning(f"Fallback to {provider_name} failed: {e}")
                self._failed_providers.add(provider_name)

        return False

    def get_provider_info(self) -> Dict[str, Any]:
        """Get current provider info."""
        if self._llm_provider:
            return self._llm_provider.get_provider_info()
        return {"provider": "unavailable", "available": False, "fallback_mode": True}

    def _format_prompt(self, prompt: Union[str, PromptTemplate], **kwargs) -> str:
        """Safely format prompt with template variables."""
        if isinstance(prompt, PromptTemplate):
            try:
                return prompt.format(**kwargs)
            except (KeyError, ValueError):
                template_text = getattr(prompt, "template", str(prompt))
                for key, value in kwargs.items():
                    template_text = template_text.replace(f"{{{key}}}", str(value))
                return template_text
        return str(prompt)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
    )
    def invoke(self, prompt: Union[str, PromptTemplate], use_cache: bool = True, **kwargs) -> str:
        """Invoke LLM with prompt."""
        if not self.llm:
            return "Error: LLM provider unavailable"

        try:
            formatted_prompt = self._format_prompt(prompt, **kwargs)

            # Check cache
            if use_cache:
                cache_key = self._cache_key("llm", prompt=formatted_prompt, provider=self.provider_name)
                cached = self._get_cached(cache_key)
                if cached:
                    self._cache_hits += 1
                    return cached

            # Invoke LLM
            start_time = time.time()
            response = self.llm.invoke(formatted_prompt)
            elapsed_ms = (time.time() - start_time) * 1000

            self._call_count += 1
            self._total_latency_ms += elapsed_ms

            # Extract content
            if hasattr(response, "content"):
                result = response.content
            elif isinstance(response, str):
                result = response
            else:
                result = str(response)

            # Cache result
            if use_cache:
                self._set_cached(cache_key, result)

            logger.debug(f"LLM invoke: {elapsed_ms:.0f}ms ({self.provider_name})")
            return result

        except Exception as e:
            error_str = str(e)
            is_quota_error = any(
                x in error_str.lower()
                for x in ["429", "quota", "rate", "exhausted", "exceeded"]
            )

            if is_quota_error:
                logger.warning(f"⚠️ Rate limit: {error_str[:100]}")
                if self._attempt_provider_fallback():
                    return self.invoke(prompt, use_cache=use_cache, **kwargs)
                return "Error: Rate limit exceeded, no fallback available"

            logger.error(f"LLM invoke failed: {e}")
            return f"Error: {str(e)}"

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
    )
    def invoke_with_structured_output(
        self, prompt: Union[str, PromptTemplate], output_schema: Dict[str, Any], **kwargs
    ) -> Dict[str, Any]:
        """Invoke LLM expecting structured JSON output."""
        if not self.llm:
            return {"error": "LLM provider unavailable"}

        try:
            formatted_prompt = self._format_prompt(prompt, **kwargs)

            # Add JSON instruction
            def serialize_type(o):
                if isinstance(o, type):
                    return o.__name__
                return str(o)

            try:
                schema_str = json.dumps(output_schema, indent=2, default=serialize_type)
            except Exception:
                schema_str = str(output_schema)

            json_instruction = (
                f"\n\nRESPOND WITH ONLY VALID JSON matching this schema "
                f"(no markdown, no code blocks):\n{schema_str}"
            )
            if "json" not in formatted_prompt.lower():
                formatted_prompt += json_instruction

            # Invoke
            start_time = time.time()
            response = self.llm.invoke(formatted_prompt)
            elapsed_ms = (time.time() - start_time) * 1000

            self._call_count += 1
            self._total_latency_ms += elapsed_ms

            # Extract content
            if hasattr(response, "content"):
                content = response.content
            else:
                content = str(response)

            # Clean markdown code blocks
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]
            elif content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            if not content:
                return {"error": "Empty response from LLM", "fallback": True}

            # Parse JSON
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                # Try to extract JSON from text
                json_match = re.search(r"\{(?:[^{}]|(?:\{[^{}]*\}))*\}", content, re.DOTALL)
                if json_match:
                    try:
                        return json.loads(json_match.group(0))
                    except json.JSONDecodeError:
                        pass

                # Try Python dict literal
                try:
                    import ast
                    result = ast.literal_eval(content)
                    if isinstance(result, dict):
                        return result
                except Exception:
                    pass

                # Extract Python code block
                code_match = re.search(r"```python\s*(.*?)\s*```", content, re.DOTALL)
                if code_match:
                    return {
                        "code": code_match.group(1).strip(),
                        "explanation": "Extracted from code block"
                    }

                return {"error": "Invalid JSON", "raw_content": content[:500], "fallback": True}

        except Exception as e:
            error_str = str(e)
            is_quota_error = any(
                x in error_str.lower()
                for x in ["429", "quota", "rate", "exhausted", "exceeded"]
            )

            if is_quota_error:
                if self._attempt_provider_fallback():
                    return self.invoke_with_structured_output(prompt, output_schema, **kwargs)
                return {"error": "Rate limit exceeded", "fallback": True}

            logger.error(f"Structured invoke failed: {e}")
            return {"error": str(e), "fallback": True}

    def get_metrics(self) -> Dict[str, Any]:
        """Get telemetry metrics."""
        avg_latency = self._total_latency_ms / max(1, self._call_count)
        return {
            "provider": self.provider_name,
            "total_calls": self._call_count,
            "avg_latency_ms": round(avg_latency, 2),
            "cache_hits": self._cache_hits,
        }

    def reset_metrics(self):
        """Reset telemetry counters."""
        self._call_count = 0
        self._total_latency_ms = 0
        self._cache_hits = 0


# Singleton instance
_llm_wrapper: Optional[LLMWrapper] = None


def get_llm_wrapper(config: Optional[Dict[str, Any]] = None) -> LLMWrapper:
    """Get or create singleton LLM wrapper."""
    global _llm_wrapper

    if _llm_wrapper is None:
        if config is None:
            from app.config import settings
            config = {
                "provider_preference": settings.llm.provider_preference,
                "temperature": settings.llm.temperature,
                "max_retries": settings.llm.max_retries,
                "redis_enabled": settings.cache.redis_enabled,
                "redis_url": settings.cache.redis_url,
            }
        _llm_wrapper = LLMWrapper(config)

    return _llm_wrapper


__all__ = ["LLMWrapper", "get_llm_wrapper"]
