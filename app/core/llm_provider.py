"""
LLM Provider - Production-grade multi-provider LLM orchestration.
Supports: Google Gemini, Groq, OpenAI, Ollama, OpenRouter.
Includes intelligent fallback, retry logic, and metrics tracking.
"""
import os
import logging
import time
from typing import Optional, Dict, Any, Union
import subprocess
import shutil
import atexit
import json
import sys

# Prevent transformers from loading torch which causes DLL issues on Windows
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'

try:
    import requests
    REQUESTS_AVAILABLE = True
except Exception:
    REQUESTS_AVAILABLE = False

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Import all possible LLM providers
# Use Exception instead of ImportError to catch DLL loading errors on Windows

# HuggingFace imports commented out to avoid torch DLL dependency
# Uncomment if HuggingFace provider is needed
# try:
#     from langchain_huggingface import HuggingFaceEndpoint
#     HUGGINGFACE_AVAILABLE = True
# except Exception:
#     HUGGINGFACE_AVAILABLE = False
HUGGINGFACE_AVAILABLE = False

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    GOOGLE_GENAI_AVAILABLE = True
except Exception:
    GOOGLE_GENAI_AVAILABLE = False

try:
    from langchain_openai import ChatOpenAI
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

try:
    from langchain_groq import ChatGroq
    GROQ_AVAILABLE = True
except Exception:
    GROQ_AVAILABLE = False

# HuggingFace Hub API also commented out
# try:
#     from huggingface_hub import HfApi
#     HF_API_AVAILABLE = True
# except Exception:
#     HF_API_AVAILABLE = False
HF_API_AVAILABLE = False

# Attempt LangChain Ollama (community package)
try:
    from langchain_community.chat_models import ChatOllama
    OLLAMA_LANGCHAIN_AVAILABLE = True
except Exception:
    OLLAMA_LANGCHAIN_AVAILABLE = False

logger = logging.getLogger(__name__)


class LLMProvider:
    """
    Production-grade LLM provider with:
    - Intelligent multi-provider fallback chain
    - Automatic retry with exponential backoff
    - Performance metrics tracking
    - Circuit breaker pattern
    - Ollama auto-pull capability
    - OpenRouter HTTP fallback
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.current_provider: Optional[str] = None
        self.llm = None
        self._cooldowns: Dict[str, float] = {}
        self._ollama_process = None

        # Metrics tracking
        self._total_invocations = 0
        self._failed_invocations = 0
        self._total_latency = 0.0
        self._last_error: Optional[str] = None

        # Retry configuration
        self.max_retries = config.get("max_retries", 3)
        self.retry_delay = config.get("retry_delay", 1.0)

        # Circuit breaker
        self._circuit_open = False
        self._circuit_failures = 0
        self._circuit_threshold = 5
        self._circuit_reset_time = 0
        self._circuit_timeout = 60

        self._setup_llm()

    def _setup_llm(self):
        """Setup LLM with fallback chain based on configuration."""
        provider_preference = self.config.get(
            "provider_preference",
            ["google_genai", "groq", "openrouter", "ollama", "openai", "huggingface", "fallback"]
        )

        provider_map = {
            "huggingface": ("HuggingFace", self._try_huggingface),
            "google_genai": ("Google Gemini", self._try_google_genai),
            "groq": ("Groq", self._try_groq),
            "ollama": ("Ollama", self._try_ollama),
            "openrouter": ("OpenRouter", self._try_openrouter),
            "openai": ("OpenAI", self._try_openai),
            "fallback": ("Fallback", self._create_fallback_llm),
        }

        provider_blacklist = set(self.config.get("provider_blacklist") or [])
        providers = []
        
        for key in provider_preference:
            if key in provider_map and key not in provider_blacklist:
                display, fn = provider_map[key]
                providers.append((key, display, fn))

        if not self.config.get("strict_provider_list"):
            for key, (display, fn) in provider_map.items():
                if key not in provider_preference and key not in provider_blacklist:
                    providers.append((key, display, fn))

        failed_providers = []

        for key, display_name, provider_func in providers:
            cooldown_end = self._cooldowns.get(key)
            if cooldown_end and time.time() < cooldown_end:
                logger.debug(f"Skipping {display_name} - in cooldown")
                continue

            try:
                logger.info(f"Testing {display_name} provider...")
                self.llm = provider_func()
                if self.llm:
                    model_name = getattr(self.llm, "model_name", getattr(self.llm, "model", "unknown"))
                    self.current_provider = key
                    logger.info(f"✅ Initialized {display_name} with model: {model_name}")
                    return
            except Exception as e:
                error_msg = str(e)
                if any(x in error_msg.lower() for x in ["quota", "429", "rate", "resource_exhausted"]):
                    cooldown_seconds = int(self.config.get("provider_cooldown", 60))
                    self.mark_provider_failure(key, "quota", cooldown_seconds)
                failed_providers.append(display_name)
                logger.warning(f"⚠️ {display_name}: {error_msg[:100]}")
                continue

        logger.warning(f"⚠️ All providers failed, using fallback mode")
        self.llm = self._create_fallback_llm()

    def _try_huggingface(self):
        """Initialize HuggingFace LLM."""
        if not HUGGINGFACE_AVAILABLE:
            raise ImportError("langchain_huggingface not available")

        api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if not api_key:
            raise ValueError("HUGGINGFACEHUB_API_TOKEN not set")

        models_to_try = ["microsoft/DialoGPT-medium", "gpt2"]
        temperature = self.config.get("temperature", 0.1)

        for model_name in models_to_try:
            try:
                llm = HuggingFaceEndpoint(
                    repo_id=model_name,
                    huggingfacehub_api_token=api_key,
                    task="text-generation",
                    temperature=temperature,
                )
                test_response = llm.invoke("Test")
                if test_response:
                    return llm
            except Exception:
                continue
        raise ValueError("All HuggingFace models failed")

    def _try_google_genai(self):
        """Initialize Google Gemini LLM."""
        if not GOOGLE_GENAI_AVAILABLE:
            raise ImportError("langchain_google_genai not available")

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not set")

        model = self.config.get("google_model", "gemini-2.5-flash")
        
        try:
            llm = ChatGoogleGenerativeAI(
                model=model,
                google_api_key=api_key,
                temperature=self.config.get("temperature", 0.1),
                max_retries=0,
            )
            test_response = llm.invoke("Hi")
            if test_response:
                return llm
        except Exception as e:
            error_str = str(e)
            if any(x in error_str for x in ["429", "quota", "rate", "RESOURCE_EXHAUSTED"]):
                raise ValueError("Google Gemini quota exceeded")
            raise

    def _try_openai(self):
        """Initialize OpenAI LLM."""
        if not OPENAI_AVAILABLE:
            raise ImportError("langchain_openai not available")

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")

        models_to_try = ["gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"]

        for model in models_to_try:
            try:
                # max_retries=0 to fail fast - our wrapper handles retries
                llm = ChatOpenAI(
                    model=model,
                    openai_api_key=api_key,
                    temperature=self.config.get("temperature", 0.1),
                    max_retries=0,
                )
                test_response = llm.invoke("Test")
                if test_response:
                    return llm
            except Exception:
                continue
        raise ValueError("All OpenAI models failed")

    def _try_groq(self):
        """Initialize Groq LLM."""
        if not GROQ_AVAILABLE:
            raise ImportError("langchain_groq not available")

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set")

        models_to_try = [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768",
        ]

        for model in models_to_try:
            try:
                # CRITICAL: max_retries=0 to fail fast and let our fallback handle it
                # This prevents Groq's internal exponential backoff (4s, 6s, 11s, 12s delays)
                llm = ChatGroq(
                    model=model,
                    groq_api_key=api_key,
                    temperature=self.config.get("temperature", 0.1),
                    max_retries=0,  # Fail fast - our wrapper handles retries/fallback
                )
                test_response = llm.invoke("Test")
                if test_response:
                    logger.info(f"✅ Groq model {model} initialized")
                    return llm
            except Exception:
                continue
        raise ValueError("All Groq models failed")

    def _try_ollama(self):
        """Initialize Ollama provider with auto-pull and CLI fallback."""
        enabled = os.getenv("OLLAMA_ENABLED", "").lower() in ("true", "1", "yes")
        if not enabled and not self.config.get("ollama_enabled"):
            raise ImportError("Ollama not enabled")

        model = os.getenv("OLLAMA_MODEL") or self.config.get("ollama_model", "llama3.2")

        # Try LangChain Ollama first
        if OLLAMA_LANGCHAIN_AVAILABLE:
            try:
                llm = ChatOllama(model=model)
                test = llm.invoke("Test")
                if test:
                    logger.info(f"✅ Ollama (LangChain) initialized with {model}")
                    return llm
            except Exception as e:
                logger.warning(f"LangChain Ollama failed: {e}")

        # CLI fallback
        if not shutil.which("ollama"):
            raise ValueError("Ollama binary not found")

        # Auto-pull if configured
        auto_pull = os.getenv("AUTO_PULL_MODE", "").lower() in ("true", "1")
        if auto_pull or self.config.get("ollama_auto_pull"):
            try:
                logger.info(f"Pulling Ollama model: {model}")
                subprocess.run(["ollama", "pull", model], check=False, timeout=300)
            except Exception as e:
                logger.warning(f"Ollama pull failed: {e}")

        # Create CLI wrapper
        class OllamaCLIWrapper:
            def __init__(self, model_name: str):
                self.model = model_name
                self.model_name = model_name

            def invoke(self, prompt, **kwargs):
                try:
                    result = subprocess.run(
                        ["ollama", "run", self.model, str(prompt)],
                        capture_output=True, text=True, timeout=120, encoding="utf-8"
                    )
                    return result.stdout.strip() or result.stderr.strip()
                except Exception as e:
                    raise RuntimeError(f"Ollama CLI failed: {e}")

        try:
            cli = OllamaCLIWrapper(model)
            test = cli.invoke("Test")
            if test:
                logger.info(f"✅ Ollama CLI initialized with {model}")
                return cli
        except Exception as e:
            logger.warning(f"Ollama CLI failed: {e}")

        raise ValueError("Ollama initialization failed")

    def _try_openrouter(self):
        """Initialize OpenRouter with HTTP fallback."""
        # Auto-enable if API key is present (production-friendly)
        api_key = os.getenv("OPENROUTER_API_KEY") or self.config.get("openrouter_api_key")
        enabled = (
            os.getenv("OPENROUTER_ENABLED", "").lower() in ("true", "1", "yes") or
            self.config.get("openrouter_enabled") or
            bool(api_key)  # Auto-enable if API key exists
        )
        
        if not enabled:
            raise ImportError("OpenRouter not enabled and no API key found")

        # Use reliable free model: openai/gpt-5.2-codex
        model = os.getenv("OPENROUTER_MODEL") or self.config.get("openrouter_model", "openai/gpt-5.2-codex")

        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not set")

        if not REQUESTS_AVAILABLE:
            raise ImportError("requests required for OpenRouter")

        class OpenRouterHTTPWrapper:
            def __init__(self, api_key: str, model_name: str):
                self.api_key = api_key
                self.model = model_name
                self.model_name = model_name
                self.endpoint = "https://openrouter.ai/api/v1/chat/completions"

            def invoke(self, prompt, **kwargs):
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/valuenaire/AI-ML-Pipeline", # Required by OpenRouter for free tier
                    "X-Title": "AI CA Agent"
                }
                payload = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": str(prompt)}],
                    "max_tokens": kwargs.get("max_tokens", 1024),
                }
                try:
                    r = requests.post(self.endpoint, headers=headers, json=payload, timeout=60)
                    r.raise_for_status()
                    data = r.json()
                    if "choices" in data and data["choices"]:
                        msg = data["choices"][0].get("message", {})
                        return msg.get("content", json.dumps(data))
                    return json.dumps(data)
                except Exception as e:
                    if 'r' in locals() and hasattr(r, 'text'):
                        logger.error(f"OpenRouter Error ({self.model}): {r.text}")
                    raise RuntimeError(f"OpenRouter request failed: {e}")

        # Try models in sequence
        models_to_try = [model]
        # Add fallbacks for robustness if using free tier
        if "free" in model or "exp" in model or "codex" in model:
            models_to_try.extend([
                "openai/gpt-5.2-codex",
                "meta-llama/llama-3.2-11b-vision-instruct:free",
                "mistralai/mistral-7b-instruct:free",
                "microsoft/phi-3-mini-128k-instruct:free"
            ])
        
        # Deduplicate preserving order
        models_to_try = list(dict.fromkeys(models_to_try))

        for m in models_to_try:
            try:
                # logger.info(f"Trying OpenRouter model: {m}")
                wrapper = OpenRouterHTTPWrapper(api_key, m)
                test = wrapper.invoke("Test")
                if test:
                    logger.info(f"✅ OpenRouter initialized with {m}")
                    return wrapper
            except Exception as e:
                logger.warning(f"OpenRouter model {m} failed: {str(e)[:100]}")
                continue

        raise ValueError("OpenRouter initialization failed (all models)")

    def _create_fallback_llm(self):
        """Create fallback LLM when all providers fail."""
        class FallbackLLM:
            def __init__(self):
                self.model_name = "fallback_mode"

            def invoke(self, prompt, **kwargs):
                logger.warning("Using fallback LLM - providers unavailable")
                return {
                    "content": "LLM services unavailable. Please configure API keys."
                }

        self.current_provider = "fallback"
        return FallbackLLM()

    def invoke(self, prompt: str, **kwargs) -> Union[str, Dict[str, Any], None]:
        """Invoke LLM with retry logic and circuit breaker."""
        if self._circuit_open:
            if time.time() < self._circuit_reset_time:
                logger.warning("Circuit breaker open")
                return None
            self._circuit_open = False
            self._circuit_failures = 0

        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                self._total_invocations += 1
                response = self.llm.invoke(prompt, **kwargs)
                self._total_latency += time.time() - start_time
                self._circuit_failures = max(0, self._circuit_failures - 1)

                if isinstance(response, dict):
                    return response.get("content") or response.get("text") or str(response)
                elif isinstance(response, str):
                    return response
                elif hasattr(response, "content"):
                    return response.content
                return str(response)

            except Exception as e:
                self._failed_invocations += 1
                self._last_error = str(e)
                self._circuit_failures += 1

                if self._circuit_failures >= self._circuit_threshold:
                    self._circuit_open = True
                    self._circuit_reset_time = time.time() + self._circuit_timeout
                    logger.error(f"Circuit breaker opened for {self._circuit_timeout}s")

                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)
                    logger.warning(f"Retry {attempt + 1}/{self.max_retries}: {e}")
                    time.sleep(delay)
                else:
                    logger.error(f"Failed after {self.max_retries} attempts: {e}")
                    return None

        return None

    @property
    def name(self) -> str:
        """Get current provider name."""
        return getattr(self.llm, "name", self.current_provider or "unknown")

    def get_provider_info(self) -> Dict[str, Any]:
        """Get provider information."""
        return {
            "provider": self.current_provider,
            "available": self.llm is not None,
            "fallback_mode": self.current_provider == "fallback",
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        avg_latency = self._total_latency / max(1, self._total_invocations)
        failure_rate = self._failed_invocations / max(1, self._total_invocations) * 100

        return {
            "provider": self.current_provider,
            "total_invocations": self._total_invocations,
            "failed_invocations": self._failed_invocations,
            "failure_rate_pct": round(failure_rate, 2),
            "avg_latency_ms": round(avg_latency * 1000, 2),
            "circuit_open": self._circuit_open,
            "last_error": self._last_error,
        }

    def mark_provider_failure(self, provider_name: str, error_type: str = None, cooldown_seconds: int = 60):
        """Mark provider as failed with cooldown."""
        self._cooldowns[provider_name] = time.time() + cooldown_seconds
        logger.warning(f"Provider {provider_name} cooldown: {cooldown_seconds}s ({error_type})")


def create_llm_provider(config: Dict[str, Any]) -> LLMProvider:
    """Factory function to create LLM provider."""
    return LLMProvider(config)
