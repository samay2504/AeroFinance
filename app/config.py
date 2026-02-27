"""Environment-driven configuration for AI-CA system. No hardcoding - all settings from env/yaml."""

import os
import logging
from typing import List, Optional, Dict, Any, Union
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings
from pathlib import Path
import yaml

logger = logging.getLogger(__name__)

# Default LLM provider order - ONLY used as last resort if LLM_PROVIDER_PREFERENCE env not set
# Production: Always set LLM_PROVIDER_PREFERENCE in .env
DEFAULT_LLM_PROVIDERS = [
    "google_genai",
    "groq",
    "openrouter",
    "ollama",
    "openai",
    "huggingface",
    "fallback",
]


def _first_env(*keys: str) -> Optional[str]:
    """Return the first non-empty environment value from keys."""
    for key in keys:
        value = os.getenv(key)
        if value not in (None, ""):
            return value
    return None

# Base directories
PROJECT_ROOT = Path(__file__).parent.parent

# Load .env so Settings() sees vars when run from IDE or without sourcing shell
try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

# Derive data directories from env with sensible fallbacks
# DATA_DIR is the BASE data directory — not uploads or cache
DATA_DIR = Path(os.getenv("DATA_DIR", str(PROJECT_ROOT / "data")))
UPLOADS_DIR = Path(os.getenv("FILE_STORE_PATH", str(DATA_DIR / "uploads")))
CACHE_DIR = Path(os.getenv("STORAGE_DATAFRAME_CACHE_PATH", str(DATA_DIR / "dataframe_cache")))

# Ensure directories exist
for d in [DATA_DIR, UPLOADS_DIR, CACHE_DIR]:
    d.mkdir(parents=True, exist_ok=True)


class LLMSettings(BaseSettings):
    """LLM provider configuration."""

    # Union so env string (e.g. "groq,openai") is not parsed as JSON by pydantic-settings
    provider_preference: Union[List[str], str] = Field(default=DEFAULT_LLM_PROVIDERS)
    temperature: float = Field(default=0.1)
    max_retries: int = Field(default=3)
    retry_delay: float = Field(default=1.0)
    provider_cooldown: int = Field(default=60)

    # Ollama settings
    ollama_enabled: bool = Field(default=False)
    ollama_model: str = Field(default="llama3.2")
    ollama_auto_pull: bool = Field(default=False)
    ollama_run_on_init: bool = Field(default=False)

    # OpenRouter settings
    openrouter_enabled: bool = Field(default=False)
    openrouter_model: str = Field(default="gpt-4o-mini")

    model_config = {
        "env_prefix": "LLM_",
        "extra": "ignore",
    }

    @field_validator("provider_preference", mode="before")
    @classmethod
    def parse_list(cls, v):
        """Parse comma-separated string to list. Tolerate empty/malformed env so Settings() does not fail."""
        if v is None or v == "":
            return DEFAULT_LLM_PROVIDERS
        if isinstance(v, str):
            out = [p.strip() for p in v.split(",") if p.strip()]
            return out if out else DEFAULT_LLM_PROVIDERS
        if isinstance(v, list):
            return v
        return DEFAULT_LLM_PROVIDERS


class VectorDBSettings(BaseSettings):
    """Vector database configuration."""

    primary: Optional[str] = Field(default=None)
    qdrant_url: Optional[str] = Field(default=None)
    qdrant_collection: Optional[str] = Field(default=None)
    chroma_persist_dir: str = Field(default=str(DATA_DIR / "chroma"))
    embedding_model: str = Field(default="all-MiniLM-L6-v2")

    model_config = {
        "env_prefix": "VECTORDB_",
        "extra": "ignore",
        "validate_default": True,
    }

    @field_validator("primary", mode="before")
    @classmethod
    def _primary_from_env(cls, v):
        return v or _first_env("VECTOR_DB_TYPE")

    @field_validator("qdrant_url", mode="before")
    @classmethod
    def _qdrant_url_from_env(cls, v):
        return v or _first_env("VECTOR_DB_QDRANT_HOST")

    @field_validator("qdrant_collection", mode="before")
    @classmethod
    def _qdrant_collection_from_env(cls, v):
        return v or _first_env("QDRANT_COLLECTION")


class CacheSettings(BaseSettings):
    """Redis cache configuration."""

    redis_url: Optional[str] = Field(default=None)
    redis_enabled: bool = Field(default=False)
    ttl_query: int = Field(default=1800)  # 30 min
    ttl_embedding: int = Field(default=86400)  # 24h
    lru_max_size: int = Field(default=10)

    model_config = {
        "env_prefix": "CACHE_",
        "extra": "ignore",
        "validate_default": True,
    }

    @field_validator("redis_url", mode="before")
    @classmethod
    def _redis_url_from_env(cls, v):
        return v or _first_env("REDIS_URL")


class DeploymentSettings(BaseSettings):
    """
    Deployment environment configuration.

    Controls storage backend selection based on deployment context:
    - local: Uses local filesystem + optional Redis
    - aws: Uses S3 for DataFrames + ElastiCache for metadata

    Environment Variables:
        DEPLOYMENT_ENV: 'local' or 'aws'
        DEPLOYMENT_AWS_S3_BUCKET: S3 bucket name for production
        DEPLOYMENT_AWS_S3_PREFIX: S3 key prefix for DataFrames
        AWS_DEFAULT_REGION: AWS region (uses shared AWS config)
    """

    env: str = Field(
        default="local", description="Deployment environment: 'local' or 'aws'"
    )
    aws_s3_bucket: Optional[str] = Field(
        default=None, description="S3 bucket for production storage"
    )
    aws_s3_prefix: str = Field(default="dataframes/", description="S3 key prefix")
    temp_dir: str = Field(
        default="/tmp/ai_ca", description="Temp directory for Lambda/serverless"
    )
    enable_local_cache: bool = Field(
        default=True, description="Enable local LRU cache even in AWS mode"
    )

    model_config = {
        "env_prefix": "DEPLOYMENT_",
        "extra": "ignore",
    }

    @property
    def aws_region(self) -> str:
        """Get AWS region from shared AWS_DEFAULT_REGION env var."""
        return os.getenv("AWS_DEFAULT_REGION", "ap-south-1")

    @property
    def is_aws(self) -> bool:
        """Check if running in AWS mode."""
        return self.env.lower() == "aws" or bool(self.aws_s3_bucket)

    @property
    def is_local(self) -> bool:
        """Check if running in local mode."""
        return not self.is_aws


class StorageSettings(BaseSettings):
    """File storage configuration."""

    file_store_path: str = Field(default=str(DATA_DIR))
    dataframe_cache_path: str = Field(default=str(CACHE_DIR))
    max_lru_dataframes: int = Field(default=10)
    parquet_compression: str = Field(
        default="snappy", description="Parquet compression: snappy, gzip, zstd"
    )
    max_dataframe_size_mb: int = Field(
        default=500, description="Max DataFrame size before chunking"
    )

    # Bloom Filter Settings
    bloom_capacity: int = Field(
        default=10000, description="Estimated number of datasets"
    )
    bloom_fp_rate: float = Field(
        default=0.01, description="Target false positive rate (0.01 = 1%)"
    )

    model_config = {
        "env_prefix": "STORAGE_",
        "extra": "ignore",
    }


class DuckDBSettings(BaseSettings):
    """DuckDB SQL engine configuration with S3 integration and performance tuning."""

    memory_limit: str = Field(default="2GB")
    threads: int = Field(default=4)
    enable_progress_bar: bool = Field(default=False)

    # S3 / Cloud Integration
    enable_s3: bool = Field(
        default=False,
        description="Load httpfs extension for direct S3 Parquet queries"
    )
    s3_region: Optional[str] = Field(
        default=None,
        description="Override AWS region for DuckDB S3 access (defaults to DEPLOYMENT region)"
    )
    s3_access_key: Optional[str] = Field(
        default=None, description="Explicit S3 access key (optional)"
    )
    s3_secret_key: Optional[str] = Field(
        default=None, description="Explicit S3 secret key (optional)"
    )
    s3_endpoint: Optional[str] = Field(
        default=None,
        description="Custom S3-compatible endpoint (e.g. MinIO, LocalStack)"
    )

    # Performance Tuning
    enable_object_cache: bool = Field(
        default=True,
        description="Cache Parquet metadata in memory for faster repeated scans"
    )
    object_cache_size: str = Field(
        default="256MB",
        description="Parquet metadata cache size"
    )
    preserve_insertion_order: bool = Field(
        default=False,
        description="Disable for ~15% scan speedup on wide tables"
    )
    enable_parallel_csv: bool = Field(
        default=True,
        description="Parallel CSV reader for faster ingestion"
    )

    model_config = {
        "env_prefix": "DUCKDB_",
        "extra": "ignore",
    }


class SandboxSettings(BaseSettings):
    """Sandbox executor configuration."""

    timeout_seconds: int = Field(default=30)
    max_memory_mb: int = Field(default=512)
    pool_size: int = Field(default=max(1, os.cpu_count() - 1) if os.cpu_count() else 2)

    model_config = {
        "env_prefix": "SANDBOX_",
        "extra": "ignore",
    }


class ZMQSettings(BaseSettings):
    """ZeroMQ IPC configuration."""

    enabled: bool = Field(default=False)
    socket_path: str = Field(default="ipc:///tmp/ai_ca.sock")

    model_config = {
        "env_prefix": "ZMQ_",
        "extra": "ignore",
    }


# ═══════════════════════════════════════════════════════════════════
# FEATURE FLAG SETTINGS — PRD Enhancement Toggles
# ═══════════════════════════════════════════════════════════════════

class RouterSettings(BaseSettings):
    """Enhancement 1: Semantic Router feature flags."""

    enable_fast_pass: bool = Field(
        default=True,
        description="L0 regex fast-pass routing (0.01s latency)"
    )
    enable_semantic_routing: bool = Field(
        default=True,
        description="L1 spaCy vector similarity routing (0.05s)"
    )
    enable_llm_fallback: bool = Field(
        default=True,
        description="L2 LLM classification fallback (1.5s)"
    )
    regex_confidence: float = Field(
        default=0.95,
        description="Minimum confidence for L0 regex to accept"
    )
    semantic_confidence: float = Field(
        default=0.70,
        description="Minimum confidence for L1 semantic to accept"
    )
    cache_enabled: bool = Field(
        default=True,
        description="Cache repeated routing decisions"
    )
    cache_max_size: int = Field(
        default=1000,
        description="Max cached routing decisions"
    )

    model_config = {
        "env_prefix": "ROUTER_",
        "extra": "ignore",
    }


class HealingSettings(BaseSettings):
    """Enhancement 2: Self-Healing Execution feature flags."""

    enabled: bool = Field(
        default=True,
        description="Master switch for self-healing execution loop"
    )
    max_retries: int = Field(
        default=2,
        description="Max retry attempts before failing"
    )
    enable_error_hints: bool = Field(
        default=True,
        description="Include type-specific debugging hints in healing prompts"
    )
    enable_fuzzy_column_fix: bool = Field(
        default=True,
        description="Auto-correct typos in column names via fuzzy matching"
    )
    log_healing_prompts: bool = Field(
        default=False,
        description="Log full healing prompts (verbose, for debugging)"
    )
    backoff_base_ms: int = Field(
        default=100,
        description="Exponential backoff base delay in ms"
    )

    model_config = {
        "env_prefix": "HEALING_",
        "extra": "ignore",
    }


class SchemaSettings(BaseSettings):
    """Enhancement 3: Intelligent Schema Filtering feature flags."""

    enable_filtering: bool = Field(
        default=True,
        description="Master switch for intelligent column filtering"
    )
    max_columns: int = Field(
        default=20,
        description="Max columns to send to LLM context"
    )
    fuzzy_match_cutoff: float = Field(
        default=0.70,
        description="Fuzzy matching threshold (0.0-1.0)"
    )
    enable_semantic_expansion: bool = Field(
        default=True,
        description="Expand related columns (e.g. revenue→cost, margin)"
    )
    enable_key_column_detection: bool = Field(
        default=True,
        description="Always include date/ID/name columns"
    )
    min_fallback_columns: int = Field(
        default=5,
        description="Minimum columns when nothing matches query"
    )

    model_config = {
        "env_prefix": "SCHEMA_",
        "extra": "ignore",
    }


class HotCacheSettings(BaseSettings):
    """
    Enhancement 5: Hot Cache (L1/L2/L3) configuration.
    
    Tiers:
    - L1: Local RAM (fastest, per-worker)
    - L2: Redis (shared, persistent) 
    - L3: Semantic (embedding-based)
    """

    enabled: bool = Field(
        default=True,
        description="Master switch for caching"
    )
    max_entries: int = Field(
        default=1000,
        description="L1 Cache max entries"
    )
    ttl_seconds: int = Field(
        default=1800,
        description="Default TTL in seconds"
    )
    semantic: bool = Field(
        default=True,
        description="Enable semantic similarity matching"
    )
    similarity: float = Field(
        default=0.92,
        description="Cosine similarity threshold (0.0-1.0)"
    )
    embedding_provider: str = Field(
        default="auto",
        description="Provider for semantic embeddings"
    )
    
    # Advanced L1 settings
    hot_key_threshold: int = Field(
        default=10, description="Hits required to promote to L1"
    )
    l1_ttl: int = Field(
        default=60, description="TTL for L1 (local) entries in seconds"
    )
    
    model_config = {
        "env_prefix": "HOT_CACHE_",
        "extra": "ignore",
    }


class CompressionSettings(BaseSettings):
    """Enhancement 6: DataFrame Token Compression feature flags."""

    enabled: bool = Field(
        default=True,
        description="Master switch for DataFrame token compression"
    )
    max_sample_rows: int = Field(
        default=5,
        description="Sample rows in compressed LLM output"
    )
    include_statistics: bool = Field(
        default=True,
        description="Include column statistics in compressed output"
    )
    include_patterns: bool = Field(
        default=True,
        description="Include detected data patterns"
    )

    model_config = {
        "env_prefix": "COMPRESSION_",
        "extra": "ignore",
    }


class PDFSettings(BaseSettings):
    """PDF Ingestion & Vision Pipeline configuration."""

    enable_vision: bool = Field(
        default=True, description="Enable vision cascade for scanned/image PDFs"
    )
    vision_threshold: float = Field(
        default=50, description=(
            "Minimum characters extracted by pdfplumber for a page to be "
            "considered text-extractable. Pages with fewer chars are routed "
            "to the vision cascade (Gemini → VLM → RapidOCR → Tesseract). "
            "Set lower (e.g. 20) to be more aggressive with text extraction, "
            "higher (e.g. 200) to send more pages through vision."
        )
    )
    max_chunk_tokens: int = Field(
        default=1024, description="Max tokens per semantic chunk"
    )
    parallel_workers: int = Field(
        default=4, description="ProcessPoolExecutor worker count for page extraction"
    )
    cache_dir: str = Field(
        default="./data/pdf_cache", description="Disk cache for fingerprints, chunks, vision cache"
    )

    # dots.ocr (self-hosted 1.7B VLM) — L2 of vision cascade
    dots_ocr_enabled: bool = Field(
        default=False, description="Enable dots.ocr as L2 vision fallback"
    )
    dots_ocr_endpoint: Optional[str] = Field(
        default=None,
        description="dots.ocr inference endpoint (scheme://host:port). "
                    "Required when dots_ocr_enabled=true. Set via PDF_DOTS_OCR_ENDPOINT.",
    )
    dots_ocr_timeout: int = Field(
        default=30, description="dots.ocr HTTP request timeout in seconds"
    )
    dots_ocr_min_confidence: float = Field(
        default=0.6, description="Minimum confidence to accept dots.ocr results"
    )
    dots_ocr_auto_docker: bool = Field(
        default=False,
        description="Auto-pull and manage dots.ocr Docker container. "
                    "Detects GPU VRAM and tunes memory params to avoid OOM. "
                    "Set via PDF_DOTS_OCR_AUTO_DOCKER.",
    )
    dots_ocr_docker_image: str = Field(
        default="vllm/vllm-openai:latest",
        description="Docker image for self-hosted VLM. "
                    "Set via PDF_DOTS_OCR_DOCKER_IMAGE.",
    )
    dots_ocr_model: str = Field(
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        description="HuggingFace model ID for self-hosted VLM OCR. "
                    "Default is ungated (no HF token needed). "
                    "Alternative: rednote-hilab/dots.ocr-1.5 (gated, needs HF_TOKEN). "
                    "Set via PDF_DOTS_OCR_MODEL.",
    )

    # Google Vision + Gemini (L1 of vision cascade)
    vision_model: str = Field(
        default="gemini-2.0-flash",
        description="Gemini model for vision OCR reasoning. "
                    "Free tier: gemini-2.0-flash (15 RPM, 1M ctx). "
                    "Set via PDF_VISION_MODEL.",
    )
    vision_embedding_model: str = Field(
        default="models/text-embedding-004",
        description="Google embedding model for vision provider. "
                    "Set via PDF_VISION_EMBEDDING_MODEL.",
    )
    google_credentials_path: Optional[str] = Field(
        default=None,
        description="Path to Google Cloud service-account JSON. "
                    "Only needed for Cloud Vision OCR (L1). "
                    "Falls back to GOOGLE_APPLICATION_CREDENTIALS env var or ADC. "
                    "Set via PDF_GOOGLE_CREDENTIALS_PATH.",
    )

    # Lazy embedding
    embedding_batch_size: int = Field(
        default=16, description="Batch size for lazy embed on retrieval"
    )

    # Retrieval
    bm25_prefilter_k: int = Field(
        default=500, description="BM25 pre-filter candidate count"
    )
    vector_cap: int = Field(
        default=50, description="Max candidates passed to vector similarity stage"
    )
    enable_rerank: bool = Field(
        default=True, description="Enable L4 reranking in retrieval funnel"
    )
    rerank_model: str = Field(
        default="auto", description="Rerank strategy: auto, jina, cohere, cross_encoder, none"
    )
    vision_concurrency: int = Field(
        default=3,
        description=(
            "Maximum simultaneous cloud-vision API calls (Gemini / Cloud Vision). "
            "Prevents 429 rate-limit cascades on large multi-page PDFs. "
            "Tune to your API tier: free=2, standard=4, enterprise=8. "
            "Set via PDF_VISION_CONCURRENCY."
        ),
    )
    max_vision_pages: int = Field(
        default=20,
        description=(
            "Maximum pages to send through the cloud-vision cascade (Gemini/VLM). "
            "Pages beyond this cap fall through to local OCR only, regardless of text sparseness. "
            "Prevents runaway API costs and latency on large scanned documents (e.g. 146-page annual reports). "
            "Sorted by text sparseness — least-text pages are prioritised for vision. "
            "Set via PDF_MAX_VISION_PAGES."
        ),
    )
    vision_call_timeout: int = Field(
        default=30,
        description=(
            "Per-call timeout in seconds for cloud vision API calls (Gemini / Cloud Vision). "
            "A single stuck call beyond this limit is abandoned and the cascade falls to L2/L3. "
            "Prevents one slow Gemini response from blocking a worker thread indefinitely. "
            "Set via PDF_VISION_CALL_TIMEOUT."
        ),
    )

    model_config = {
        "env_prefix": "PDF_",
        "extra": "ignore",
    }


class LoggingSettings(BaseSettings):
    """Logging configuration."""

    level: str = Field(default="INFO")
    format: str = Field(default="json")
    dir: str = Field(default="./data/logs")
    to_s3: bool = Field(default=False)
    s3_bucket: Optional[str] = Field(default=None)
    s3_prefix: str = Field(default="logs/ai-ca")
    high_value_retention_days: int = Field(default=365)
    medium_value_retention_days: int = Field(default=14)
    low_value_retention_hours: int = Field(default=72)
    s3_batch_interval_hours: int = Field(default=24)

    model_config = {
        "env_prefix": "LOG_",
        "extra": "ignore",
    }


class Settings(BaseSettings):
    """Main application settings."""

    app_name: str = Field(default="AI-CA")
    app_description: str = Field(
        default="Agentic AI Chartered Accountant RAG System",
        description="FastAPI app description for OpenAPI docs",
    )
    app_version: str = Field(default="1.0.0", description="API version string")
    app_company: str = Field(default="Valuenaire", description="Company name for root endpoint")
    root_path: str = Field(
        default="/ai-ca",
        description="ASGI root_path — set when behind a reverse proxy",
    )
    api_prefix: str = Field(
        default="/v1/api",
        description="URL prefix for all versioned API routes",
    )
    enable_swagger: bool = Field(
        default=True,
        description="Expose /docs and /redoc OpenAPI UIs",
    )
    debug: Optional[bool] = Field(default=None)
    host: Optional[str] = Field(default=None)
    port: Optional[int] = Field(default=None)
    env: Optional[str] = Field(default=None)
    workers: Optional[int] = Field(default=None)

    # Sub-settings
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    vectordb: VectorDBSettings = Field(default_factory=VectorDBSettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    deployment: DeploymentSettings = Field(default_factory=DeploymentSettings)
    duckdb: DuckDBSettings = Field(default_factory=DuckDBSettings)
    sandbox: SandboxSettings = Field(default_factory=SandboxSettings)
    zmq: ZMQSettings = Field(default_factory=ZMQSettings)

    # Feature flags (PRD Enhancement toggles)
    router: RouterSettings = Field(default_factory=RouterSettings)
    healing: HealingSettings = Field(default_factory=HealingSettings)
    schema_filter: SchemaSettings = Field(default_factory=SchemaSettings)
    hot_cache: HotCacheSettings = Field(default_factory=HotCacheSettings)
    compression: CompressionSettings = Field(default_factory=CompressionSettings)
    pdf: PDFSettings = Field(default_factory=PDFSettings)
    
    # CORS
    enable_cors: bool = Field(default=True, description="Enable CORS middleware")
    cors_origins: List[str] = Field(
        default=["*"],
        description="Allowed CORS origins. Set CORS_ORIGINS in .env as JSON list.",
    )

    # Concurrency / Performance
    single_flight_timeout: float = Field(
        default=120.0, 
        description="SingleFlight timeout seconds"
    )

    model_config = {
        "env_prefix": "FASTAPI_",
        "env_nested_delimiter": "__",
        "extra": "ignore",
    }

    @model_validator(mode="after")
    def hydrate_from_env(self):
        """Ensure core runtime settings are sourced from env."""
        if self.host is None:
            self.host = _first_env("FASTAPI_HOST")
        if self.port is None:
            port_value = _first_env("FASTAPI_PORT")
            self.port = int(port_value) if port_value else None
        if self.debug is None:
            debug_value = _first_env("FASTAPI_DEBUG")
            if debug_value is not None:
                self.debug = str(debug_value).strip().lower() in ("1", "true", "yes", "on")
        if self.env is None:
            self.env = _first_env("FASTAPI_ENV")
        # CORS origins from env (JSON list string)
        cors_raw = _first_env("CORS_ORIGINS")
        if cors_raw and self.cors_origins == ["*"]:
            import json as _json
            try:
                parsed = _json.loads(cors_raw)
                if isinstance(parsed, list):
                    self.cors_origins = parsed
            except (_json.JSONDecodeError, TypeError):
                # Comma-separated fallback
                self.cors_origins = [o.strip() for o in cors_raw.split(",") if o.strip()]
        enable_cors_raw = _first_env("ENABLE_CORS")
        if enable_cors_raw is not None:
            self.enable_cors = str(enable_cors_raw).strip().lower() in ("1", "true", "yes", "on")
        if self.workers is None:
            workers_value = _first_env("FASTAPI_WORKERS")
            self.workers = int(workers_value) if workers_value else None
        # ENABLE_SWAGGER from env
        swagger_raw = _first_env("ENABLE_SWAGGER")
        if swagger_raw is not None:
            self.enable_swagger = str(swagger_raw).strip().lower() in ("1", "true", "yes", "on")
        # Root path / API prefix overrides
        rp = _first_env("FASTAPI_ROOT_PATH")
        if rp is not None:
            self.root_path = rp
        ap = _first_env("FASTAPI_API_PREFIX")
        if ap is not None:
            self.api_prefix = ap

        # ── Re-instantiate nested BaseSettings models ──
        # pydantic-settings env_nested_delimiter="__" shadows nested models'
        # own env_prefix, so SANDBOX_TIMEOUT_SECONDS etc. aren't read.
        # Fix: re-instantiate each nested model so it reads its own env vars.
        from pydantic_settings import BaseSettings as _BS
        for field_name, field_info in self.model_fields.items():
            field_type = field_info.annotation
            if (
                isinstance(field_type, type)
                and issubclass(field_type, _BS)
                and field_type is not type(self)
            ):
                try:
                    setattr(self, field_name, field_type())
                except Exception:
                    pass  # Keep default_factory value if env parsing fails

        return self


def load_yaml_config(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load optional YAML configuration overlay."""
    if path is None:
        path = PROJECT_ROOT / "app" / "config.yaml"
    if path.exists():
        try:
            with open(path, encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.warning(f"Failed to load config.yaml: {e}")
    return {}


# Global settings instance - wrapped to handle env parsing errors gracefully
try:
    settings = Settings()
except Exception as e:
    import logging

    # If this appears, env values were ignored and defaults (e.g. port 8000) are used.
    logging.warning("Settings failed to load from env, using defaults: %s", e)
    # Use defaults without env parsing
    settings = Settings.model_construct(
        llm=LLMSettings.model_construct(),
        vectordb=VectorDBSettings.model_construct(),
        cache=CacheSettings.model_construct(),
        storage=StorageSettings.model_construct(),
        deployment=DeploymentSettings.model_construct(),
        duckdb=DuckDBSettings.model_construct(),
        sandbox=SandboxSettings.model_construct(),
        zmq=ZMQSettings.model_construct(),
        router=RouterSettings.model_construct(),
        healing=HealingSettings.model_construct(),
        schema_filter=SchemaSettings.model_construct(),
        hot_cache=HotCacheSettings.model_construct(),
        compression=CompressionSettings.model_construct(),
        pdf=PDFSettings.model_construct(),
    )

# Apply YAML overlay if present — env vars ALWAYS take precedence over YAML
yaml_config = load_yaml_config()
if yaml_config:
    for key, value in yaml_config.items():
        if not hasattr(settings, key) or value is None:
            continue
        # Handle nested settings (dicts) — skip keys that have a live env var
        if isinstance(value, dict) and hasattr(getattr(settings, key), "model_dump"):
            sub_settings = getattr(settings, key)
            env_prefix = ""
            if hasattr(sub_settings, "model_config") and isinstance(sub_settings.model_config, dict):
                env_prefix = sub_settings.model_config.get("env_prefix", "")
            for sub_key, sub_value in value.items():
                if sub_value is None or not hasattr(sub_settings, sub_key):
                    continue
                # If the env var for this field is set, env wins — skip YAML
                env_var = f"{env_prefix}{sub_key}".upper()
                if os.getenv(env_var) is not None:
                    continue
                setattr(sub_settings, sub_key, sub_value)
        # Handle scalar values — skip if FASTAPI_{key} env var is set
        elif value is not None:
            env_var = f"FASTAPI_{key}".upper()
            if os.getenv(env_var) is not None:
                continue
            setattr(settings, key, value)
