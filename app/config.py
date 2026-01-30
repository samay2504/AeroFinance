"""Environment-driven configuration for AI-CA system. No hardcoding - all settings from env/yaml."""

import os
import logging
from typing import List, Optional, Dict, Any, Union
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings
from pathlib import Path
import yaml

logger = logging.getLogger(__name__)

DEFAULT_LLM_PROVIDERS = [
    "groq",
    "google_genai",
    "ollama",
    "openrouter",
    "openai",
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

DATA_DIR = PROJECT_ROOT / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
CACHE_DIR = DATA_DIR / "dataframe_cache"

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

    model_config = {
        "env_prefix": "STORAGE_",
        "extra": "ignore",
    }


class DuckDBSettings(BaseSettings):
    """DuckDB SQL engine configuration."""

    memory_limit: str = Field(default="2GB")
    threads: int = Field(default=4)
    enable_progress_bar: bool = Field(default=False)

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
        if self.workers is None:
            workers_value = _first_env("FASTAPI_WORKERS")
            self.workers = int(workers_value) if workers_value else None
        return self


def load_yaml_config(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load optional YAML configuration overlay."""
    if path is None:
        path = PROJECT_ROOT / "app" / "config.yaml"
    if path.exists():
        try:
            with open(path) as f:
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
    )

# Apply YAML overlay if present
yaml_config = load_yaml_config()
if yaml_config:
    for key, value in yaml_config.items():
        if hasattr(settings, key) and isinstance(value, dict):
            sub_settings = getattr(settings, key)
            for sub_key, sub_value in value.items():
                if hasattr(sub_settings, sub_key) and sub_value is not None:
                    setattr(sub_settings, sub_key, sub_value)
