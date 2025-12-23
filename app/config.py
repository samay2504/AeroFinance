"""Environment-driven configuration for AI-CA system. No hardcoding - all settings from env/yaml."""
import os
import logging
from typing import List, Optional, Dict, Any
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings
from pathlib import Path
import yaml

logger = logging.getLogger(__name__)

# Base directories
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
CACHE_DIR = DATA_DIR / "dataframe_cache"

# Ensure directories exist
for d in [DATA_DIR, UPLOADS_DIR, CACHE_DIR]:
    d.mkdir(parents=True, exist_ok=True)


class LLMSettings(BaseSettings):
    """LLM provider configuration."""
    provider_preference: List[str] = Field(
        default=["groq", "google_genai", "ollama", "openrouter", "openai", "fallback"]
    )
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
    
    @field_validator('provider_preference', mode='before')
    @classmethod
    def parse_list(cls, v):
        """Parse comma-separated string to list."""
        if isinstance(v, str):
            return [p.strip() for p in v.split(",") if p.strip()]
        return v


class VectorDBSettings(BaseSettings):
    """Vector database configuration."""
    primary: str = Field(default="qdrant")
    qdrant_url: str = Field(default="http://localhost:6333")
    qdrant_collection: str = Field(default="ai_ca_docs")
    chroma_persist_dir: str = Field(default=str(DATA_DIR / "chroma"))
    embedding_model: str = Field(default="all-MiniLM-L6-v2")
    
    model_config = {
        "env_prefix": "VECTORDB_",
        "extra": "ignore",
    }


class CacheSettings(BaseSettings):
    """Redis cache configuration."""
    redis_url: str = Field(default="redis://localhost:6379/0")
    redis_enabled: bool = Field(default=False)
    ttl_query: int = Field(default=1800)  # 30 min
    ttl_embedding: int = Field(default=86400)  # 24h
    lru_max_size: int = Field(default=10)
    
    class Config:
        env_prefix = "CACHE_"


class StorageSettings(BaseSettings):
    """File storage configuration."""
    file_store_path: str = Field(default=str(DATA_DIR))
    dataframe_cache_path: str = Field(default=str(CACHE_DIR))
    max_lru_dataframes: int = Field(default=10)
    
    class Config:
        env_prefix = "STORAGE_"


class DuckDBSettings(BaseSettings):
    """DuckDB SQL engine configuration."""
    memory_limit: str = Field(default="2GB")
    threads: int = Field(default=4)
    enable_progress_bar: bool = Field(default=False)
    
    class Config:
        env_prefix = "DUCKDB_"


class SandboxSettings(BaseSettings):
    """Sandbox executor configuration."""
    timeout_seconds: int = Field(default=30)
    max_memory_mb: int = Field(default=512)
    pool_size: int = Field(default=max(1, os.cpu_count() - 1) if os.cpu_count() else 2)
    
    class Config:
        env_prefix = "SANDBOX_"


class ZMQSettings(BaseSettings):
    """ZeroMQ IPC configuration."""
    enabled: bool = Field(default=False)
    socket_path: str = Field(default="ipc:///tmp/ai_ca.sock")
    
    class Config:
        env_prefix = "ZMQ_"


class Settings(BaseSettings):
    """Main application settings."""
    app_name: str = Field(default="AI-CA")
    debug: bool = Field(default=False)
    log_level: str = Field(default="INFO")
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    
    # Sub-settings
    llm: LLMSettings = Field(default_factory=LLMSettings)
    vectordb: VectorDBSettings = Field(default_factory=VectorDBSettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    duckdb: DuckDBSettings = Field(default_factory=DuckDBSettings)
    sandbox: SandboxSettings = Field(default_factory=SandboxSettings)
    zmq: ZMQSettings = Field(default_factory=ZMQSettings)
    
    model_config = {
        "env_prefix": "AICA_",
        "env_nested_delimiter": "__",
        "extra": "ignore",
    }


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
    logging.warning(f"Settings initialization warning (using defaults): {e}")
    # Use defaults without env parsing
    settings = Settings.model_construct(
        llm=LLMSettings.model_construct(),
        vectordb=VectorDBSettings.model_construct(),
        cache=CacheSettings.model_construct(),
        storage=StorageSettings.model_construct(),
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
                if hasattr(sub_settings, sub_key):
                    setattr(sub_settings, sub_key, sub_value)
