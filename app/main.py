"""
AI-CA Main Application - FastAPI entrypoint.
Production-grade Agentic AI Chartered Accountant RAG System.
"""

import sys
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from app.config import settings as _settings

# Windows DLL path fix
try:
    from app.core.dll_fix import apply_dll_fix

    apply_dll_fix()
except ImportError:
    pass

# Configure logging — level from .env LOG_LEVEL
_log_level = getattr(logging, _settings.logging.level.upper(), logging.INFO)
logging.basicConfig(
    level=_log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("ai-ca")


# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown.

    Heavy singletons (LLM, data-registry, ZMQ) are initialised in a
    background asyncio Task so the server begins accepting HTTP requests
    immediately — matching the ChatGPT/Claude upload-speed experience where
    the API is ready the instant the process starts.
    """
    import asyncio

    logger.info("Starting AI-CA application…")

    async def _warm_up():
        try:
            from app.config import settings
            from app.core.llm_wrapper import get_llm_wrapper
            from app.core.data_registry import get_data_registry
            from app.ipc.zmq_bridge import get_zmq_bridge

            llm = get_llm_wrapper()
            logger.info(f"LLM Provider: {llm.provider_name}")

            registry = get_data_registry()
            logger.info(f"Data registry: {len(registry.list_all())} datasets")

            if settings.zmq.enabled:
                bridge = get_zmq_bridge()
                bridge.start()

        except Exception as e:
            logger.error(f"Warm-up error (non-fatal): {e}")

    # Start in background — server is ready immediately
    asyncio.create_task(_warm_up())

    yield

    # Cleanup
    logger.info("Shutting down AI-CA…")
    try:
        from app.ipc.zmq_bridge import get_zmq_bridge

        bridge = get_zmq_bridge()
        bridge.stop()
    except Exception:
        pass


# OpenAPI tags for Swagger sections
openapi_tags = [
    {"name": "Health", "description": "Service health checks"},
    {"name": "Ingest", "description": "Upload and ingest data"},
    {"name": "Query", "description": "Query ingested data"},
    {"name": "Datasets", "description": "List datasets"},
    {"name": "Metrics", "description": "Service metrics"},
    {"name": "Rehydrate", "description": "Load cached datasets into memory"},
    {"name": "IDs", "description": "ID utilities"},
]

# Create FastAPI app — all values from .env / settings
app = FastAPI(
    title=_settings.app_name,
    description=_settings.app_description,
    version=_settings.app_version,
    root_path=_settings.root_path,
    lifespan=lifespan,
    openapi_tags=openapi_tags,
    docs_url="/docs" if _settings.enable_swagger else None,
    redoc_url="/redoc" if _settings.enable_swagger else None,
    openapi_url="/openapi.json" if _settings.enable_swagger else None,
)

# CORS middleware — origins from .env CORS_ORIGINS (falls back to ["*"])
if _settings.enable_cors:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Routers
from app.routes.health import router as health_router
from app.routes.ingest import router as ingest_router
from app.routes.query import router as query_router
from app.routes.datasets import router as datasets_router
from app.routes.metrics import router as metrics_router
from app.routes.rehydrate import router as rehydrate_router
from app.routes.stream import router as stream_router
from app.routes.ids import router as ids_router


@app.get("/")
async def root():
    return {
        "message": f"Welcome to {_settings.app_name}",
        "version": _settings.app_version,
        "company": _settings.app_company,
    }


base_prefix = _settings.api_prefix
app.include_router(health_router, prefix=base_prefix)
app.include_router(ingest_router, prefix=base_prefix)
app.include_router(query_router, prefix=base_prefix)
app.include_router(datasets_router, prefix=base_prefix)
app.include_router(metrics_router, prefix=base_prefix)
app.include_router(rehydrate_router, prefix=base_prefix)
app.include_router(stream_router, prefix=base_prefix)
# IDs router uses fully-qualified paths (e.g. /api/ids/create) — mount at root
app.include_router(ids_router)


if __name__ == "__main__":
    from app.config import settings

    logger.info(
        "Starting AI-CA on %s:%s (from FASTAPI_HOST/FASTAPI_PORT in .env)",
        settings.host,
        settings.port,
    )
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.logging.level.lower(),
    )
else:
    # When started as "uvicorn app.main:app", uvicorn CLI args override .env
    # For .env-driven setup with settings, use: python -m app.main or python serve.py
    # For uvicorn CLI: uvicorn app.main:app --host 0.0.0.0 --port 9999 --reload
    pass
