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

# Windows DLL path fix
try:
    from app.core.dll_fix import apply_dll_fix

    apply_dll_fix()
except ImportError:
    pass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("ai-ca")


# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown."""
    logger.info("Starting AI-CA application...")

    # Initialize components
    try:
        from app.config import settings
        from app.core.llm_wrapper import get_llm_wrapper
        from app.core.data_registry import get_data_registry
        from app.ipc.zmq_bridge import get_zmq_bridge

        # Warm up LLM wrapper
        llm = get_llm_wrapper()
        logger.info(f"LLM Provider: {llm.provider_name}")

        # Initialize data registry
        registry = get_data_registry()
        logger.info(f"Data registry: {len(registry.list_all())} datasets")

        # Start ZMQ bridge if enabled
        if settings.zmq.enabled:
            bridge = get_zmq_bridge()
            bridge.start()

    except Exception as e:
        logger.error(f"Startup error: {e}")

    yield

    # Cleanup
    logger.info("Shutting down AI-CA...")
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

# Create FastAPI app
app = FastAPI(
    title="AI-CA",
    description="Agentic AI Chartered Accountant RAG System",
    version="1.0.0",
    lifespan=lifespan,
    openapi_tags=openapi_tags,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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


@app.get("/")
async def root():
    return {
        "message": "Welcome to AI-CA",
        "version": "1.0.0",
        "company": "Valuenaire",
    }


base_prefix = "/v1/ai-ca"
app.include_router(health_router, prefix=base_prefix)
app.include_router(ingest_router, prefix=base_prefix)
app.include_router(query_router, prefix=base_prefix)
app.include_router(datasets_router, prefix=base_prefix)
app.include_router(metrics_router, prefix=base_prefix)
app.include_router(rehydrate_router, prefix=base_prefix)
app.include_router(stream_router, prefix=base_prefix)


if __name__ == "__main__":
    from app.config import settings

    logger.info(
        "Binding to %s:%s (from AICA_PORT / .env)", settings.host, settings.port
    )
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
else:
    # When started as "uvicorn app.main:app", our port is ignored; uvicorn uses --port (default 8000).
    # To use port from .env, run: python -m app.main
    pass
