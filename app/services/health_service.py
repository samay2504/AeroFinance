"""Health check service."""

import logging
from app.types.health import HealthComponents

logger = logging.getLogger("ai-ca")


def get_health() -> HealthComponents:
    """Gather component health status."""
    components = HealthComponents(
        api="ok",
        sql_engine="unknown",
        vector_db="unknown",
        llm="unknown",
    )

    try:
        from app.sql_engine import get_sql_engine

        engine = get_sql_engine()
        components.sql_engine = "ok" if engine else "unavailable"
    except Exception as e:
        components.sql_engine = "error"
        logger.error(f"SQL engine unavailable: {e}")

    try:
        from app.rag.ingest import get_document_ingestor

        ingestor = get_document_ingestor()
        components.vector_db = ingestor._active_store or "unavailable"
    except Exception as e:
        logger.error(f"Vector database unavailable: {e}")
        components.vector_db = "error"

    try:
        from app.core.llm_wrapper import get_llm_wrapper

        llm = get_llm_wrapper()
        components.llm = llm.provider_name or "fallback"
    except Exception as e:
        logger.error(f"LLM unavailable: {e}")
        components.llm = "error"

    return components
