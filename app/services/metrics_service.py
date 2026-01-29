"""Metrics service."""


def get_metrics() -> dict:
    """Get system metrics."""
    try:
        from app.core.llm_wrapper import get_llm_wrapper
        from app.core.data_registry import get_data_registry

        llm = get_llm_wrapper()
        registry = get_data_registry()

        return {
            "llm": llm.get_metrics(),
            "datasets": len(registry.list_all()),
            "provider": llm.provider_name
        }

    except Exception as e:
        return {"error": str(e)}
