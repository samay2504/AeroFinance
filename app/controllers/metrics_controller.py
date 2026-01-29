"""Metrics controller."""
from app.services.metrics_service import get_metrics as get_metrics_service


async def get_metrics() -> dict:
    """Controller for system metrics."""
    return get_metrics_service()
