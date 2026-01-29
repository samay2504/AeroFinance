"""Metrics routes."""

from fastapi import APIRouter
from app.controllers.metrics_controller import get_metrics
from app.types.schemas import RootResponse

router = APIRouter()


@router.get("/metrics", response_model=RootResponse, tags=["Metrics"])
async def metrics_route():
    """Get system metrics."""
    result = await get_metrics()
    return RootResponse(
        message="Metrics fetched",
        data=result,
        error=None,
        status="ok",
    )
