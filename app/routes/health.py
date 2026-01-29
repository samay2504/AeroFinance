"""Health routes."""

from fastapi import APIRouter
from app.controllers.health_controller import health_check
from app.types.schemas import RootResponse

router = APIRouter()


@router.get("/health", response_model=RootResponse, tags=["Health"])
async def health_route():
    """Health check endpoint."""
    return await health_check()
