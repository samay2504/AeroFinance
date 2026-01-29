"""Rehydrate routes."""

from fastapi import APIRouter
from app.controllers.rehydrate_controller import rehydrate
from app.types.schemas import RootResponse, RehydrateRequest

router = APIRouter()


@router.post("/rehydrate", response_model=RootResponse, tags=["Rehydrate"])
async def rehydrate_route(request: RehydrateRequest):
    """Rehydrate datasets from disk into memory."""
    result = await rehydrate(request)
    return RootResponse(
        message="Rehydrate completed",
        data=result,
        error=None,
        status="ok",
    )
