"""Dataset routes."""

from fastapi import APIRouter, Query
from app.controllers.datasets_controller import list_datasets
from app.types.schemas import RootResponse

router = APIRouter()


@router.get("/datasets", response_model=RootResponse, tags=["Datasets"])
async def list_datasets_route(client_id: str = Query(...)):
    """List datasets for a client."""
    result = await list_datasets(client_id)
    return RootResponse(
        message="Datasets fetched",
        data=result,
        error=None,
        status="ok",
    )
