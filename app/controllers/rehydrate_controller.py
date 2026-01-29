"""Rehydrate controller."""

from app.types.schemas import RehydrateRequest
from app.services.rehydrate_service import rehydrate_datasets


async def rehydrate(request: RehydrateRequest) -> dict:
    """Controller for dataset rehydration."""
    return rehydrate_datasets(
        client_id=request.client_id,
        dataset_ids=request.dataset_ids,
    )
