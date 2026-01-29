"""ID routes."""

from fastapi import APIRouter, Query
from app.controllers.id_controller import (
    create_id_controller,
    list_ids_controller,
    validate_id_controller,
)
from app.types.schemas import IDCreateRequest, IDValidateRequest, RootResponse

router = APIRouter()


@router.post("/api/ids/create", response_model=RootResponse, tags=["IDs"])
async def create_id_route(request: IDCreateRequest):
    result = await create_id_controller(request)
    return RootResponse(
        message="ID created",
        data=result,
        error=None if result.get("success") else result.get("error"),
        status="ok" if result.get("success") else "error",
    )


@router.get("/api/ids/list", response_model=RootResponse, tags=["IDs"])
async def list_ids_route(user_id: str = Query(...)):
    result = await list_ids_controller(user_id)
    return RootResponse(
        message="IDs fetched",
        data=result,
        error=None,
        status="ok",
    )


@router.post("/api/ids/validate", response_model=RootResponse, tags=["IDs"])
async def validate_id_route(request: IDValidateRequest):
    result = await validate_id_controller(request)
    return RootResponse(
        message="ID validated",
        data=result,
        error=None,
        status="ok",
    )
