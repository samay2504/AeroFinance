"""Query routes."""

from fastapi import APIRouter
from app.controllers.query_controller import query
from app.types.schemas import QueryRequest, RootResponse

router = APIRouter()


@router.post("/query", response_model=RootResponse, tags=["Query"])
async def query_route(request: QueryRequest):
    """Execute analytical query with unique query ID for audit trail."""
    result = await query(request)
    return RootResponse(
        message="Query completed",
        data=result,
        error=None,
        status="ok",
    )
