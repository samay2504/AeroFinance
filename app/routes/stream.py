"""Streaming query routes."""

from fastapi import APIRouter
from app.controllers.stream_controller import stream_query
from app.types.schemas import StreamQueryRequest

router = APIRouter()


@router.post("/query/stream", tags=["Query"])
async def stream_query_route(request: StreamQueryRequest):
    """Stream query response using Server-Sent Events (SSE)."""
    return await stream_query(request)
