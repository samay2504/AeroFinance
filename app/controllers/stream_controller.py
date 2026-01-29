"""Streaming query controller."""

from fastapi.responses import StreamingResponse
from app.services.stream_service import stream_query_events


async def stream_query(request):
    """Controller for streaming query."""
    return StreamingResponse(
        stream_query_events(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
