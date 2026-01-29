"""Health controller."""

from app.types.schemas import RootResponse
from app.services.health_service import get_health


async def health_check() -> RootResponse:
    """Health check endpoint."""
    res = get_health()
    return RootResponse(
        message="Health check successfully completed",
        data=res,
        error=None,
        status="ok",
    )
