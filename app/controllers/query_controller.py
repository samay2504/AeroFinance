"""Query controller."""

from app.types.schemas import QueryRequest, QueryResponse
from app.services.query_service import handle_query


async def query(request: QueryRequest) -> QueryResponse:
    """Controller for query execution."""
    return await handle_query(request)
