"""Ingestion controller."""

from typing import Optional
from fastapi import BackgroundTasks, UploadFile
from app.types.schemas import UploadResponse, JSONIngestRequest, UploadUrlRequest
from app.services.ingest_service import upload_file as upload_file_service
from app.services.ingest_service import ingest_json_text as ingest_json_text_service
from app.services.ingest_service import ingest_file_url as ingest_file_url_service


async def upload_file(
    file: UploadFile,
    client_id: str,
    ingest_all: bool,
    chat_id: Optional[str] = None,
    background_tasks: Optional[BackgroundTasks] = None,
) -> UploadResponse:
    """Controller for file upload ingestion.

    Passes ``background_tasks`` to the service so PDF uploads can return
    immediately while ingestion continues in the background.
    """
    return await upload_file_service(
        file=file,
        client_id=client_id,
        ingest_all=ingest_all,
        chat_id=chat_id,
        background_tasks=background_tasks,
    )


async def ingest_json_text(request: JSONIngestRequest) -> UploadResponse:
    """Controller for JSON text ingestion."""
    return await ingest_json_text_service(request)


async def ingest_file_url(request: UploadUrlRequest) -> UploadResponse:
    """Controller for URL ingestion."""
    return await ingest_file_url_service(request)
