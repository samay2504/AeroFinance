"""Ingestion routes."""

from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form
from app.controllers.ingest_controller import (
    upload_file,
    ingest_json_text,
    ingest_file_url,
)
from app.types.schemas import RootResponse, JSONIngestRequest, UploadUrlRequest

router = APIRouter()


@router.post("/upload", response_model=RootResponse, tags=["Ingest"])
async def upload_route(
    file: UploadFile = File(...),
    client_id: str = Form(...),
    ingest_all: bool = Form(True),
    chat_id: Optional[str] = Form(None),
):
    """Upload and ingest Excel/CSV/JSON file with unique document ID generation."""
    result = await upload_file(file=file, client_id=client_id, ingest_all=ingest_all, chat_id=chat_id)
    return RootResponse(
        message="Upload completed",
        data=result,
        error=None,
        status="ok",
    )


@router.post("/ingest-json", response_model=RootResponse, tags=["Ingest"])
async def ingest_json_route(request: JSONIngestRequest):
    """Ingest raw JSON text directly (for pasted content) with RAG support."""
    result = await ingest_json_text(request)
    return RootResponse(
        message="JSON ingest completed",
        data=result,
        error=None,
        status="ok",
    )


@router.post("/upload-url", response_model=RootResponse, tags=["Ingest"])
async def upload_url_route(request: UploadUrlRequest):
    """Download a file by URL (e.g., S3 presigned) and ingest it."""
    result = await ingest_file_url(request)
    return RootResponse(
        message="URL upload completed",
        data=result,
        error=None,
        status="ok",
    )
