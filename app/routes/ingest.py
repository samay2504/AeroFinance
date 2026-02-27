"""Ingestion routes."""

from typing import Optional
from fastapi import APIRouter, BackgroundTasks, HTTPException, UploadFile, File, Form
from app.controllers.ingest_controller import (
    upload_file,
    ingest_json_text,
    ingest_file_url,
)
from app.services.ingest_service import get_job_status
from app.types.schemas import RootResponse, JSONIngestRequest, UploadUrlRequest

router = APIRouter()


@router.post("/upload", response_model=RootResponse, tags=["Ingest"])
async def upload_route(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    client_id: str = Form(...),
    ingest_all: bool = Form(True),
    chat_id: Optional[str] = Form(None),
):
    """Upload and ingest a file.

    * **PDF** — returns immediately (< 1 s) with ``task_id`` and
      ``status="processing"``.  Ingestion continues in the background.
      Poll ``GET /upload/status/{task_id}`` for the final result.
    * **Excel / CSV / JSON** — synchronous; result returned directly.
    """
    result = await upload_file(
        file=file,
        client_id=client_id,
        ingest_all=ingest_all,
        chat_id=chat_id,
        background_tasks=background_tasks,
    )
    return RootResponse(
        message="Upload accepted",
        data=result,
        error=None,
        status="ok",
    )


@router.get("/upload/status/{task_id}", response_model=RootResponse, tags=["Ingest"])
async def upload_status_route(task_id: str):
    """Poll background PDF ingestion status.

    Returns ``status`` as one of: ``queued`` | ``processing`` | ``done`` | ``error``.
    When ``status=="done"`` the full ingestion result is in ``data.result``.
    """
    status = get_job_status(task_id)
    if status is None:
        raise HTTPException(
            status_code=404,
            detail=f"Task '{task_id}' not found. "
                   "Tasks expire 1 hour after completion.",
        )
    return RootResponse(
        message=f"Task {task_id}: {status['status']}",
        data=status,
        error=status.get("error"),
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
