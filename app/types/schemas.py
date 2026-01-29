"""Pydantic request/response models for API."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel
from app.types.health import HealthComponents


class QueryRequest(BaseModel):
    """Query request with client and optional session tracking."""

    client: str  # user_id or client_id
    query: str
    dataset_id: Optional[str] = None
    session_id: Optional[str] = None  # For conversation tracking
    use_cache: bool = True


class QueryResponse(BaseModel):
    """Query response with unique IDs for audit trail."""

    success: bool
    result: Any = None
    method: str = "unknown"
    explanation: str = ""
    error: Optional[str] = None
    query_id: str = ""  # Unique query ID for audit
    metadata: Dict[str, Any] = {}


class UploadResponse(BaseModel):
    """Upload response with document and dataset IDs."""

    success: bool
    doc_id: str = ""  # Unique document ID
    datasets: List[Dict[str, Any]] = []
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    version: str
    provider: str
    components: HealthComponents


class JSONIngestRequest(BaseModel):
    """Request for ingesting raw JSON text."""

    json_text: str
    client_id: str
    source_name: str = "pasted_json"


class UploadUrlRequest(BaseModel):
    """Request for ingesting a file from a URL (e.g., S3 presigned)."""

    file_url: str
    client_id: str
    ingest_all: bool = True
    filename: Optional[str] = None


class RehydrateRequest(BaseModel):
    """Request to rehydrate datasets into memory."""

    client_id: str
    dataset_ids: Optional[List[str]] = None


class RootResponse(BaseModel):
    message: str
    data: Any = None
    error: Optional[str] = None
    status: str = "ok"
