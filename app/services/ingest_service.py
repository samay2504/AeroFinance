"""File and JSON ingestion service."""

import logging
import tempfile
from pathlib import Path
from urllib.parse import urlparse
from typing import Optional

import httpx
from fastapi import HTTPException, UploadFile
from app.types.schemas import UploadResponse, JSONIngestRequest, UploadUrlRequest

logger = logging.getLogger("ai-ca")


def _infer_filename_from_url(file_url: str) -> str:
    parsed = urlparse(file_url)
    name = Path(parsed.path).name
    return name or "uploaded_file"


def _register_callback(agent, client_id: str, doc_id: str):
    from app.core.id_generator import generate_dataset_id

    def register_cb(dataset_id: str, df, metadata: dict):
        sheet_name = dataset_id.split(":")[-1] if ":" in dataset_id else dataset_id
        full_dataset_id = generate_dataset_id(client_id, doc_id, sheet_name)
        agent.register_dataframe(
            full_dataset_id,
            df,
            preprocessing_report=metadata.get("preprocessing"),
            client_id=client_id,
        )
        metadata["full_dataset_id"] = full_dataset_id

    return register_cb


def _ingest_file_content(
    content: bytes,
    filename: str,
    client_id: str,
    ingest_all: bool,
    chat_id: Optional[str] = None,
) -> UploadResponse:
    from app.agents.data_analyst import get_data_analyst_agent
    from app.core.id_generator import generate_doc_id

    agent = get_data_analyst_agent()
    doc_id = generate_doc_id(client_id, filename, chat_id=chat_id)
    logger.info(f"Generated doc_id: {doc_id} for {filename}")
    register_cb = _register_callback(agent, client_id, doc_id)

    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if ext == "json":
        from app.ingest.json_ingest import JSONIngestor

        ingestor = JSONIngestor()
        result = ingestor.ingest_json_file(
            file_content=content,
            filename=filename,
            client_id=client_id,
            register_callback=register_cb,
        )
        results = result.get("datasets", [])
        success = result.get("success", False)
    elif ext in ["xlsx", "xls", "csv"]:
        from app.ingest.excel_ingest import ExcelIngestor

        ingestor = ExcelIngestor()
        if ingest_all:
            results = ingestor.ingest_all_sheets(
                file_content=content,
                filename=filename,
                client_id=client_id,
                register_callback=register_cb,
            )
        else:
            result = ingestor.ingest_best_sheet(
                file_content=content,
                filename=filename,
                client_id=client_id,
                register_callback=register_cb,
            )
            results = [result]
        success = any(r.get("success") for r in results)
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: .{ext}. Supported: .xlsx, .xls, .csv, .json",
        )

    # Add doc_id to each dataset for consistency
    for dataset in results:
        if dataset.get("success"):
            dataset["doc_id"] = doc_id

    return UploadResponse(success=success, doc_id=doc_id, datasets=results)


def _ingest_file_path(
    file_path: Path,
    filename: str,
    client_id: str,
    ingest_all: bool,
) -> UploadResponse:
    from app.agents.data_analyst import get_data_analyst_agent
    from app.core.id_generator import generate_doc_id

    agent = get_data_analyst_agent()
    doc_id = generate_doc_id(client_id, filename)
    logger.info(f"Generated doc_id: {doc_id} for {filename}")
    register_cb = _register_callback(agent, client_id, doc_id)

    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if ext == "json":
        from app.ingest.json_ingest import JSONIngestor

        ingestor = JSONIngestor()
        result = ingestor.ingest_json_path(
            file_path=str(file_path),
            filename=filename,
            client_id=client_id,
            register_callback=register_cb,
        )
        results = result.get("datasets", [])
        success = result.get("success", False)
    elif ext in ["xlsx", "xls", "csv"]:
        from app.ingest.excel_ingest import ExcelIngestor

        ingestor = ExcelIngestor()
        if ingest_all:
            results = ingestor.ingest_all_sheets_from_path(
                file_path=file_path,
                filename=filename,
                client_id=client_id,
                register_callback=register_cb,
            )
        else:
            result = ingestor.ingest_best_sheet_from_path(
                file_path=file_path,
                filename=filename,
                client_id=client_id,
                register_callback=register_cb,
            )
            results = [result]
        success = any(r.get("success") for r in results)
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: .{ext}. Supported: .xlsx, .xls, .csv, .json",
        )

    # Add doc_id to each dataset for consistency
    for dataset in results:
        if dataset.get("success"):
            dataset["doc_id"] = doc_id

    return UploadResponse(success=success, doc_id=doc_id, datasets=results)


async def upload_file(
    file: UploadFile, client_id: str, ingest_all: bool, chat_id: Optional[str] = None
) -> UploadResponse:
    """Ingest uploaded file content."""
    try:
        content = await file.read()
        filename = file.filename or "uploaded_file"
        return _ingest_file_content(content, filename, client_id, ingest_all, chat_id=chat_id)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def ingest_file_url(request: UploadUrlRequest) -> UploadResponse:
    """Download a file from URL and ingest it from a local path."""
    filename = request.filename or _infer_filename_from_url(request.file_url)
    if not filename:
        filename = "uploaded_file"

    temp_path: Optional[Path] = None
    try:
        suffix = Path(filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            temp_path = Path(tmp.name)
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream(
                    "GET", request.file_url, follow_redirects=True
                ) as response:
                    response.raise_for_status()
                    async for chunk in response.aiter_bytes():
                        tmp.write(chunk)

        return _ingest_file_path(
            temp_path, filename, request.client_id, request.ingest_all
        )
    except httpx.HTTPError as e:
        raise HTTPException(status_code=400, detail=f"Failed to download file: {e}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"URL ingest error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_path and temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass


async def ingest_json_text(request: JSONIngestRequest) -> UploadResponse:
    """Ingest raw JSON text directly (for pasted content) with RAG support."""
    try:
        from app.ingest.json_ingest import JSONIngestor
        from app.agents.data_analyst import get_data_analyst_agent
        from app.rag.ingest import get_rag_pipeline
        from app.core.id_generator import generate_doc_id

        agent = get_data_analyst_agent()
        ingestor = JSONIngestor()
        
        # Generate doc_id for consistency
        doc_id = generate_doc_id(request.client_id, request.source_name)
        logger.info(f"Generated doc_id: {doc_id} for JSON ingest")

        # Get RAG pipeline (may be None if unavailable)
        rag_pipeline = None
        try:
            rag_pipeline = get_rag_pipeline()
        except Exception:
            pass

        # Register callback
        def register_cb(dataset_id: str, df, metadata: dict):
            agent.register_dataframe(
                dataset_id,
                df,
                preprocessing_report=metadata.get("preprocessing"),
                client_id=request.client_id,
            )

        # Parse JSON first
        data, parse_error = ingestor.parse_json_text(request.json_text)
        if parse_error:
            return UploadResponse(success=False, doc_id="", error=parse_error)

        # Use RAG ingestion if available
        if rag_pipeline and rag_pipeline.is_available:
            rag_result = ingestor.ingest_to_rag(
                data=data,
                source_name=request.source_name,
                client_id=request.client_id,
                rag_pipeline=rag_pipeline,
            )
            logger.info(f"RAG indexed {rag_result.get('rag_chunks', 0)} chunks")

        # Also register DataFrames for SQL queries
        result = ingestor.ingest_json(
            data=data,
            source_name=request.source_name,
            client_id=request.client_id,
            register_callback=register_cb,
        )
        
        # Add doc_id to each dataset for consistency
        datasets = result.get("datasets", [])
        for dataset in datasets:
            if dataset.get("success"):
                dataset["doc_id"] = doc_id

        return UploadResponse(
            success=result.get("success", False),
            doc_id=doc_id,
            datasets=datasets
        )

    except Exception as e:
        logger.error(f"JSON ingest error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
