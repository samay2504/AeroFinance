"""File and JSON ingestion service."""

import asyncio
import dataclasses
import json
import logging
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import httpx
from fastapi import BackgroundTasks, HTTPException, UploadFile
from app.types.schemas import UploadResponse, JSONIngestRequest, UploadUrlRequest

logger = logging.getLogger("ai-ca")

_MAX_JOB_AGE_SECONDS = 3600  # purge jobs older than 1 hour


@dataclass
class _JobState:
    task_id: str
    filename: str
    client_id: str
    status: str = "queued"          # queued | processing | done | error
    result: Optional[Any] = None
    error: Optional[str] = None
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    pages_done: int = 0
    total_pages: int = 0
    doc_id: str = ""


class _JobStore:
    """
    Thread-safe job store — Redis-first with in-memory fallback.

    When Redis is configured the store writes through to Redis (TTL = 1 h),
    making every job visible across all process replicas behind a load balancer.
    When Redis is unavailable (local dev, no env var) the store falls back to
    a process-local dict with zero behaviour change for single-instance setups.
    """

    _REDIS_PREFIX = "ai-ca:job:"
    _REDIS_TTL = _MAX_JOB_AGE_SECONDS

    def __init__(self) -> None:
        self._local: Dict[str, _JobState] = {}
        self._lock = threading.Lock()

    # ── Redis helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _redis():  # type: ignore[return]
        """Return Redis client or None (never raises)."""
        try:
            from app.core.redis_client import get_redis
            return get_redis()
        except Exception:
            return None

    @staticmethod
    def _job_to_json(job: _JobState) -> str:
        d = dataclasses.asdict(job)
        return json.dumps(d, default=str)

    @staticmethod
    def _job_from_json(raw: bytes) -> _JobState:
        d = json.loads(raw)
        d.pop("pages_done", None)   # ignore unknown future fields gracefully
        d.pop("total_pages", None)
        return _JobState(**{k: v for k, v in d.items() if k in _JobState.__dataclass_fields__})

    # ── Public API ──────────────────────────────────────────────────────────

    def put(self, job: _JobState) -> None:
        """Insert or overwrite a job."""
        r = self._redis()
        if r is not None:
            try:
                r.setex(f"{self._REDIS_PREFIX}{job.task_id}", self._REDIS_TTL, self._job_to_json(job))
            except Exception as exc:
                logger.debug(f"Redis job write failed (falling back to local): {exc}")
        with self._lock:
            self._local[job.task_id] = job

    def get(self, task_id: str) -> Optional[_JobState]:
        """Return job or None."""
        r = self._redis()
        if r is not None:
            try:
                raw = r.get(f"{self._REDIS_PREFIX}{task_id}")
                if raw:
                    return self._job_from_json(raw)
            except Exception as exc:
                logger.debug(f"Redis job read failed (falling back to local): {exc}")
        with self._lock:
            return self._local.get(task_id)

    def update(self, task_id: str, **kwargs: Any) -> None:
        """Patch fields on an existing job in-place."""
        job = self.get(task_id)
        if job is None:
            return
        for k, v in kwargs.items():
            if hasattr(job, k):
                setattr(job, k, v)
        self.put(job)

    def prune_expired(self) -> None:
        """Remove completed/failed jobs older than _MAX_JOB_AGE_SECONDS (local only)."""
        cutoff = time.time() - _MAX_JOB_AGE_SECONDS
        with self._lock:
            stale = [
                k for k, v in self._local.items()
                if v.status in ("done", "error") and (v.finished_at or 0) < cutoff
            ]
            for k in stale:
                del self._local[k]


# Module-level singleton — all functions share this instance
_job_store = _JobStore()


def get_job_status(task_id: str) -> Optional[Dict[str, Any]]:
    """Return job status dict or None if task_id is unknown."""
    job = _job_store.get(task_id)
    if job is None:
        return None
    out: Dict[str, Any] = {
        "task_id": job.task_id,
        "status": job.status,
        "filename": job.filename,
        "client_id": job.client_id,
        "doc_id": job.doc_id,
        "started_at": job.started_at,
        "finished_at": job.finished_at,
        "elapsed_seconds": round((job.finished_at or time.time()) - job.started_at, 2),
    }
    if job.status == "done":
        out["result"] = job.result
    elif job.status == "error":
        out["error"] = job.error
    return out


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
    _t0 = time.time()

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
        logger.info(f"JSON ingest completed in {round(time.time() - _t0, 2)}s: {filename}")
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
        logger.info(f"Excel/CSV ingest completed in {round(time.time() - _t0, 2)}s: {filename}")
    elif ext == "pdf":
        import json
        import pandas as pd
        from app.ingest.pdf_ingest import get_pdf_ingestor
        from app.rag.ingest import get_document_ingestor

        _t0 = time.time()
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        try:
            ingestor = get_pdf_ingestor()  # singleton — no re-init cost
            ingest_result = ingestor.ingest_document(
                file_path=tmp_path,
                client_id=client_id,
                metadata={"original_filename": filename, "chat_id": chat_id},
            )
            result_dict = vars(ingest_result)
            success = result_dict.get("success", True)

            # ── Push chunks to DocumentIngestor + register DataFrame ──────────
            # When a cloud vector store is active, SmartPDFIngestor pushes chunks
            # directly during ingest_document() and skips writing the JSONL file.
            # When no cloud store is available the JSONL IS written and we read it.
            # Additionally, we register a lightweight DataFrame of (page, text)
            # with the DataAnalystAgent so summarize_dataset() works for PDFs.
            chunk_records: list = []
            try:
                jsonl_path = ingestor.cache_dir / f"{doc_id}.chunks.jsonl"
                if jsonl_path.exists():
                    doc_ingestor = get_document_ingestor()
                    bulk_text_parts = []
                    with open(jsonl_path) as jf:
                        for line in jf:
                            line = line.strip()
                            if not line:
                                continue
                            chunk = json.loads(line)
                            text = chunk.get("text") or chunk.get("summary") or ""
                            if text:
                                page = chunk.get("page_num", "?")
                                bulk_text_parts.append(f"[Page {page}] {text}")
                                chunk_records.append({"page": page, "text": text[:2000]})
                    if bulk_text_parts:
                        doc_ingestor.ingest_text(
                            text="\n\n".join(bulk_text_parts),
                            client_id=client_id,
                            dataset_id=doc_id,
                            metadata={"source": "pdf", "filename": filename, "doc_id": doc_id},
                        )
                        logger.info(
                            f"PDF chunks pushed to DocumentIngestor via JSONL: {len(bulk_text_parts)} pages"
                        )
                else:
                    logger.debug(
                        "JSONL not found — chunks were already pushed directly to cloud store "
                        "during ingest_document() or doc_id mismatch (tmp vs original filename)"
                    )
            except Exception as _ve:
                logger.warning(f"DocumentIngestor push skipped (non-fatal): {_ve}")

            # Register PDF chunk DataFrame with DataAnalystAgent so
            # summarize_dataset() can generate summaries for PDF documents.
            if chunk_records:
                try:
                    pdf_df = pd.DataFrame(chunk_records)
                    agent.register_dataframe(
                        dataset_id=doc_id,
                        df=pdf_df,
                        preprocessing_report={"source": "pdf", "type": "chunk_text"},
                        client_id=client_id,
                    )
                    logger.info(f"PDF DataFrame registered for summarize_dataset: {doc_id} ({len(pdf_df)} rows)")
                except Exception as _re:
                    logger.warning(f"PDF DataFrame registration skipped (non-fatal): {_re}")

            _elapsed = round(time.time() - _t0, 2)
            logger.info(f"PDF ingest completed in {_elapsed}s: {filename}")

            results = [{"success": success, "doc_id": doc_id, **result_dict}]
        finally:
            import os as _os
            try:
                _os.unlink(tmp_path)
            except Exception:
                pass
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: .{ext}. Supported: .xlsx, .xls, .csv, .json, .pdf",
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
    """
    Ingest from a local file path by reading bytes and delegating to
    _ingest_file_content.  This keeps routing logic in one place (DRY).

    PDF special case: call ingest_document() directly with the real path so
    we avoid writing a redundant temp file — the document is already on disk.
    """
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if ext == "pdf":
        # Direct-path PDF ingest — no temp file overhead
        import json
        import pandas as pd
        from app.agents.data_analyst import get_data_analyst_agent
        from app.core.id_generator import generate_doc_id
        from app.ingest.pdf_ingest import get_pdf_ingestor
        from app.rag.ingest import get_document_ingestor

        _t0 = time.time()
        agent = get_data_analyst_agent()
        doc_id = generate_doc_id(client_id, filename)
        ingestor = get_pdf_ingestor()
        ingest_result = ingestor.ingest_document(
            file_path=str(file_path),
            client_id=client_id,
            metadata={"original_filename": filename},
        )
        result_dict = vars(ingest_result)
        success = result_dict.get("success", True)
        chunk_records: list = []
        try:
            jsonl_path = ingestor.cache_dir / f"{doc_id}.chunks.jsonl"
            if jsonl_path.exists():
                doc_ingestor = get_document_ingestor()
                parts = []
                with open(jsonl_path) as jf:
                    for line in jf:
                        line = line.strip()
                        if not line:
                            continue
                        chunk = json.loads(line)
                        text = chunk.get("text") or chunk.get("summary") or ""
                        if text:
                            page = chunk.get("page_num", "?")
                            parts.append(f"[Page {page}] {text}")
                            chunk_records.append({"page": page, "text": text[:2000]})
                if parts:
                    doc_ingestor.ingest_text(
                        text="\n\n".join(parts),
                        client_id=client_id,
                        dataset_id=doc_id,
                        metadata={"source": "pdf", "filename": filename, "doc_id": doc_id},
                    )
        except Exception as _ve:
            logger.warning(f"PDF JSONL push skipped (non-fatal): {_ve}")

        # Register PDF chunk DataFrame for summarize_dataset() support
        if chunk_records:
            try:
                pdf_df = pd.DataFrame(chunk_records)
                agent.register_dataframe(
                    dataset_id=doc_id,
                    df=pdf_df,
                    preprocessing_report={"source": "pdf", "type": "chunk_text"},
                    client_id=client_id,
                )
                logger.info(f"PDF DataFrame registered (path ingest): {doc_id} ({len(pdf_df)} rows)")
            except Exception as _re:
                logger.warning(f"PDF DataFrame registration skipped (non-fatal): {_re}")

        _elapsed = round(time.time() - _t0, 2)
        logger.info(f"PDF path-ingest completed in {_elapsed}s: {filename}")
        results = [{"success": success, "doc_id": doc_id, **result_dict}]
        return UploadResponse(success=success, doc_id=doc_id, datasets=results)

    # For JSON / Excel / CSV: read bytes and reuse _ingest_file_content (single dispatch point)
    content = file_path.read_bytes()
    return _ingest_file_content(content, filename, client_id, ingest_all)


def _run_background(
    task_id: str,
    content: bytes,
    filename: str,
    client_id: str,
    ingest_all: bool,
    chat_id: Optional[str],
) -> None:
    """
    Universal background worker for all file types.

    Writes job state through _job_store (Redis-first / in-memory fallback),
    enabling horizontal scaling across multiple process replicas.
    """
    _job_store.update(task_id, status="processing")
    try:
        result = _ingest_file_content(content, filename, client_id, ingest_all, chat_id=chat_id)
        result_payload = result.model_dump() if hasattr(result, "model_dump") else vars(result)
        _job_store.update(task_id, status="done", result=result_payload, finished_at=time.time())
        logger.info(f"Background job {task_id} done for {filename}")
    except Exception as exc:
        _job_store.update(task_id, status="error", error=str(exc), finished_at=time.time())
        logger.error(f"Background job {task_id} failed: {exc}", exc_info=True)
    finally:
        _job_store.prune_expired()


async def upload_file(
    file: UploadFile,
    client_id: str,
    ingest_all: bool,
    chat_id: Optional[str] = None,
    background_tasks: Optional[BackgroundTasks] = None,
) -> UploadResponse:
    """
    Accept any file and return sub-100 ms HTTP 200 with a ``task_id``.

    All parsing, chunking and indexing runs in a FastAPI BackgroundTask,
    matching ChatGPT / Claude upload UX: instant acknowledgement, async result.

    Poll ``GET /upload/status/{task_id}`` for progress and final output.
    Supports all file types: PDF, Excel, CSV, JSON.
    """
    # ── Supported extension → MIME-type map (fast O(1) lookup, no LLM) ──────
    _SUPPORTED_EXTS = {"pdf", "xlsx", "xls", "csv", "json"}
    _MIME_EXT_MAP: Dict[str, str] = {
        "application/pdf": "pdf",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
        "application/vnd.ms-excel": "xls",
        "text/csv": "csv",
        "application/json": "json",
        "text/json": "json",
        "text/plain": "",  # allow; resolve by extension
    }
    try:
        content = await file.read()
        filename = file.filename or "uploaded_file"
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

        # Fast MIME gate — reject unsupported types before allocating a job ID
        mime = (file.content_type or "").split(";")[0].strip().lower()
        if mime and mime not in _MIME_EXT_MAP and ext not in _SUPPORTED_EXTS:
            raise HTTPException(
                status_code=415,
                detail=(
                    f"Unsupported media type '{mime}' / extension '.{ext}'. "
                    f"Accepted: PDF, XLSX, XLS, CSV, JSON."
                ),
            )
        if ext not in _SUPPORTED_EXTS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file extension '.{ext}'. Accepted: pdf, xlsx, xls, csv, json.",
            )

        from app.core.id_generator import generate_doc_id
        task_id = uuid.uuid4().hex[:12]
        doc_id = generate_doc_id(client_id, filename, chat_id=chat_id)

        # Register job before scheduling so status endpoint finds it immediately
        job = _JobState(task_id=task_id, filename=filename, client_id=client_id, doc_id=doc_id)
        _job_store.put(job)

        if background_tasks is not None:
            # ── Non-blocking path: return immediately, process in background ──
            background_tasks.add_task(
                _run_background,
                task_id, content, filename, client_id, ingest_all, chat_id,
            )
            logger.info(
                f"Upload accepted: {filename} ({len(content) // 1024} KB) "
                f"→ background task {task_id}"
            )
            return UploadResponse(
                success=True,
                doc_id=doc_id,
                datasets=[{
                    "task_id": task_id,
                    "status": "processing",
                    "filename": filename,
                    "message": (
                        f"File accepted and queued for ingestion. "
                        f"Poll GET /upload/status/{task_id} for result."
                    ),
                }],
            )

        # ── Inline fallback (no background_tasks context, e.g. tests) ─────────
        return await asyncio.to_thread(
            _ingest_file_content, content, filename, client_id, ingest_all, chat_id
        )
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

        return await asyncio.to_thread(
            _ingest_file_path, temp_path, filename, request.client_id, request.ingest_all
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
