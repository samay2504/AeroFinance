"""
AI-CA Main Application - FastAPI entrypoint with CLI runner.
Production-grade Agentic AI Chartered Accountant RAG System.
"""
import sys
import os
import time
from collections import defaultdict

try:
    from app.core.dll_fix import apply_dll_fix
    apply_dll_fix()
except ImportError:
    pass

import logging
import argparse
import json
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from app.core.id_generator import generate_request_id, get_iso_timestamp, normalize_client_id
from app.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("ai-ca")

MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "50"))
ALLOWED_EXTENSIONS = {"xlsx", "xls", "csv", "pdf", "docx", "json"}
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
RATE_LIMIT_WINDOW_SEC = int(os.getenv("RATE_LIMIT_WINDOW_SEC", "60"))

_rate_limit_store: Dict[str, List[float]] = defaultdict(list)


class APIEnvelope(BaseModel):
    status: str = Field(..., description="'ok' or 'error'")
    request_id: str = Field(..., description="Unique request identifier")
    data: Optional[Any] = Field(None, description="Response payload")
    error: Optional[Dict[str, Any]] = Field(None, description="Error details if status=error")


def make_response(
    data: Any = None, 
    request_id: str = None,
    status: str = "ok",
    error_code: str = None,
    error_message: str = None
) -> Dict[str, Any]:
    request_id = request_id or generate_request_id()
    if status == "error":
        return {
            "status": "error",
            "request_id": request_id,
            "data": None,
            "error": {"code": error_code or "UNKNOWN_ERROR", "message": error_message or "An error occurred"}
        }
    return {"status": "ok", "request_id": request_id, "data": data, "error": None}


def log_request(
    request_id: str,
    route: str,
    user_id: str = None,
    doc_id: str = None,
    status: str = "ok",
    duration_ms: float = 0,
    extra: Dict[str, Any] = None,
    log_type: str = "request"
):
    log_entry = {
        "timestamp": get_iso_timestamp(),
        "level": "INFO" if status == "ok" else "ERROR",
        "request_id": request_id,
        "route": route,
        "status": status,
        "duration_ms": round(duration_ms, 2),
        "log_type": log_type
    }
    if user_id:
        log_entry["user_id"] = user_id
    if doc_id:
        log_entry["doc_id"] = doc_id
    if extra:
        log_entry.update({k: v for k, v in extra.items() if k not in ("content", "file_content", "data", "password", "token")})
    
    logger.info(json.dumps(log_entry))
    
    
    log_base = Path(settings.logging.dir)
    try:
        date_str = datetime.utcnow().strftime('%Y-%m-%d')
        safe_client = normalize_client_id(user_id) if user_id else "system"
        
        if log_type == "request":
            log_dir = log_base / "requests" / safe_client / date_str
        elif log_type == "upload":
            log_dir = log_base / "uploads" / safe_client / date_str
        elif log_type == "query":
            log_dir = log_base / "queries" / safe_client / date_str
        elif log_type == "error":
            log_dir = log_base / "errors" / safe_client / date_str
        else:
            log_dir = log_base / "misc" / safe_client / date_str
        
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{request_id}.json"
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(log_entry, f, indent=2, ensure_ascii=False)
    except Exception:
        pass
    
    if settings.logging.to_s3:
        try:
            from app.core.data_registry import get_data_registry
            registry = get_data_registry()
            if hasattr(registry, '_backend') and hasattr(registry._backend, '_s3_client'):
                bucket = settings.logging.s3_bucket or settings.deployment.aws_s3_bucket
                if bucket:
                    prefix = settings.logging.s3_prefix.strip("/")
                    s3_key = f"{prefix}/{log_type}/{safe_client}/{date_str}/{request_id}.json"
                    registry._backend._s3_client.put_object(
                        Bucket=bucket, Key=s3_key, Body=json.dumps(log_entry, ensure_ascii=False).encode('utf-8')
                    )
        except Exception:
            pass


def check_rate_limit(client_key: str) -> bool:
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW_SEC
    _rate_limit_store[client_key] = [t for t in _rate_limit_store[client_key] if t > window_start]
    if len(_rate_limit_store[client_key]) >= RATE_LIMIT_REQUESTS:
        return False
    _rate_limit_store[client_key].append(now)
    return True


# Request/Response models
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
    components: Dict[str, str]


# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown."""
    logger.info("Starting AI-CA application...")
    
    # Initialize components
    try:
        from app.config import settings
        from app.core.llm_wrapper import get_llm_wrapper
        from app.core.data_registry import get_data_registry
        from app.ipc.zmq_bridge import get_zmq_bridge
        
        # Warm up LLM wrapper
        llm = get_llm_wrapper()
        logger.info(f"LLM Provider: {llm.provider_name}")
        
        # Initialize data registry
        registry = get_data_registry()
        logger.info(f"Data registry: {len(registry.list_all())} datasets")
        
        # Start ZMQ bridge if enabled
        if settings.zmq.enabled:
            bridge = get_zmq_bridge()
            bridge.start()
        
    except Exception as e:
        logger.error(f"Startup error: {e}")
    
    yield
    
    # Cleanup
    logger.info("Shutting down AI-CA...")
    try:
        from app.ipc.zmq_bridge import get_zmq_bridge
        bridge = get_zmq_bridge()
        bridge.stop()
    except Exception:
        pass


# Create FastAPI app
app = FastAPI(
    title="AI-CA",
    description="Agentic AI Chartered Accountant RAG System",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Import centralized natural response formatter from prompts
from app.core.prompts import format_natural_response as _format_natural_response


@app.get("/health")
async def health_check(x_request_id: Optional[str] = Header(None)):
    request_id = x_request_id or generate_request_id()
    start_time = time.time()
    from app.config import settings
    
    components = {"api": "ok", "sql_engine": "unknown", "vector_db": "unknown", "llm": "unknown"}
    
    try:
        from app.sql_engine import get_sql_engine
        engine = get_sql_engine()
        components["sql_engine"] = "ok" if engine else "unavailable"
    except Exception:
        components["sql_engine"] = "error"
    
    try:
        from app.rag.ingest import get_document_ingestor
        ingestor = get_document_ingestor()
        components["vector_db"] = ingestor._active_store or "unavailable"
    except Exception:
        components["vector_db"] = "error"
    
    try:
        from app.core.llm_wrapper import get_llm_wrapper
        llm = get_llm_wrapper()
        components["llm"] = llm.provider_name or "fallback"
    except Exception:
        components["llm"] = "error"
    
    uptime_s = int(time.time() - start_time)
    log_request(request_id, "/health", duration_ms=(time.time() - start_time) * 1000)
    return make_response(
        data={"version": "1.0.0", "provider": components["llm"], "components": components, "uptime_s": uptime_s},
        request_id=request_id
    )


@app.post("/v1/ai-ca/upload")
async def upload_file(
    file: UploadFile = File(...),
    client_id: str = Form(...),
    ingest_all: bool = Form(True),
    x_request_id: Optional[str] = Header(None)
):
    request_id = x_request_id or generate_request_id()
    start_time = time.time()
    doc_id = None
    safe_client = normalize_client_id(client_id)
    
    try:
        if not check_rate_limit(safe_client):
            return JSONResponse(
                status_code=429,
                content=make_response(request_id=request_id, status="error", error_code="RATE_LIMIT_EXCEEDED", error_message="Too many requests")
            )
        
        filename = file.filename or "uploaded_file"
        ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
        
        if ext not in ALLOWED_EXTENSIONS:
            log_request(request_id, "/v1/ai-ca/upload", user_id=safe_client, status="error", duration_ms=(time.time() - start_time) * 1000)
            return JSONResponse(
                status_code=400,
                content=make_response(request_id=request_id, status="error", error_code="INVALID_INPUT", error_message=f"Unsupported file type: .{ext}. Allowed: {', '.join(ALLOWED_EXTENSIONS)}")
            )
        
        content = await file.read()
        size_mb = len(content) / (1024 * 1024)
        if size_mb > MAX_UPLOAD_SIZE_MB:
            log_request(request_id, "/v1/ai-ca/upload", user_id=safe_client, status="error", duration_ms=(time.time() - start_time) * 1000)
            return JSONResponse(
                status_code=400,
                content=make_response(request_id=request_id, status="error", error_code="INVALID_INPUT", error_message=f"File too large: {size_mb:.1f}MB. Max: {MAX_UPLOAD_SIZE_MB}MB")
            )
        
        from app.core.data_registry import get_data_registry
        from app.agents.data_analyst import get_data_analyst_agent
        from app.core.id_generator import generate_doc_id, generate_dataset_id
        
        registry = get_data_registry()
        agent = get_data_analyst_agent()
        doc_id = generate_doc_id(safe_client, filename)
        dataset_ids = []
        sheet_names = []
        sheets_count = 0
        
        def register_cb(dataset_id: str, df, metadata: dict):
            nonlocal sheets_count
            sheet_name = dataset_id.split(':')[-1] if ':' in dataset_id else dataset_id
            full_dataset_id = generate_dataset_id(safe_client, doc_id, sheet_name)
            agent.register_dataframe(full_dataset_id, df, preprocessing_report=metadata.get("preprocessing"), client_id=safe_client)
            metadata["full_dataset_id"] = full_dataset_id
            dataset_ids.append(full_dataset_id)
            sheet_names.append(sheet_name)
            sheets_count += 1
        
        if ext == 'json':
            from app.ingest.json_ingest import JSONIngestor
            ingestor = JSONIngestor()
            result = ingestor.ingest_json_file(file_content=content, filename=filename, client_id=safe_client, register_callback=register_cb)
            results = result.get("datasets", [])
            success = result.get("success", False)
        elif ext in ['xlsx', 'xls', 'csv']:
            from app.ingest.excel_ingest import ExcelIngestor
            ingestor = ExcelIngestor()
            if ingest_all:
                results = ingestor.ingest_all_sheets(file_content=content, filename=filename, client_id=safe_client, register_callback=register_cb)
            else:
                result = ingestor.ingest_best_sheet(file_content=content, filename=filename, client_id=safe_client, register_callback=register_cb)
                results = [result]
            success = any(r.get("success") for r in results)
        else:
            success = False
            results = []
        
        duration_ms = (time.time() - start_time) * 1000
        log_request(request_id, "/v1/ai-ca/upload", user_id=safe_client, doc_id=doc_id, status="ok" if success else "error", duration_ms=duration_ms, extra={"sheets_count": sheets_count, "filename": filename}, log_type="upload")
        
        return make_response(
            data={"doc_id": doc_id, "sheet_names": sheet_names, "sheets_count": sheets_count, "dataset_ids": dataset_ids, "datasets": results},
            request_id=request_id
        )
        
    except Exception as e:
        log_request(request_id, "/v1/ai-ca/upload", user_id=safe_client, doc_id=doc_id, status="error", duration_ms=(time.time() - start_time) * 1000, extra={"error": str(e)}, log_type="error")
        logger.error(f"Upload error: {e}")
        return JSONResponse(
            status_code=500,
            content=make_response(request_id=request_id, status="error", error_code="INTERNAL_ERROR", error_message=str(e))
        )


class JSONIngestRequest(BaseModel):
    """Request for ingesting raw JSON text."""
    json_text: str
    client_id: str
    source_name: str = "pasted_json"


@app.post("/v1/ai-ca/ingest-json", response_model=UploadResponse)
async def ingest_json_text(request: JSONIngestRequest):
    """Ingest raw JSON text directly (for pasted content) with RAG support."""
    try:
        from app.ingest.json_ingest import JSONIngestor
        from app.agents.data_analyst import get_data_analyst_agent
        from app.rag.ingest import get_rag_pipeline
        
        agent = get_data_analyst_agent()
        ingestor = JSONIngestor()
        
        # Get RAG pipeline (may be None if unavailable)
        rag_pipeline = None
        try:
            rag_pipeline = get_rag_pipeline()
        except Exception:
            pass
        
        # Register callback
        def register_cb(dataset_id: str, df, metadata: dict):
            agent.register_dataframe(
                dataset_id, df,
                preprocessing_report=metadata.get("preprocessing"),
                client_id=request.client_id
            )
        
        # Parse JSON first
        data, parse_error = ingestor.parse_json_text(request.json_text)
        if parse_error:
            return UploadResponse(success=False, error=parse_error)
        
        # Use RAG ingestion if available
        if rag_pipeline and rag_pipeline.is_available:
            rag_result = ingestor.ingest_to_rag(
                data=data,
                source_name=request.source_name,
                client_id=request.client_id,
                rag_pipeline=rag_pipeline
            )
            logger.info(f"RAG indexed {rag_result.get('rag_chunks', 0)} chunks")
        
        # Also register DataFrames for SQL queries
        result = ingestor.ingest_json(
            data=data,
            source_name=request.source_name,
            client_id=request.client_id,
            register_callback=register_cb
        )
        
        return UploadResponse(
            success=result.get("success", False),
            datasets=result.get("datasets", [])
        )
        
    except Exception as e:
        logger.error(f"JSON ingest error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/ai-ca/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Execute analytical query with unique query ID for audit trail."""
    try:
        from app.agents.router import (
            get_router_agent, TRACK_DATA, TRACK_DOC, TRACK_WEB,
            TRACK_DOC_SUMMARY, TRACK_OUT_OF_DOMAIN
        )
        from app.agents.data_analyst import get_data_analyst_agent
        from app.core.llm_wrapper import get_llm_wrapper
        from app.core.data_registry import get_data_registry
        from app.core.id_generator import generate_query_id, get_iso_timestamp
        
        # Generate unique query ID for audit trail
        query_id = generate_query_id()
        timestamp = get_iso_timestamp()
        logger.info(f"Query {query_id}: '{request.query[:50]}...' from {request.client}")
        
        llm = get_llm_wrapper()
        router = get_router_agent(llm)
        agent = get_data_analyst_agent(llm)
        registry = get_data_registry()
        
        # Check if client has data loaded
        datasets = agent.list_datasets_for_client(request.client)
        has_loaded_data = len(datasets) > 0
        
        # Route query with data context
        route_result = router.route(request.query, f"Client: {request.client}", has_loaded_data=has_loaded_data)
        track = route_result.get("track", TRACK_DATA)
        
        logger.info(f"Query {query_id}: Routed to {track} (confidence: {route_result.get('confidence', 0):.2f})")
        
        # ==================================================================
        # HANDLE OUT-OF-DOMAIN QUERIES
        # ==================================================================
        if track == TRACK_OUT_OF_DOMAIN:
            return QueryResponse(
                success=True,
                result="I'm an AI Chartered Accountant assistant. I can help you with financial data analysis, revenue/expense calculations, and document search. Please ask me something related to your financial data!",
                method="out_of_domain",
                explanation="Query was not related to CA/financial domain",
                query_id=query_id,
                metadata={"route": track}
            )
        
        # ==================================================================
        # HANDLE DATASET SUMMARY QUERIES
        # ==================================================================
        if track == TRACK_DOC_SUMMARY:
            if not datasets:
                return QueryResponse(
                    success=False,
                    error="No datasets found. Please upload data first.",
                    method="summary",
                    query_id=query_id
                )
            
            # Use summarize_dataset for each dataset
            summaries = []
            for ds in datasets[:5]:  # Limit to 5
                ds_id = ds.get("dataset_id", "")
                result = agent.summarize_dataset(ds_id, client_id=request.client)
                if result.get("value"):
                    sheet_name = ds_id.split(":")[-1]
                    summaries.append(f"**{sheet_name}:** {result['value']}")
            
            if summaries:
                combined = "\n\n".join(summaries)
                return QueryResponse(
                    success=True,
                    result=combined,
                    method="summarize_dataset",
                    explanation=f"Summary of {len(summaries)} dataset(s)",
                    query_id=query_id,
                    metadata={"route": track, "datasets": len(summaries)}
                )
            else:
                return QueryResponse(
                    success=False,
                    error="Could not generate dataset summaries",
                    method="summary",
                    query_id=query_id
                )
        
        if track == TRACK_DATA:
            # datasets already fetched earlier (for has_loaded_data check)
            
            if not datasets:
                return QueryResponse(
                    success=False,
                    error="No datasets found for client. Please upload data first.",
                    query_id=query_id
                )
            
            # Try specific dataset if provided, otherwise find matching one
            best_result = None
            best_dataset_id = None
            
            if request.dataset_id:
                # Use specified dataset
                result = agent.execute_sql_query(
                    query=request.query,
                    df_id=request.dataset_id,
                    client_id=request.client,
                    use_cache=request.use_cache
                )
                if result.success:
                    best_result = result
                    best_dataset_id = request.dataset_id
            else:
                # Keyword-based matching for JSON tables
                query_lower = request.query.lower()
                keyword_map = {
                    'balance sheet': 'balance_sheet', 'assets': 'balance_sheet', 'liabilities': 'balance_sheet',
                    'income statement': 'income_statement', 'revenue': 'income_statement', 'profit': 'income_statement',
                    'company': 'company_meta', 'ticker': 'company_meta', 'employees': 'company_meta',
                }
                
                keyword_match = None
                for keyword, table_suffix in keyword_map.items():
                    if keyword in query_lower:
                        for ds in datasets:
                            ds_id = ds.get('dataset_id', '')
                            if ds_id.endswith(table_suffix):
                                keyword_match = ds_id
                                break
                        if keyword_match:
                            break
                
                # Priority: keyword match > smart match
                matched_id = keyword_match or agent.match_dataset_by_query(request.query, datasets)
                
                if matched_id:
                    result = agent.execute_sql_query(
                        query=request.query,
                        df_id=matched_id,
                        client_id=request.client,
                        use_cache=request.use_cache
                    )
                    if result.success:
                        best_result = result
                        best_dataset_id = matched_id
                
                # If matched dataset failed, try all datasets
                if not best_result or not best_result.success:
                    for ds in datasets[:5]:  # Limit to 5 datasets
                        ds_id = ds.get('dataset_id')
                        if ds_id == matched_id:
                            continue
                        result = agent.execute_sql_query(
                            query=request.query,
                            df_id=ds_id,
                            client_id=request.client,
                            use_cache=request.use_cache
                        )
                        if result.success:
                            if best_result is None or result.value is not None:
                                best_result = result
                                best_dataset_id = ds_id
                                if result.value is not None:
                                    break  # Found a good result
            
            if best_result and best_result.success:
                # Format as human-like response
                natural_result = _format_natural_response(
                    query=request.query,
                    raw_result=best_result.result,
                    explanation=best_result.explanation,
                    llm_wrapper=llm
                )
                
                return QueryResponse(
                    success=True,
                    result=natural_result,
                    method=best_result.method,
                    explanation=best_result.explanation,
                    error=None,
                    query_id=query_id,
                    metadata={"dataset_id": best_dataset_id, "route": track, "raw_value": best_result.value}
                )
            
            # FALLBACK: RAG Semantic Search (disabled for structured datasets to minimize latency)
            if not datasets:  # only consider RAG when no structured data is available
                try:
                    from app.rag.ingest import get_rag_pipeline
                    rag = get_rag_pipeline()
                    if rag and rag.is_available:
                        rag_results = rag._ingestor.search(
                            query=request.query,
                            client_id=request.client,
                            top_k=3,
                            score_threshold=0.3
                        )
                        if rag_results:
                            context = "\n\n".join([
                                f"Source: {r.get('metadata', {}).get('table_name', 'data')}\n{r.get('content', '')[:800]}"
                                for r in rag_results[:3]
                            ])
                            prompt = f"Based on this data:\n\n{context}\n\nAnswer: {request.query}"
                            response = llm.invoke(prompt)
                            return QueryResponse(
                                success=True,
                                result=response,
                                method="rag:semantic",
                                explanation="Answer synthesized from indexed data",
                                query_id=query_id,
                                metadata={"route": track, "fallback": "rag"}
                            )
                except Exception as e:
                    logger.debug(f"RAG fallback failed: {e}")
            
            # FALLBACK: Comprehensive LLM analysis
            try:
                # Collect data from all datasets
                all_data_context = []
                for ds in datasets[:3]:
                    df = agent._get_dataframe(ds.get('dataset_id'))
                    if df is not None:
                        table_name = ds.get('dataset_id', '').split(':')[-1]
                        context_parts = [f"\n--- TABLE: {table_name} ---"]
                        context_parts.append(f"Columns: {list(df.columns)}")
                        context_parts.append(f"Data:\n{df.to_string()}")
                        all_data_context.append("\n".join(context_parts))
                
                if all_data_context:
                    full_context = "\n".join(all_data_context)[:6000]
                    prompt = f"""Analyze this data and answer:\n{full_context}\n\nQuestion: {request.query}"""
                    response = llm.invoke(prompt)
                    return QueryResponse(
                        success=True,
                        result=response,
                        method="llm:comprehensive",
                        explanation="Answer from comprehensive data analysis",
                        query_id=query_id,
                        metadata={"route": track, "fallback": "llm"}
                    )
            except Exception as e:
                logger.debug(f"LLM fallback failed: {e}")
            
            return QueryResponse(
                success=False,
                error=best_result.error if best_result else "Query execution failed",
                method="failed",
                query_id=query_id
            )
            
        elif track == TRACK_DOC:
            # Document RAG query
            from app.rag.ingest import get_document_ingestor
            from app.core.prompts import get_document_rag_prompt
            
            ingestor = get_document_ingestor()
            docs = ingestor.search(request.query, request.client)
            
            if not docs:
                return QueryResponse(
                    success=False,
                    error="No relevant documents found",
                    method="rag",
                    query_id=query_id
                )
            
            # Synthesize response
            context = "\n\n".join([d.get("content", "") for d in docs[:3]])
            prompt = get_document_rag_prompt(context, request.query)
            response = llm.invoke(prompt)
            
            return QueryResponse(
                success=True,
                result=response,
                method="rag:document",
                explanation=f"Found {len(docs)} relevant documents",
                query_id=query_id,
                metadata={"route": track, "docs": len(docs)}
            )
            
        elif track == TRACK_WEB:
            # Web search
            from app.tools.web_search import web_search
            
            result = web_search(request.query, num_results=3)
            
            if result.get("result") == "success":
                # Synthesize response
                from app.core.prompts import get_web_search_prompt
                
                snippets = "\n\n".join([
                    f"**{r['title']}**\n{r['snippet']}\nURL: {r['url']}"
                    for r in result.get("results", [])
                ])
                
                prompt = get_web_search_prompt(snippets, request.query)
                response = llm.invoke(prompt)
                
                return QueryResponse(
                    success=True,
                    result=response,
                    method="web:search",
                    explanation=f"Synthesized from {len(result.get('results', []))} web results",
                    query_id=query_id,
                    metadata={"route": track, "sources": result.get("results", [])}
                )
            else:
                return QueryResponse(
                    success=False,
                    error="Web search returned no results",
                    method="web:search",
                    query_id=query_id
                )
        
        return QueryResponse(
            success=False,
            error=f"Unknown track: {track}",
            query_id=query_id
        )
        
    except Exception as e:
        logger.error(f"Query error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/ai-ca/datasets")
async def list_datasets(client_id: str = Query(...), x_request_id: Optional[str] = Header(None)):
    request_id = x_request_id or generate_request_id()
    start_time = time.time()
    safe_client = normalize_client_id(client_id)
    
    try:
        from app.core.data_registry import get_data_registry
        registry = get_data_registry()
        datasets = registry.list_for_client(safe_client)
        
        log_request(request_id, "/v1/ai-ca/datasets", user_id=safe_client, duration_ms=(time.time() - start_time) * 1000)
        return make_response(
            data={"client_id": safe_client, "datasets": datasets, "count": len(datasets)},
            request_id=request_id
        )
    except Exception as e:
        log_request(request_id, "/v1/ai-ca/datasets", user_id=safe_client, status="error", duration_ms=(time.time() - start_time) * 1000)
        return JSONResponse(
            status_code=500,
            content=make_response(request_id=request_id, status="error", error_code="INTERNAL_ERROR", error_message=str(e))
        )


@app.get("/v1/ai-ca/metrics")
async def get_metrics(x_request_id: Optional[str] = Header(None)):
    request_id = x_request_id or generate_request_id()
    start_time = time.time()
    
    try:
        from app.core.llm_wrapper import get_llm_wrapper
        from app.core.data_registry import get_data_registry
        
        llm = get_llm_wrapper()
        registry = get_data_registry()
        
        log_request(request_id, "/v1/ai-ca/metrics", duration_ms=(time.time() - start_time) * 1000)
        return make_response(
            data={"llm": llm.get_metrics(), "datasets": len(registry.list_all()), "provider": llm.provider_name},
            request_id=request_id
        )
    except Exception as e:
        log_request(request_id, "/v1/ai-ca/metrics", status="error", duration_ms=(time.time() - start_time) * 1000)
        return make_response(request_id=request_id, status="error", error_code="INTERNAL_ERROR", error_message=str(e))


class IDCreateRequest(BaseModel):
    user_id: str
    namespace: Optional[str] = "default"
    meta: Optional[Dict[str, Any]] = None


class IDValidateRequest(BaseModel):
    id: str
    user_id: str


_id_store: Dict[str, List[Dict[str, Any]]] = defaultdict(list)


@app.post("/api/ids/create")
async def create_id(request: IDCreateRequest, x_request_id: Optional[str] = Header(None)):
    request_id = x_request_id or generate_request_id()
    start_time = time.time()
    safe_user = normalize_client_id(request.user_id)
    
    try:
        from app.core.id_generator import generate_short_id
        
        new_id = generate_short_id(request.namespace or "id")
        entry = {
            "id": new_id,
            "user_id": safe_user,
            "namespace": request.namespace or "default",
            "meta": request.meta or {},
            "created_at": get_iso_timestamp()
        }
        _id_store[safe_user].append(entry)
        
        log_request(request_id, "/api/ids/create", user_id=safe_user, duration_ms=(time.time() - start_time) * 1000)
        return make_response(data={"id": new_id, "user_id": safe_user}, request_id=request_id)
    except Exception as e:
        log_request(request_id, "/api/ids/create", user_id=safe_user, status="error", duration_ms=(time.time() - start_time) * 1000)
        return JSONResponse(
            status_code=500,
            content=make_response(request_id=request_id, status="error", error_code="INTERNAL_ERROR", error_message=str(e))
        )


@app.get("/api/ids/list")
async def list_ids(user_id: str = Query(...), x_request_id: Optional[str] = Header(None)):
    request_id = x_request_id or generate_request_id()
    start_time = time.time()
    safe_user = normalize_client_id(user_id)
    
    try:
        ids = _id_store.get(safe_user, [])
        log_request(request_id, "/api/ids/list", user_id=safe_user, duration_ms=(time.time() - start_time) * 1000)
        return make_response(data=ids, request_id=request_id)
    except Exception as e:
        log_request(request_id, "/api/ids/list", user_id=safe_user, status="error", duration_ms=(time.time() - start_time) * 1000)
        return JSONResponse(
            status_code=500,
            content=make_response(request_id=request_id, status="error", error_code="INTERNAL_ERROR", error_message=str(e))
        )


@app.post("/api/ids/validate")
async def validate_id(request: IDValidateRequest, x_request_id: Optional[str] = Header(None)):
    request_id = x_request_id or generate_request_id()
    start_time = time.time()
    safe_user = normalize_client_id(request.user_id)
    
    try:
        from app.core.id_generator import validate_user_id
        
        ids = _id_store.get(safe_user, [])
        match = next((entry for entry in ids if entry["id"] == request.id), None)
        valid = match is not None
        meta = match.get("meta", {}) if match else {}
        
        log_request(request_id, "/api/ids/validate", user_id=safe_user, duration_ms=(time.time() - start_time) * 1000)
        return make_response(data={"valid": valid, "meta": meta}, request_id=request_id)
    except Exception as e:
        log_request(request_id, "/api/ids/validate", user_id=safe_user, status="error", duration_ms=(time.time() - start_time) * 1000)
        return JSONResponse(
            status_code=500,
            content=make_response(request_id=request_id, status="error", error_code="INTERNAL_ERROR", error_message=str(e))
        )


# =============================================================================
# STREAMING QUERY ENDPOINT - Server-Sent Events for real-time token streaming
# =============================================================================
from fastapi.responses import StreamingResponse
import asyncio


class StreamQueryRequest(BaseModel):
    """Streaming query request."""
    client: str
    query: str
    dataset_id: Optional[str] = None


@app.post("/v1/ai-ca/query/stream")
async def stream_query(request: StreamQueryRequest):
    """
    Stream query response using Server-Sent Events (SSE).
    
    Postman Usage:
    - Method: POST
    - URL: http://localhost:8000/v1/ai-ca/query/stream
    - Body (raw JSON): {"client": "test_user", "query": "What is total revenue?"}
    - Headers: Accept: text/event-stream
    
    Response: Real-time token-by-token streaming visible in Postman.
    """
    from app.core.id_generator import generate_query_id
    
    query_id = generate_query_id()
    safe_client = normalize_client_id(request.client)
    
    async def generate_sse():
        """Generate Server-Sent Events stream."""
        try:
            # Send initial metadata
            yield f"event: start\ndata: {json.dumps({'query_id': query_id, 'client': safe_client, 'status': 'processing'})}\n\n"
            
            # Import dependencies
            from app.agents.data_analyst import get_data_analyst_agent
            from app.agents.router import get_router_agent
            from app.core.llm_wrapper import get_llm_wrapper
            from app.core.data_registry import get_data_registry
            
            agent = get_data_analyst_agent()
            router = get_router_agent()
            llm = get_llm_wrapper()
            registry = get_data_registry()
            
            # Get datasets
            datasets = registry.list_for_client(safe_client)
            has_data = len(datasets) > 0
            
            # Route query
            route_result = router.route(request.query, has_loaded_data=has_data)
            track = route_result.get("track", "TRACK_DATA")
            
            yield f"event: route\ndata: {json.dumps({'track': track, 'confidence': route_result.get('confidence', 0)})}\n\n"
            
            # Execute based on track
            result_text = ""
            method = "unknown"
            
            if track == "TRACK_DOC_SUMMARY" and datasets:
                # Summary mode - stream summaries
                summaries = []
                for i, ds in enumerate(datasets[:3]):
                    ds_id = ds.get("dataset_id", "")
                    sheet_name = ds_id.split(":")[-1] if ":" in ds_id else ds_id
                    yield f"event: progress\ndata: {json.dumps({'sheet': sheet_name, 'index': i+1, 'total': min(len(datasets), 3)})}\n\n"
                    
                    summary_result = agent.summarize_dataset(ds_id, client_id=safe_client)
                    if summary_result.get("value"):
                        summary_text = f"**{sheet_name}:** {summary_result['value']}"
                        summaries.append(summary_text)
                        # Stream each word for visibility
                        for word in summary_text.split():
                            yield f"event: token\ndata: {json.dumps({'token': word + ' '})}\n\n"
                            await asyncio.sleep(0.01)  # Small delay for visibility
                        newline_token = "\n\n"
                        yield f"event: token\ndata: {json.dumps({'token': newline_token})}\n\n"
                
                result_text = "\n\n".join(summaries)
                method = "summarize_dataset"
                
            elif track == "TRACK_DATA" and datasets:
                # Data query - find best match and execute
                matched_id = request.dataset_id
                if not matched_id:
                    matched_id = agent.match_dataset_by_query(request.query, datasets)
                if not matched_id and datasets:
                    matched_id = datasets[0].get("dataset_id")
                
                yield f"event: progress\ndata: {json.dumps({'step': 'executing_sql', 'dataset': matched_id})}\n\n"
                
                sql_result = agent.execute_sql_query(
                    query=request.query,
                    df_id=matched_id,
                    client_id=safe_client
                )
                
                if sql_result and sql_result.success:
                    result_text = str(sql_result.result)
                    method = sql_result.method
                    # Stream result
                    for word in result_text.split():
                        yield f"event: token\ndata: {json.dumps({'token': word + ' '})}\n\n"
                        await asyncio.sleep(0.01)
                else:
                    result_text = sql_result.error if sql_result else "Query execution failed"
                    method = "error"
                    
            else:
                # Fallback - use LLM directly
                yield f"event: progress\ndata: {json.dumps({'step': 'llm_generation'})}\n\n"
                
                try:
                    response = llm.invoke(f"Answer this query: {request.query}")
                    result_text = response
                    method = "llm_direct"
                    # Stream response
                    for word in result_text.split():
                        yield f"event: token\ndata: {json.dumps({'token': word + ' '})}\n\n"
                        await asyncio.sleep(0.01)
                except Exception as e:
                    result_text = f"Error: {str(e)}"
                    method = "error"
            
            # Send final result
            yield f"event: complete\ndata: {json.dumps({'query_id': query_id, 'result': result_text, 'method': method, 'success': method != 'error'})}\n\n"
            
        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield f"event: error\ndata: {json.dumps({'error': str(e), 'query_id': query_id})}\n\n"
    
    return StreamingResponse(
        generate_sse(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )


# CLI Runner
def run_cli():
    """Run CLI for interactive testing."""
    parser = argparse.ArgumentParser(description="AI-CA CLI")
    subparsers = parser.add_subparsers(dest="command")
    
    # Upload command
    upload_parser = subparsers.add_parser("upload", help="Upload file")
    upload_parser.add_argument("--file", required=True, help="File path")
    upload_parser.add_argument("--client", required=True, help="Client ID")
    upload_parser.add_argument("--ingest_all", action="store_true", help="Ingest all sheets")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Execute query")
    query_parser.add_argument("--client", required=True, help="Client ID")
    query_parser.add_argument("--query", required=True, help="Query string")
    query_parser.add_argument("--dataset", help="Dataset ID")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List datasets")
    list_parser.add_argument("--client", required=True, help="Client ID")
    
    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start API server")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Host")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port")
    
    args = parser.parse_args()
    
    if args.command == "upload":
        from app.ingest.excel_ingest import ExcelIngestor
        from app.agents.data_analyst import get_data_analyst_agent
        
        agent = get_data_analyst_agent()
        ingestor = ExcelIngestor()
        
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"Error: File not found: {args.file}")
            return
        
        with open(file_path, "rb") as f:
            content = f.read()
        
        def register_cb(dataset_id, df, metadata):
            agent.register_dataframe(
                dataset_id, df,
                preprocessing_report=metadata.get("preprocessing"),
                client_id=args.client
            )
            print(f"Registered: {dataset_id} ({len(df)} rows)")
        
        if args.ingest_all:
            results = ingestor.ingest_all_sheets(
                content, file_path.name, args.client, register_cb
            )
        else:
            result = ingestor.ingest_best_sheet(
                content, file_path.name, args.client, register_cb
            )
            results = [result]
        
        print(f"\nIngested {len([r for r in results if r.get('success')])} sheets")
        
    elif args.command == "query":
        from app.agents.data_analyst import get_data_analyst_agent
        from app.core.llm_wrapper import get_llm_wrapper
        
        llm = get_llm_wrapper()
        agent = get_data_analyst_agent(llm)
        
        datasets = agent.list_datasets_for_client(args.client)
        if not datasets:
            print("No datasets found. Upload data first.")
            return
        
        dataset_id = args.dataset or agent.match_dataset_by_query(args.query, datasets)
        if not dataset_id:
            print("Could not match query to dataset")
            return
        
        print(f"Using dataset: {dataset_id}")
        
        result = agent.execute_sql_query(
            query=args.query,
            df_id=dataset_id,
            client_id=args.client
        )
        
        print(f"\nSuccess: {result.success}")
        print(f"Result: {result.result}")
        print(f"Method: {result.method}")
        print(f"Explanation: {result.explanation}")
        if result.error:
            print(f"Error: {result.error}")
        
    elif args.command == "list":
        from app.core.data_registry import get_data_registry
        
        registry = get_data_registry()
        datasets = registry.list_for_client(args.client)
        
        print(f"\nDatasets for {args.client}:")
        for ds in datasets:
            print(f"  - {ds.get('dataset_id')}: {ds.get('rows', '?')} rows")
        
    elif args.command == "serve":
        uvicorn.run(
            "app.main:app",
            host=args.host,
            port=args.port,
            reload=False
        )
        
    else:
        parser.print_help()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_cli()
    else:
        # Default: start server
        from app.config import settings
        uvicorn.run(
            "app.main:app",
            host=settings.host,
            port=settings.port,
            reload=settings.debug
        )
