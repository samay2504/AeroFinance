"""
AI-CA Main Application - FastAPI entrypoint with CLI runner.
Production-grade Agentic AI Chartered Accountant RAG System.
"""
import sys
import os

# Windows DLL path fix
try:
    from app.core.dll_fix import apply_dll_fix
    apply_dll_fix()
except ImportError:
    pass

import logging
import argparse
import json
from typing import Any, Dict, List, Optional
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("ai-ca")


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


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
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
    
    return HealthResponse(
        status="ok",
        version="1.0.0",
        provider=components["llm"],
        components=components
    )


@app.post("/v1/ai-ca/upload", response_model=UploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    client_id: str = Form(...),
    ingest_all: bool = Form(True)
):
    """Upload and ingest Excel/CSV/JSON file with unique document ID generation."""
    try:
        from app.core.data_registry import get_data_registry
        from app.agents.data_analyst import get_data_analyst_agent
        from app.core.id_generator import generate_doc_id, generate_dataset_id
        
        registry = get_data_registry()
        agent = get_data_analyst_agent()
        
        # Read file content
        content = await file.read()
        filename = file.filename or "uploaded_file"
        
        # Generate unique document ID
        doc_id = generate_doc_id(client_id, filename)
        logger.info(f"Generated doc_id: {doc_id} for {filename}")
        
        # Register callback with proper hierarchical IDs
        def register_cb(dataset_id: str, df, metadata: dict):
            # Build proper hierarchical dataset ID: client:doc:sheet
            sheet_name = dataset_id.split(':')[-1] if ':' in dataset_id else dataset_id
            full_dataset_id = generate_dataset_id(client_id, doc_id, sheet_name)
            
            agent.register_dataframe(
                full_dataset_id, df,
                preprocessing_report=metadata.get("preprocessing"),
                client_id=client_id
            )
            # Update the metadata with the full ID
            metadata["full_dataset_id"] = full_dataset_id
        
        # Detect file type and use appropriate ingestor
        ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
        
        if ext == 'json':
            # JSON file
            from app.ingest.json_ingest import JSONIngestor
            ingestor = JSONIngestor()
            result = ingestor.ingest_json_file(
                file_content=content,
                filename=filename,
                client_id=client_id,
                register_callback=register_cb
            )
            results = result.get("datasets", [])
            success = result.get("success", False)
            
        elif ext in ['xlsx', 'xls', 'csv']:
            # Excel/CSV file
            from app.ingest.excel_ingest import ExcelIngestor
            ingestor = ExcelIngestor()
            
            if ingest_all:
                results = ingestor.ingest_all_sheets(
                    file_content=content,
                    filename=filename,
                    client_id=client_id,
                    register_callback=register_cb
                )
            else:
                result = ingestor.ingest_best_sheet(
                    file_content=content,
                    filename=filename,
                    client_id=client_id,
                    register_callback=register_cb
                )
                results = [result]
            success = any(r.get("success") for r in results)
            
        else:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type: .{ext}. Supported: .xlsx, .xls, .csv, .json"
            )
        
        return UploadResponse(
            success=success,
            doc_id=doc_id,
            datasets=results
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
                    llm=llm
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
            
            # FALLBACK: RAG Semantic Search
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
async def list_datasets(client_id: str = Query(...)):
    """List datasets for a client."""
    try:
        from app.core.data_registry import get_data_registry
        
        registry = get_data_registry()
        datasets = registry.list_for_client(client_id)
        
        return {
            "success": True,
            "client_id": client_id,
            "datasets": datasets,
            "count": len(datasets)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/ai-ca/metrics")
async def get_metrics():
    """Get system metrics."""
    try:
        from app.core.llm_wrapper import get_llm_wrapper
        from app.core.data_registry import get_data_registry
        
        llm = get_llm_wrapper()
        registry = get_data_registry()
        
        return {
            "llm": llm.get_metrics(),
            "datasets": len(registry.list_all()),
            "provider": llm.provider_name
        }
        
    except Exception as e:
        return {"error": str(e)}


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
