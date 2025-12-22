"""
AI-CA Main Application - FastAPI entrypoint with CLI runner.
Production-grade Agentic AI Chartered Accountant RAG System.
"""
import sys
import os

# Windows DLL path fix for torch (must be before any other imports)
if sys.platform == 'win32':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    # Add torch lib to DLL search path if needed
    torch_lib = os.path.join(os.path.dirname(sys.executable), 'Lib', 'site-packages', 'torch', 'lib')
    if os.path.exists(torch_lib):
        os.environ['PATH'] = torch_lib + os.pathsep + os.environ.get('PATH', '')
        try:
            os.add_dll_directory(torch_lib)
        except (AttributeError, OSError):
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
    client: str
    query: str
    dataset_id: Optional[str] = None
    use_cache: bool = True


class QueryResponse(BaseModel):
    success: bool
    result: Any = None
    method: str = "unknown"
    explanation: str = ""
    error: Optional[str] = None
    metadata: Dict[str, Any] = {}


class UploadResponse(BaseModel):
    success: bool
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
    """Upload and ingest Excel/CSV file."""
    try:
        from app.ingest.excel_ingest import ExcelIngestor
        from app.core.data_registry import get_data_registry
        from app.agents.data_analyst import get_data_analyst_agent
        
        registry = get_data_registry()
        agent = get_data_analyst_agent()
        ingestor = ExcelIngestor()
        
        # Read file content
        content = await file.read()
        filename = file.filename or "uploaded.xlsx"
        
        # Register callback
        def register_cb(dataset_id: str, df, metadata: dict):
            agent.register_dataframe(
                dataset_id, df,
                preprocessing_report=metadata.get("preprocessing"),
                client_id=client_id
            )
        
        # Ingest
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
        
        # Count successes
        successes = [r for r in results if r.get("success")]
        
        return UploadResponse(
            success=len(successes) > 0,
            datasets=results
        )
        
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/ai-ca/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Execute analytical query."""
    try:
        from app.agents.router import get_router_agent, TRACK_DATA, TRACK_DOC, TRACK_WEB
        from app.agents.data_analyst import get_data_analyst_agent
        from app.core.llm_wrapper import get_llm_wrapper
        from app.core.data_registry import get_data_registry
        
        llm = get_llm_wrapper()
        router = get_router_agent(llm)
        agent = get_data_analyst_agent(llm)
        registry = get_data_registry()
        
        # Route query
        route_result = router.route(request.query, f"Client: {request.client}")
        track = route_result.get("track", TRACK_DATA)
        
        logger.info(f"Routed to {track} (confidence: {route_result.get('confidence', 0):.2f})")
        
        if track == TRACK_DATA:
            # Determine dataset
            dataset_id = request.dataset_id
            
            if not dataset_id:
                # Find matching dataset
                datasets = agent.list_datasets_for_client(request.client)
                if not datasets:
                    return QueryResponse(
                        success=False,
                        error="No datasets found for client. Please upload data first."
                    )
                dataset_id = agent.match_dataset_by_query(request.query, datasets)
            
            if not dataset_id:
                return QueryResponse(
                    success=False,
                    error="Could not match query to any dataset"
                )
            
            # Execute query
            result = agent.execute_sql_query(
                query=request.query,
                df_id=dataset_id,
                client_id=request.client,
                use_cache=request.use_cache
            )
            
            return QueryResponse(
                success=result.success,
                result=result.result,
                method=result.method,
                explanation=result.explanation,
                error=result.error,
                metadata={"dataset_id": dataset_id, "route": track}
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
                    method="rag"
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
                    metadata={"route": track, "sources": result.get("results", [])}
                )
            else:
                return QueryResponse(
                    success=False,
                    error="Web search returned no results",
                    method="web:search"
                )
        
        return QueryResponse(
            success=False,
            error=f"Unknown track: {track}"
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
