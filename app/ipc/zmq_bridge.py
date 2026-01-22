"""
ZeroMQ Bridge - Optional IPC for Node.js integration.
Listens on unix socket and processes JSON requests.
"""
import logging
import json
import threading
from typing import Dict, Any, Optional, Callable
import os

logger = logging.getLogger(__name__)

try:
    import zmq
    ZMQ_AVAILABLE = True
except ImportError:
    ZMQ_AVAILABLE = False


class ZMQBridge:
    """
    ZeroMQ IPC bridge for external process integration.
    Listens for JSON requests and routes to handlers.
    """

    def __init__(
        self,
        socket_path: str = "ipc:///tmp/ai_ca.sock",
        enabled: bool = False
    ):
        self.socket_path = socket_path
        self.enabled = enabled and ZMQ_AVAILABLE
        self._context = None
        self._socket = None
        self._thread = None
        self._running = False
        self._handlers: Dict[str, Callable] = {}

    def register_handler(self, action: str, handler: Callable[[Dict], Dict]):
        """Register a handler for an action type."""
        self._handlers[action] = handler
        logger.info(f"Registered ZMQ handler: {action}")

    def start(self):
        """Start the ZMQ listener in a background thread."""
        if not self.enabled:
            logger.info("ZMQ bridge disabled")
            return False

        try:
            self._context = zmq.Context()
            self._socket = self._context.socket(zmq.REP)
            self._socket.bind(self.socket_path)
            self._running = True
            
            self._thread = threading.Thread(target=self._listen_loop, daemon=True)
            self._thread.start()
            
            logger.info(f"ZMQ bridge started on {self.socket_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start ZMQ bridge: {e}")
            return False

    def stop(self):
        """Stop the ZMQ listener."""
        self._running = False
        
        if self._socket:
            try:
                self._socket.close()
            except Exception:
                pass
                
        if self._context:
            try:
                self._context.term()
            except Exception:
                pass
                
        logger.info("ZMQ bridge stopped")

    def _listen_loop(self):
        """Main listening loop."""
        while self._running:
            try:
                # Non-blocking receive with timeout
                if self._socket.poll(timeout=1000):
                    message = self._socket.recv_string()
                    response = self._process_message(message)
                    self._socket.send_string(json.dumps(response))
            except zmq.ZMQError as e:
                if self._running:
                    logger.error(f"ZMQ error: {e}")
            except Exception as e:
                if self._running:
                    logger.error(f"ZMQ listener error: {e}")

    def _process_message(self, message: str) -> Dict[str, Any]:
        from app.core.id_generator import generate_request_id
        
        try:
            data = json.loads(message)
        except json.JSONDecodeError as e:
            req_id = generate_request_id()
            return {"status": "error", "request_id": req_id, "data": None, "error": {"code": "INVALID_JSON", "message": str(e)}}

        action = data.get("action")
        request_id = data.get("request_id") or generate_request_id()
        payload = data.get("payload", {})

        if not action:
            return {"status": "error", "request_id": request_id, "data": None, "error": {"code": "MISSING_ACTION", "message": "Missing action field"}}

        handler = self._handlers.get(action)
        if not handler:
            return {"status": "error", "request_id": request_id, "data": None, "error": {"code": "UNKNOWN_ACTION", "message": f"Unknown action: {action}", "available_actions": list(self._handlers.keys())}}

        try:
            result = handler(payload)
            is_streaming = result.get("streaming", False) if isinstance(result, dict) else False
            return {"status": "ok", "request_id": request_id, "data": result, "error": None, "meta": {"action": action, "streaming": is_streaming}}
        except Exception as e:
            logger.error(f"Handler error for {action}: {e}")
            return {"status": "error", "request_id": request_id, "data": None, "error": {"code": "HANDLER_ERROR", "message": str(e)}}



# Default handlers
def _query_handler(payload: Dict) -> Dict:
    """Handle query requests with unique ID tracking and human-like responses."""
    from app.agents.data_analyst import get_data_analyst_agent
    from app.agents.router import get_router_agent, TRACK_DATA, TRACK_DOC_SUMMARY, TRACK_OUT_OF_DOMAIN
    from app.core.llm_wrapper import get_llm_wrapper
    from app.core.id_generator import normalize_client_id, generate_query_id
    
    agent = get_data_analyst_agent()
    llm = get_llm_wrapper()
    router = get_router_agent(llm)
    
    query = payload.get("query", "")
    df_id = payload.get("dataset_id", "")
    client_id = payload.get("client_id", "")
    
    if not query:
        return {"error": "Missing query"}
    
    # Normalize client_id and generate query_id
    safe_client_id = normalize_client_id(client_id)
    query_id = generate_query_id()
    
    # Check if data is loaded
    datasets = agent.list_datasets_for_client(safe_client_id)
    has_loaded_data = len(datasets) > 0
    
    # Route query
    route_result = router.route(query, f"Client: {client_id}", has_loaded_data=has_loaded_data)
    track = route_result.get("track", TRACK_DATA)
    
    # Handle out-of-domain
    if track == TRACK_OUT_OF_DOMAIN:
        return {
            "success": True,
            "result": "I'm an AI Chartered Accountant assistant. I can help you with financial data analysis. Please ask me something related to your financial data!",
            "method": "out_of_domain",
            "natural_response": True,
            "query_id": query_id,
            "client_id": safe_client_id
        }
    
    # Handle summary
    if track == TRACK_DOC_SUMMARY:
        summaries = []
        for ds in datasets[:5]:
            ds_id = ds.get("dataset_id", "")
            result = agent.summarize_dataset(ds_id, client_id=safe_client_id)
            if result.get("value"):
                sheet_name = ds_id.split(":")[-1]
                summaries.append(f"{sheet_name}: {result['value']}")
        
        if summaries:
            return {
                "success": True,
                "result": "\n\n".join(summaries),
                "method": "summarize_dataset",
                "natural_response": True,
                "query_id": query_id,
                "client_id": safe_client_id
            }
    
    # Regular data query
    if not df_id and datasets:
        # Auto-match dataset
        df_id = agent.match_dataset_by_query(query, datasets)
    
    if not df_id:
        return {"error": "No dataset found for query"}
    
    result = agent.execute_sql_query(query, df_id, client_id=safe_client_id)
    
    # Format as human-like response
    natural_response = _format_natural_response(
        query=query,
        raw_result=result.result,
        explanation=result.explanation,
        llm=llm
    )
    
    return {
        "success": result.success,
        "result": natural_response if result.success else result.result,
        "raw_value": result.value,
        "method": result.method,
        "explanation": result.explanation,
        "query_id": query_id,
        "client_id": safe_client_id,
        "natural_response": True
    }


# Import centralized response formatter from prompts
from app.core.prompts import format_natural_response as _format_natural_response


def _list_datasets_handler(payload: Dict) -> Dict:
    """Handle list datasets request."""
    from app.core.data_registry import get_data_registry
    
    registry = get_data_registry()
    client_id = payload.get("client_id", "")
    
    if client_id:
        datasets = registry.list_for_client(client_id)
    else:
        datasets = [{"dataset_id": d} for d in registry.list_all()]
    
    return {"datasets": datasets}


def _stream_query_handler(payload: Dict) -> Dict:
    """Handle streaming query requests with token-by-token delivery."""
    from app.agents.data_analyst import get_data_analyst_agent
    from app.core.llm_wrapper import get_llm_wrapper
    from app.core.id_generator import normalize_client_id, generate_query_id, generate_short_id
    import time
    
    agent = get_data_analyst_agent()
    llm = get_llm_wrapper()
    
    query = payload.get("query", "")
    client_id = payload.get("client_id", "")
    chat_id = payload.get("chat_id") or generate_short_id("chat")
    
    if not query:
        return {"error": "Missing query"}
    
    # Normalize IDs
    safe_client_id = normalize_client_id(client_id)
    query_id = generate_query_id()
    stream_channel = f"chat:{chat_id}"
    
    # Collect streaming tokens
    tokens = []
    first_token_time = None
    start_time = time.time()
    
    def on_token(token: str, seq: int):
        nonlocal first_token_time
        if first_token_time is None:
            first_token_time = time.time()
        tokens.append({"seq": seq, "token": token})
    
    def on_start(meta):
        logger.info(f"Streaming started: chat_id={chat_id}, channel={stream_channel}")
    
    def on_end(meta):
        logger.info(f"Streaming ended: {meta.total_tokens} tokens, first_token={meta.first_token_latency_ms:.0f}ms")
    
    def on_error(err):
        logger.error(f"Streaming error: {err}")
    
    try:
        # Use streaming API
        response = llm.stream_chat(
            prompt=query,
            on_token=on_token,
            on_start=on_start,
            on_end=on_end,
            on_error=on_error,
            metadata={
                "chat_id": chat_id,
                "client_id": safe_client_id,
                "request_id": query_id,
                "stream_channel": stream_channel
            }
        )
        
        elapsed_ms = int((time.time() - start_time) * 1000)
        first_token_latency = 0
        if first_token_time:
            first_token_latency = int((first_token_time - start_time) * 1000)
        
        return {
            "success": True,
            "result": response,
            "streaming": True,
            "tokens_count": len(tokens),
            "first_token_latency_ms": first_token_latency,
            "total_latency_ms": elapsed_ms,
            "query_id": query_id,
            "chat_id": chat_id,
            "stream_channel": stream_channel,
            "client_id": safe_client_id
        }
        
    except Exception as e:
        logger.error(f"Stream query error: {e}")
        return {
            "success": False,
            "error": str(e),
            "query_id": query_id,
            "chat_id": chat_id
        }


# Singleton instance
_bridge: Optional[ZMQBridge] = None


def get_zmq_bridge() -> ZMQBridge:
    """Get or create singleton ZMQ bridge."""
    global _bridge
    
    if _bridge is None:
        from app.config import settings
        _bridge = ZMQBridge(
            socket_path=settings.zmq.socket_path,
            enabled=settings.zmq.enabled
        )
        
        # Register default handlers
        _bridge.register_handler("query", _query_handler)
        _bridge.register_handler("stream_query", _stream_query_handler)
        _bridge.register_handler("list_datasets", _list_datasets_handler)
    
    return _bridge


__all__ = ["ZMQBridge", "get_zmq_bridge"]

