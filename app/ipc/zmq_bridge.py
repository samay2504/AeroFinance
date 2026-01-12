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
        """Process incoming JSON message."""
        try:
            data = json.loads(message)
        except json.JSONDecodeError as e:
            return {"error": f"Invalid JSON: {e}", "request_id": None}

        action = data.get("action")
        request_id = data.get("request_id")
        payload = data.get("payload", {})

        if not action:
            return {
                "error": "Missing action field",
                "request_id": request_id
            }

        handler = self._handlers.get(action)
        if not handler:
            return {
                "error": f"Unknown action: {action}",
                "request_id": request_id,
                "available_actions": list(self._handlers.keys())
            }

        try:
            result = handler(payload)
            return {
                "success": True,
                "request_id": request_id,
                "action": action,
                "result": result
            }
        except Exception as e:
            logger.error(f"Handler error for {action}: {e}")
            return {
                "error": str(e),
                "request_id": request_id,
                "action": action
            }


# Default handlers
def _query_handler(payload: Dict) -> Dict:
    """Handle query requests with unique ID tracking."""
    from app.agents.data_analyst import get_data_analyst_agent
    from app.core.id_generator import normalize_client_id, generate_query_id
    
    agent = get_data_analyst_agent()
    query = payload.get("query", "")
    df_id = payload.get("dataset_id", "")
    client_id = payload.get("client_id", "")
    
    if not query or not df_id:
        return {"error": "Missing query or dataset_id"}
    
    # Normalize client_id and generate query_id
    safe_client_id = normalize_client_id(client_id)
    query_id = generate_query_id()
    
    result = agent.execute_sql_query(query, df_id, client_id=safe_client_id)
    
    return {
        "success": result.success,
        "result": result.result,
        "method": result.method,
        "explanation": result.explanation,
        "query_id": query_id,
        "client_id": safe_client_id
    }


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
        _bridge.register_handler("list_datasets", _list_datasets_handler)
    
    return _bridge


__all__ = ["ZMQBridge", "get_zmq_bridge"]
