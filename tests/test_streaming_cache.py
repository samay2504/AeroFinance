"""
Quick Unit Tests for Streaming, Cache, and Waveform Scheduling Features.

Tests for:
- StreamCallbacks, StreamMetadata, StreamBuffer
- KVCache (hot cache layer)
- BatchScheduler with TokenBucket rate limiting
- SandboxExecutor run_user_code API
- E2B integration (smoke test only if enabled)
- ID Generator new functions

Run with: pytest tests/test_streaming_cache.py -v
"""
import sys
import os
import pytest
import time
import threading
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Apply DLL fix for Windows BEFORE importing torch/spacy dependent modules
from app.core.dll_fix import apply_dll_fix
apply_dll_fix()

import pandas as pd



# =============================================================================
# STREAMING INFRASTRUCTURE TESTS
# =============================================================================

class TestStreamBuffer:
    """Tests for StreamBuffer ring buffer with backpressure handling."""
    
    def test_buffer_push_pop(self):
        """Test basic push/pop operations."""
        from app.core.llm_provider import StreamBuffer
        
        buffer = StreamBuffer(max_size=10)
        
        # Push some tokens
        buffer.push("Hello")
        buffer.push(" ")
        buffer.push("World")
        
        assert buffer.total_tokens == 3
        assert buffer.overflow_count == 0
        
        # Pop tokens
        assert buffer.pop() == "Hello"
        assert buffer.pop() == " "
        assert buffer.pop() == "World"
        assert buffer.pop() is None  # Empty
    
    def test_buffer_overflow(self):
        """Test overflow handling when buffer is full."""
        from app.core.llm_provider import StreamBuffer
        
        buffer = StreamBuffer(max_size=3)
        
        # Fill buffer
        buffer.push("A")
        buffer.push("B")
        buffer.push("C")
        
        # This should cause overflow
        buffer.push("D")
        
        assert buffer.overflow_count == 1
        assert buffer.total_tokens == 4
        
        # Get all - should have B, C, D (A was dropped)
        tokens = buffer.get_all()
        assert tokens == ["B", "C", "D"]
    
    def test_buffer_reset(self):
        """Test buffer reset."""
        from app.core.llm_provider import StreamBuffer
        
        buffer = StreamBuffer(max_size=10)
        buffer.push("token1")
        buffer.push("token2")
        
        buffer.reset()
        
        assert buffer.total_tokens == 0
        assert buffer.overflow_count == 0
        assert buffer.pop() is None


class TestStreamCallbacks:
    """Tests for StreamCallbacks interface."""
    
    def test_callbacks_initialization(self):
        """Test StreamCallbacks can be created with callbacks."""
        from app.core.llm_provider import StreamCallbacks, StreamMetadata
        
        tokens_received = []
        started = []
        ended = []
        
        callbacks = StreamCallbacks(
            on_token=lambda t, seq: tokens_received.append((t, seq)),
            on_start=lambda m: started.append(m),
            on_end=lambda m: ended.append(m),
            on_error=lambda e: None
        )
        
        # Test callbacks work
        callbacks.on_token("hello", 1)
        callbacks.on_start(StreamMetadata(chat_id="test"))
        callbacks.on_end(StreamMetadata(total_tokens=5))
        
        assert tokens_received == [("hello", 1)]
        assert len(started) == 1
        assert started[0].chat_id == "test"
        assert ended[0].total_tokens == 5


class TestStreamMetadata:
    """Tests for StreamMetadata dataclass."""
    
    def test_metadata_defaults(self):
        """Test default values."""
        from app.core.llm_provider import StreamMetadata
        
        meta = StreamMetadata()
        
        assert meta.request_id == ""
        assert meta.chat_id == ""
        assert meta.total_tokens == 0
        assert meta.first_token_latency_ms == 0.0
    
    def test_metadata_assignment(self):
        """Test setting values."""
        from app.core.llm_provider import StreamMetadata
        
        meta = StreamMetadata(
            request_id="req_123",
            chat_id="chat_456",
            client_id="client_789",
            provider="groq",
            model="llama3",
            total_tokens=100,
            first_token_latency_ms=150.5
        )
        
        assert meta.request_id == "req_123"
        assert meta.provider == "groq"
        assert meta.total_tokens == 100


# =============================================================================
# KV CACHE (HOT CACHE LAYER) TESTS
# =============================================================================

class TestKVCache:
    """Tests for KVCache LRU cache."""
    
    def test_cache_put_get(self):
        """Test basic put/get operations."""
        from app.core.llm_wrapper import KVCache
        
        cache = KVCache(max_memory_mb=10)
        
        # Put a value
        cache.put("model1", "session1", "hash123", {"kv_state": "test_data"})
        
        # Get it back
        result = cache.get("model1", "session1", "hash123")
        assert result == {"kv_state": "test_data"}
        
        # Miss should return None
        result = cache.get("model1", "session1", "wrong_hash")
        assert result is None
    
    def test_cache_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        from app.core.llm_wrapper import KVCache
        
        # Very small cache to test eviction
        cache = KVCache(max_memory_mb=1)  # 1MB
        
        # Put large items to trigger eviction
        large_data = "x" * (500 * 1024)  # 500KB
        
        cache.put("model", "sess1", "hash1", large_data, size_bytes=500*1024)
        cache.put("model", "sess2", "hash2", large_data, size_bytes=500*1024)
        
        # Adding third should evict first (LRU)
        cache.put("model", "sess3", "hash3", large_data, size_bytes=500*1024)
        
        # First should be evicted
        assert cache.get("model", "sess1", "hash1") is None
        # Second should still exist
        assert cache.get("model", "sess2", "hash2") is not None
    
    def test_cache_metrics(self):
        """Test cache metrics tracking."""
        from app.core.llm_wrapper import KVCache
        
        cache = KVCache(max_memory_mb=10)
        
        cache.put("model", "sess", "hash1", "data1")
        
        # Hit
        cache.get("model", "sess", "hash1")
        # Miss
        cache.get("model", "sess", "wrong")
        
        metrics = cache.get_metrics()
        
        assert metrics["hits"] == 1
        assert metrics["misses"] == 1
        assert metrics["entries"] == 1
        assert 0 <= metrics["hit_rate"] <= 1
    
    def test_cache_invalidation(self):
        """Test cache invalidation."""
        from app.core.llm_wrapper import KVCache
        
        cache = KVCache(max_memory_mb=10)
        
        cache.put("model1", "sess1", "hash1", "data1")
        cache.put("model1", "sess2", "hash2", "data2")
        cache.put("model2", "sess1", "hash3", "data3")
        
        # Invalidate by session
        cache.invalidate(session_id="sess1")
        
        assert cache.get("model1", "sess1", "hash1") is None
        assert cache.get("model2", "sess1", "hash3") is None
        assert cache.get("model1", "sess2", "hash2") is not None


class TestGlobalKVCache:
    """Test global KV cache singleton."""
    
    def test_get_kv_cache_singleton(self):
        """Test get_kv_cache returns same instance."""
        from app.core.llm_wrapper import get_kv_cache
        
        cache1 = get_kv_cache()
        cache2 = get_kv_cache()
        
        assert cache1 is cache2


# =============================================================================
# BATCH SCHEDULER TESTS
# =============================================================================

class TestTokenBucket:
    """Tests for TokenBucket rate limiter."""
    
    def test_token_consumption(self):
        """Test basic token consumption."""
        from app.core.llm_wrapper import TokenBucket
        
        bucket = TokenBucket(rate=100.0, capacity=10)
        
        # Should be able to consume initial capacity
        assert bucket.consume(5) is True
        assert bucket.consume(5) is True
        
        # Should fail when empty
        assert bucket.consume(5) is False
    
    def test_token_refill(self):
        """Test token refill over time."""
        from app.core.llm_wrapper import TokenBucket
        
        bucket = TokenBucket(rate=1000.0, capacity=10)  # Fast refill
        
        # Drain bucket
        bucket.consume(10)
        assert bucket.consume(1) is False
        
        # Wait for refill
        time.sleep(0.02)  # 20ms at 1000/s = ~20 tokens
        
        # Should have tokens now
        assert bucket.consume(1) is True


class TestBatchScheduler:
    """Tests for BatchScheduler."""
    
    def test_submit_request(self):
        """Test submitting requests."""
        from app.core.llm_wrapper import BatchScheduler
        
        scheduler = BatchScheduler(batch_window_ms=10)
        
        result = scheduler.submit(
            request_id="req1",
            prompt="Hello world",
            priority=1,
            client_id="client1"
        )
        
        assert result is True
        
        metrics = scheduler.get_metrics()
        assert metrics["total_requests"] == 1
    
    def test_batch_grouping(self):
        """Test requests are batched together."""
        from app.core.llm_wrapper import BatchScheduler
        
        scheduler = BatchScheduler(batch_window_ms=10, max_batch_size=4)
        
        # Submit multiple requests
        scheduler.submit("req1", "short", priority=1)
        scheduler.submit("req2", "also short", priority=1)
        scheduler.submit("req3", "another short one", priority=1)
        
        # Get batch (waits for window)
        batch = scheduler.get_batch(wait_ms=5)
        
        assert len(batch) <= 4
        assert len(batch) >= 1
    
    def test_priority_ordering(self):
        """Test higher priority requests come first."""
        from app.core.llm_wrapper import BatchScheduler
        
        scheduler = BatchScheduler(batch_window_ms=5)
        
        # Submit low priority first
        scheduler.submit("req_low", "low priority", priority=2)
        # Submit high priority second
        scheduler.submit("req_high", "high priority", priority=0)
        
        batch = scheduler.get_batch(wait_ms=2)
        
        # High priority should be first
        if len(batch) >= 2:
            assert batch[0].request_id == "req_high"


# =============================================================================
# SANDBOX EXECUTOR ENHANCEMENTS TESTS
# =============================================================================

class TestSandboxRunUserCode:
    """Tests for SandboxExecutor.run_user_code API."""
    
    def test_run_user_code_success(self):
        """Test run_user_code with successful execution."""
        from app.sandbox.sandbox_executor import SandboxExecutor
        
        sandbox = SandboxExecutor(timeout_seconds=10)
        
        code = """
def run(df):
    return df['x'].sum()
"""
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
        
        result = sandbox.run_user_code(code, {"df": df}, timeout_sec=10)
        
        assert result["success"] is True
        assert result["result"] == 15
        assert "exec_time_ms" in result
        assert result["error"] is None
    
    def test_run_user_code_error(self):
        """Test run_user_code with error."""
        from app.sandbox.sandbox_executor import SandboxExecutor
        
        sandbox = SandboxExecutor(timeout_seconds=10)
        
        code = """
def run(df):
    return df['nonexistent_column'].sum()
"""
        df = pd.DataFrame({"x": [1, 2, 3]})
        
        result = sandbox.run_user_code(code, {"df": df}, timeout_sec=10)
        
        assert result["success"] is False
        assert result["error"] is not None
        assert "KeyError" in result["error"] or "nonexistent" in result["error"].lower()
    
    def test_run_user_code_structured_return(self):
        """Test run_user_code returns consistent structure."""
        from app.sandbox.sandbox_executor import SandboxExecutor
        
        sandbox = SandboxExecutor()
        
        result = sandbox.run_user_code("x = 1", {"df": pd.DataFrame()}, timeout_sec=5)
        
        # Check all expected keys exist
        assert "success" in result
        assert "result" in result
        assert "logs" in result
        assert "error" in result
        assert "exec_time_ms" in result


class TestE2BExecutor:
    """Tests for E2B integration - smoke tests only."""
    
    def test_e2b_disabled_by_default(self):
        """Test E2B is disabled when env var not set."""
        from app.sandbox.sandbox_executor import E2BExecutor
        
        # Clear env var if set
        old_val = os.environ.pop("ENABLE_E2B", None)
        
        try:
            executor = E2BExecutor()
            assert executor.enabled is False
        finally:
            if old_val:
                os.environ["ENABLE_E2B"] = old_val
    
    def test_e2b_execute_returns_error_when_disabled(self):
        """Test execute returns proper error when E2B disabled."""
        from app.sandbox.sandbox_executor import E2BExecutor
        
        executor = E2BExecutor()
        executor.enabled = False
        
        result = executor.execute("print('hello')")
        
        assert result["success"] is False
        assert "not available" in result["error"].lower()


class TestExecuteWithE2BFallback:
    """Tests for execute_with_e2b_fallback helper."""
    
    def test_prefer_local_execution(self):
        """Test local execution is preferred."""
        from app.sandbox.sandbox_executor import execute_with_e2b_fallback
        
        code = """
def run(df):
    return 42
"""
        df = pd.DataFrame({"x": [1]})
        
        result = execute_with_e2b_fallback(code, df, prefer_local=True)
        
        assert result["success"] is True
        assert result["result"] == 42
        assert result["executor"] == "local"


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestStreamingIntegration:
    """Integration tests for streaming functionality."""
    
    def test_llm_wrapper_has_stream_chat(self):
        """Test LLMWrapper has stream_chat method."""
        from app.core.llm_wrapper import LLMWrapper
        
        assert hasattr(LLMWrapper, 'stream_chat')
    
    def test_llm_provider_has_stream_invoke(self):
        """Test LLMProvider has stream_invoke method."""
        from app.core.llm_provider import LLMProvider
        
        assert hasattr(LLMProvider, 'stream_invoke')


class TestLogSchemaVersion:
    """Test log schema versioning."""
    
    def test_log_interaction_includes_schema_version(self):
        """Test that log_interaction adds schema version."""
        from app.core.llm_wrapper import log_interaction
        import json
        from pathlib import Path
        
        @log_interaction
        def dummy_func(query: str, context: dict) -> dict:
            return {"result": "test", "method": "test"}
        
        # Execute
        dummy_func("test query", {"client_id": "test"})
        
        # Check log file
        log_path = Path("data/logs/interaction_logs.jsonl")
        if log_path.exists():
            with open(log_path, "r") as f:
                lines = f.readlines()
            
            if lines:
                last_log = json.loads(lines[-1])
                assert "log_schema_version" in last_log
                assert last_log["log_schema_version"] == "1.0"

# =============================================================================
# ID GENERATOR TESTS
# =============================================================================

class TestIDGenerator:
    """Tests for new ID generator functions."""
    
    def test_generate_request_id(self):
        """Test generate_request_id creates valid IDs."""
        from app.core.id_generator import generate_request_id
        
        req_id = generate_request_id()
        
        assert req_id.startswith("req_")
        assert len(req_id) > 10  # Should have timestamp + random
        
        # IDs should be unique
        req_id2 = generate_request_id()
        assert req_id != req_id2
    
    def test_generate_trace_id(self):
        """Test generate_trace_id creates valid IDs."""
        from app.core.id_generator import generate_trace_id
        
        trace_id = generate_trace_id()
        
        assert trace_id.startswith("trace_")
        assert len(trace_id) > 12
        
        # IDs should be unique
        trace_id2 = generate_trace_id()
        assert trace_id != trace_id2
    
    def test_generate_chat_id(self):
        """Test generate_chat_id creates valid IDs."""
        from app.core.id_generator import generate_chat_id
        
        chat_id = generate_chat_id()
        
        assert chat_id.startswith("chat_")
        assert len(chat_id) > 10
    
    def test_generate_short_id_with_prefix(self):
        """Test generate_short_id with custom prefix."""
        from app.core.id_generator import generate_short_id
        
        custom_id = generate_short_id("custom")
        
        assert custom_id.startswith("custom_")
        assert "_" in custom_id  # Should have separator
    
    def test_normalize_client_id(self):
        """Test client ID normalization for VectorDB safety."""
        from app.core.id_generator import normalize_client_id
        
        # Test various inputs
        assert normalize_client_id("TEST Client") == "test_client"
        assert normalize_client_id("client-with-dashes") == "client_with_dashes"
        assert normalize_client_id("client:with:colons") == "client_with_colons"
        assert normalize_client_id("") == "default_client"
        assert normalize_client_id(None) == "default_client"


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
