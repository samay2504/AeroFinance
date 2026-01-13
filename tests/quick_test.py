"""Quick test for new router functionality."""
import sys
sys.path.insert(0, '.')

from app.core.dll_fix import apply_dll_fix
apply_dll_fix()

from app.agents.router import RouterAgent, TRACK_DOC_SUMMARY, TRACK_OUT_OF_DOMAIN

def test_summary():
    router = RouterAgent()
    
    # Test summary detection
    r = router.route('what is this data about', has_loaded_data=True)
    print(f"Summary test: Track={r['track']}, is_summary={r.get('is_summary')}")
    assert r['track'] == TRACK_DOC_SUMMARY, f"Expected TRACK_DOC_SUMMARY, got {r['track']}"
    assert r.get('is_summary') == True
    print("✅ Summary detection: PASS")

def test_out_of_domain():
    router = RouterAgent()
    
    r = router.route('tell me a joke')
    print(f"OOD test: Track={r['track']}")
    assert r['track'] == TRACK_OUT_OF_DOMAIN, f"Expected TRACK_OUT_OF_DOMAIN, got {r['track']}"
    print("✅ Out-of-domain detection: PASS")

def test_analytical():
    router = RouterAgent()
    
    r = router.route('calculate total revenue', has_loaded_data=True)
    print(f"Analytical test: is_analytical={r.get('is_analytical')}")
    assert r.get('is_analytical') == True
    print("✅ Analytical detection: PASS")

def test_pii_masking():
    from app.core.llm_wrapper import _mask_pii
    
    data = {"name": "John", "email": "john@example.com", "revenue": 1000}
    masked = _mask_pii(data)
    
    assert masked["name"] == "<MASKED>"
    assert masked["email"] == "<MASKED>"
    assert masked["revenue"] == 1000
    print("✅ PII masking: PASS")


def test_log_classification():
    """Test log value tier classification."""
    from app.core.llm_wrapper import classify_log_value
    
    # High value: has query + result + provenance + success
    high_value_entry = {
        "query_text": "What is the revenue?",
        "result": 1000000,
        "provenance": [{"dataset_id": "test"}],
        "status": "success"
    }
    assert classify_log_value(high_value_entry) == "high_value"
    
    # High value: has executed code
    code_entry = {
        "query_text": "Calculate growth",
        "executed_code_hash": "abc123",
        "status": "failed"
    }
    assert classify_log_value(code_entry) == "high_value"
    
    # Medium value: has operational metrics only
    medium_entry = {
        "latency_ms": 234,
        "tokens_used": 100
    }
    assert classify_log_value(medium_entry) == "medium_value"
    
    # Low value: minimal data
    low_entry = {
        "debug_trace": "some trace"
    }
    assert classify_log_value(low_entry) == "low_value"
    
    print("✅ Log classification: PASS")


def test_high_value_extraction():
    """Test high-value field extraction for archival."""
    from app.core.llm_wrapper import extract_high_value_fields
    
    full_entry = {
        "log_id": "log_123",
        "timestamp_utc": "2026-01-13T18:00:00Z",
        "client_id": "client_456",
        "query_text": "What is the revenue?",
        "result": 1000000,
        "analysis_method": "sql_duckdb",
        "executed_code_hash": "abc123",
        "provenance": [{"dataset_id": "ds1"}],
        "llm_provider": "groq",
        "latency_ms": 234,
        "status": "success",
        # Fields that should NOT be archived:
        "tokens_used": 1500,
        "cache_hit": False,
        "intermediate_reasoning": "blah blah"
    }
    
    extracted = extract_high_value_fields(full_entry)
    
    # Verify essential fields are present
    assert extracted["log_id"] == "log_123"
    assert extracted["user_query"] == "What is the revenue?"
    assert extracted["final_response"] == 1000000
    assert extracted["method"] == "sql_duckdb"
    assert extracted["provenance"] == [{"dataset_id": "ds1"}]
    
    # Verify non-essential fields are NOT present
    assert "tokens_used" not in extracted
    assert "cache_hit" not in extracted
    assert "intermediate_reasoning" not in extracted
    
    print("✅ High-value extraction: PASS")


def test_retention_config():
    """Test retention config defaults."""
    from app.core.llm_wrapper import LOG_RETENTION_CONFIG
    
    assert LOG_RETENTION_CONFIG["high_value"]["retention_days"] >= 365
    assert LOG_RETENTION_CONFIG["medium_value"]["retention_days"] >= 7
    assert LOG_RETENTION_CONFIG["low_value"]["retention_hours"] >= 24
    assert LOG_RETENTION_CONFIG["redis"]["ttl_seconds"] >= 60
    
    print("✅ Retention config: PASS")

if __name__ == "__main__":
    test_summary()
    test_out_of_domain()
    test_analytical()
    test_pii_masking()
    test_log_classification()
    test_high_value_extraction()
    test_retention_config()
    print("\n✅ ALL TESTS PASSED")

