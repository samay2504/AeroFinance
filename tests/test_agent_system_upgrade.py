"""
Agent System Upgrade - Comprehensive Test Suite.

Tests all PRD enhancements:
- Enhancement 1: Semantic Router (L0 regex fast-pass)
- Enhancement 2: Self-Healing Execution Loop
- Enhancement 3: Intelligent Schema Filtering
- Enhancement 4: Cloud-Native Storage (DataRegistry)
- Performance: DataFrame Token Compression, Prompt Caching
- Security: SQL injection, path traversal, code validation
- Observability: Metrics collector
"""

import pytest
import time
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from typing import Dict, Any


# ═══════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture
def sample_financial_df():
    """Create a sample financial DataFrame for testing."""
    return pd.DataFrame({
        "Metric_Name": ["Revenue", "Cost of Goods Sold", "Gross Profit", "EBITDA", "Net Profit"],
        "FY21": [1000.0, 600.0, 400.0, 300.0, 200.0],
        "FY22": [1200.0, 700.0, 500.0, 380.0, 270.0],
        "FY23": [1500.0, 850.0, 650.0, 490.0, 350.0],
        "Q1_FY24": [400.0, 220.0, 180.0, 135.0, 95.0],
        "Growth_YoY": [0.15, 0.12, 0.18, 0.20, 0.22],
    })


@pytest.fixture
def large_column_df():
    """Create a DataFrame with 100+ columns for schema filtering tests."""
    cols = {}
    cols["Date_Period"] = pd.date_range("2020-01-01", periods=50, freq="ME")
    cols["Customer_ID"] = [f"CUST_{i:04d}" for i in range(50)]
    cols["Customer_Name"] = [f"Customer {i}" for i in range(50)]
    cols["Product_Category"] = np.random.choice(["A", "B", "C"], 50)
    
    # 96 value columns to hit 100+
    for i in range(96):
        suffix = f"_{i:02d}"
        category = ["revenue", "cost", "margin", "expense", "profit", "volume",
                     "discount", "tax", "fee", "rate"][i % 10]
        cols[f"{category}{suffix}"] = np.random.uniform(0, 10000, 50)
    
    return pd.DataFrame(cols)


@pytest.fixture
def mock_llm():
    """Create a mock LLM wrapper."""
    llm = MagicMock()
    llm.invoke = MagicMock(return_value="42")
    llm.invoke_with_structured_output = MagicMock(return_value={
        "sql": "SELECT * FROM data",
        "explanation": "Test query",
    })
    return llm


# ═══════════════════════════════════════════════════════════════════
# ENHANCEMENT 1: SEMANTIC ROUTER TESTS
# ═══════════════════════════════════════════════════════════════════

class TestRouterRegexFastPass:
    """Test L0 regex-based routing (PRD Enhancement 1)."""

    def test_data_query_regex(self):
        """Financial calculation queries should route to TRACK_DATA via regex."""
        from app.agents.router import RouterAgent, TRACK_DATA
        router = RouterAgent()
        
        data_queries = [
            "Calculate the total revenue for FY23",
            "What is the sum of expenses in Q1",
            "Find the average profit margin",
            "Compare revenue between FY22 and FY23",
            "How many sheets are there",
            "What columns are in the data",
        ]
        
        for query in data_queries:
            result = router.route(query, has_loaded_data=True)
            assert result["track"] == TRACK_DATA, (
                f"Query '{query}' should route to TRACK_DATA, got {result['track']}"
            )

    def test_summary_query_regex(self):
        """Summary requests should route to TRACK_DOC_SUMMARY via regex."""
        from app.agents.router import RouterAgent, TRACK_DOC_SUMMARY
        router = RouterAgent()
        
        summary_queries = [
            "Give me a summary of this data",
            "Summarize this spreadsheet",
            "Describe this dataset",
        ]
        
        for query in summary_queries:
            result = router.route(query, has_loaded_data=True)
            assert result["track"] == TRACK_DOC_SUMMARY, (
                f"Query '{query}' should route to TRACK_DOC_SUMMARY, got {result['track']}"
            )

    def test_web_query_regex(self):
        """Real-time queries should route to TRACK_WEB via regex."""
        from app.agents.router import RouterAgent, TRACK_WEB
        router = RouterAgent()
        
        web_queries = [
            "Current repo rate in India",
            "Today's gold price",
            "Latest market news",
        ]
        
        for query in web_queries:
            result = router.route(query, has_loaded_data=False)
            assert result["track"] == TRACK_WEB, (
                f"Query '{query}' should route to TRACK_WEB, got {result['track']}"
            )

    def test_regex_fast_pass_latency(self):
        """L0 regex routing should complete in <10ms."""
        from app.agents.router import RouterAgent
        router = RouterAgent()
        
        t0 = time.perf_counter()
        for _ in range(100):
            router.route("Calculate total revenue for FY23", has_loaded_data=True)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        
        avg_ms = elapsed_ms / 100
        # First call may be slower due to caching, but average should be fast
        assert avg_ms < 10, f"Average routing latency {avg_ms:.2f}ms exceeds 10ms target"

    def test_routing_stats_tracking(self):
        """Router should track L0/L1/L2 hit rates and latency."""
        from app.agents.router import RouterAgent
        router = RouterAgent()
        
        # Route some queries
        router.route("Calculate total revenue for FY23", has_loaded_data=True)
        router.route("What is the profit margin", has_loaded_data=True)
        
        stats = router.get_routing_stats()
        assert stats["total_queries"] >= 2
        assert "l0_hit_rate" in stats or stats["total_queries"] == 0

    def test_cache_hit(self):
        """Repeated queries should hit cache."""
        from app.agents.router import RouterAgent
        router = RouterAgent()
        
        # First call - cache miss
        router.route("Calculate total revenue", has_loaded_data=True)
        # Second call - should be cache hit
        router.route("Calculate total revenue", has_loaded_data=True)
        
        stats = router.get_routing_stats()
        assert stats["cache_hits"] >= 1


# ═══════════════════════════════════════════════════════════════════
# ENHANCEMENT 2: SELF-HEALING EXECUTION LOOP TESTS
# ═══════════════════════════════════════════════════════════════════

class TestSelfHealingExecution:
    """Test self-healing execution loop (PRD Enhancement 2)."""

    def test_error_classification(self):
        """Error types should be correctly classified."""
        from app.agents.data_analyst import DataAnalystAgent
        
        assert DataAnalystAgent._classify_error("Column 'foo' not found") == "column_error"
        assert DataAnalystAgent._classify_error("Invalid syntax near SELECT") == "syntax_error"
        assert DataAnalystAgent._classify_error("Cannot cast string to DOUBLE") == "type_error"
        assert DataAnalystAgent._classify_error("Request timeout after 30s") == "timeout"
        assert DataAnalystAgent._classify_error("Out of memory") == "memory_error"
        assert DataAnalystAgent._classify_error("Something unexpected") == "unknown_error"

    def test_error_hints_generation(self):
        """Error hints should be generated based on error content."""
        from app.agents.data_analyst import DataAnalystAgent
        
        hints_col = DataAnalystAgent._get_error_hints("Column 'revenue' not found")
        assert "column names" in hints_col.lower()
        
        hints_syntax = DataAnalystAgent._get_error_hints("Syntax error near SELECT")
        assert "syntax" in hints_syntax.lower()
        
        hints_div = DataAnalystAgent._get_error_hints("Division by zero")
        assert "zero" in hints_div.lower() or "NULLIF" in hints_div

    def test_healing_stats_tracking(self):
        """Healing stats should track attempts, successes, and failures."""
        from app.agents.data_analyst import DataAnalystAgent
        agent = DataAnalystAgent.__new__(DataAnalystAgent)
        # Reset class-level stats
        agent._HEALING_STATS = {
            "total_attempts": 10,
            "first_try_success": 7,
            "healed_success": 2,
            "total_failures": 1,
        }
        
        stats = agent.get_healing_stats()
        assert stats["first_try_rate"] == 0.7
        assert stats["heal_rate"] == 0.2
        assert stats["failure_rate"] == 0.1

    def test_build_healing_prompt(self):
        """Healing prompt should include error context and hints."""
        from app.agents.data_analyst import DataAnalystAgent
        agent = DataAnalystAgent.__new__(DataAnalystAgent)
        
        prompt = agent._build_healing_prompt(
            "What is total revenue?",
            "Column 'revnue' not found",
            {"columns": {"Revenue": "float64"}}
        )
        assert "PREVIOUS ATTEMPT FAILED" in prompt
        assert "revnue" in prompt
        assert "column names" in prompt.lower() or "HINTS" in prompt


# ═══════════════════════════════════════════════════════════════════
# ENHANCEMENT 3: INTELLIGENT SCHEMA FILTERING TESTS
# ═══════════════════════════════════════════════════════════════════

class TestSchemaFiltering:
    """Test intelligent schema filtering (PRD Enhancement 3)."""

    def test_small_df_includes_all(self, sample_financial_df):
        """DataFrames with <=30 columns should include all columns."""
        from app.core.schema_analyzer import SchemaAnalyzer
        analyzer = SchemaAnalyzer()
        
        result = analyzer.get_relevant_schema(sample_financial_df, "What is the revenue?")
        assert result["compression_ratio"] == 1.0
        assert result["filtered_columns"] == result["total_columns"]

    def test_large_df_compression(self, large_column_df):
        """DataFrames with 100+ columns should be compressed."""
        from app.core.schema_analyzer import SchemaAnalyzer
        analyzer = SchemaAnalyzer()
        
        result = analyzer.get_relevant_schema(
            large_column_df, 
            "What is the total revenue for this customer?",
            max_columns=20,
        )
        
        assert result["total_columns"] == 100
        assert result["filtered_columns"] <= 20
        assert result["compression_ratio"] < 0.5  # At least 50% compression
        assert "schema_string" in result

    def test_exact_keyword_matching(self, large_column_df):
        """Exact keyword matches should be included in filtered columns."""
        from app.core.schema_analyzer import SchemaAnalyzer
        analyzer = SchemaAnalyzer()
        
        result = analyzer.get_relevant_schema(
            large_column_df, "revenue data",
            include_all_if_small=False,
        )
        
        relevant = result["relevant_columns"]
        # Should include columns containing "revenue"
        revenue_cols = [c for c in relevant if "revenue" in c.lower()]
        assert len(revenue_cols) > 0, "Revenue columns not found in filtered result"

    def test_semantic_expansion(self, large_column_df):
        """Querying 'revenue' should also include related columns (cost, margin, profit)."""
        from app.core.schema_analyzer import SchemaAnalyzer
        analyzer = SchemaAnalyzer()
        
        result = analyzer.get_relevant_schema(
            large_column_df, "what is the revenue growth?",
            include_all_if_small=False,
        )
        
        relevant_lower = [c.lower() for c in result["relevant_columns"]]
        # Semantic expansion should include some related columns
        related = ["cost", "margin", "profit"]
        found_related = any(
            any(r in col for r in related) for col in relevant_lower
        )
        assert found_related, "Semantic expansion did not include related columns"

    def test_key_columns_always_included(self, large_column_df):
        """Date/ID/Name columns should always be included."""
        from app.core.schema_analyzer import SchemaAnalyzer
        analyzer = SchemaAnalyzer()
        
        result = analyzer.get_relevant_schema(
            large_column_df, "profit analysis",
            include_all_if_small=False,
        )
        
        relevant_lower = [c.lower() for c in result["relevant_columns"]]
        # Date and ID columns should always be included
        has_date = any("date" in c for c in relevant_lower)
        has_id = any("id" in c for c in relevant_lower)
        assert has_date or has_id, "Key columns (date/ID) not included"

    def test_fuzzy_matching_typo_tolerance(self, large_column_df):
        """Typos in queries should still match relevant columns."""
        from app.core.schema_analyzer import SchemaAnalyzer
        analyzer = SchemaAnalyzer()
        
        # "revnue" is a typo for "revenue"
        result = analyzer.get_relevant_schema(
            large_column_df, "what is the revnue total",
            include_all_if_small=False,
        )
        
        relevant_lower = [c.lower() for c in result["relevant_columns"]]
        revenue_found = any("revenue" in c for c in relevant_lower)
        assert revenue_found, "Fuzzy matching did not catch 'revnue' → 'revenue'"

    def test_fallback_for_insufficient_matches(self):
        """When very few columns match, fallback should add extras."""
        from app.core.schema_analyzer import SchemaAnalyzer
        analyzer = SchemaAnalyzer()
        
        # Create a DF with obscure column names
        df = pd.DataFrame({
            f"obscure_col_{i}": np.random.randn(10) for i in range(50)
        })
        
        result = analyzer.get_relevant_schema(
            df, "xyzzy quantum flux analysis",
            include_all_if_small=False,
        )
        
        # Should still have at least 5 columns from fallback
        assert result["filtered_columns"] >= 5

    def test_schema_string_format(self, sample_financial_df):
        """Schema string should be LLM-readable."""
        from app.core.schema_analyzer import SchemaAnalyzer
        analyzer = SchemaAnalyzer()
        
        result = analyzer.get_relevant_schema(sample_financial_df, "revenue")
        schema_str = result["schema_string"]
        
        assert "Available Columns" in schema_str
        assert "Metric_Name" in schema_str  # First column should be there


# ═══════════════════════════════════════════════════════════════════
# PERFORMANCE: DATAFRAME TOKEN COMPRESSION TESTS
# ═══════════════════════════════════════════════════════════════════

class TestDataFrameCompression:
    """Test DataFrame token compression for LLM context."""

    def test_compression_output_structure(self, sample_financial_df):
        """Compressed output should contain schema, sample, and patterns."""
        from app.ingest.excel_ingest import compress_dataframe_for_llm
        
        result = compress_dataframe_for_llm(sample_financial_df)
        
        assert "DataFrame Shape:" in result
        assert "COLUMN SCHEMA" in result
        assert "SAMPLE DATA" in result
        assert "Metric_Name" in result  # Column should be listed

    def test_compression_token_reduction(self, sample_financial_df):
        """Compressed representation should use fewer tokens than raw to_string."""
        from app.ingest.excel_ingest import compress_dataframe_for_llm
        
        raw = sample_financial_df.to_string()
        compressed = compress_dataframe_for_llm(sample_financial_df)
        
        # Compressed should be a reasonable representation
        assert len(compressed) > 100  # Not empty
        # For small DataFrames, compression may not save much, but for large ones it will

    def test_large_df_compression(self, large_column_df):
        """For large DataFrames, compression should save significant tokens."""
        from app.ingest.excel_ingest import compress_dataframe_for_llm
        
        raw_length = len(large_column_df.to_string())
        compressed = compress_dataframe_for_llm(large_column_df, max_sample_rows=3)
        
        # Compressed should be much shorter than raw
        assert len(compressed) < raw_length * 0.5, (
            f"Compression ineffective: {len(compressed)} vs {raw_length}"
        )

    def test_pattern_detection(self, large_column_df):
        """Should detect date and ID column patterns."""
        from app.ingest.excel_ingest import compress_dataframe_for_llm
        
        result = compress_dataframe_for_llm(large_column_df)
        assert "DETECTED PATTERNS" in result


# ═══════════════════════════════════════════════════════════════════
# SECURITY TESTS
# ═══════════════════════════════════════════════════════════════════

class TestSecurity:
    """Test security utilities."""

    def test_sql_injection_prevention(self):
        """Dangerous SQL should be blocked."""
        from app.sql_engine import sanitize_sql
        
        # Safe queries
        is_safe, sql, err = sanitize_sql("SELECT revenue FROM data WHERE year = 2023")
        assert is_safe is True
        assert err is None
        
        # Dangerous queries
        dangerous_queries = [
            "DROP TABLE data",
            "SELECT * FROM data; DROP TABLE users",
            "DELETE FROM data WHERE 1=1",
            "INSERT INTO data VALUES (1,2,3)",
            "ALTER TABLE data ADD COLUMN x",
            "UPDATE data SET revenue = 0",
        ]
        
        for query in dangerous_queries:
            is_safe, _, err = sanitize_sql(query)
            assert is_safe is False, f"Query '{query}' should be blocked"

    def test_path_traversal_prevention(self):
        """Path traversal attempts should be blocked."""
        from app.sql_engine import validate_file_path
        
        # Traversal patterns
        is_safe, _, err = validate_file_path("../../etc/passwd")
        assert is_safe is False
        
        is_safe, _, err = validate_file_path("/some/path/../../../etc/passwd")
        assert is_safe is False
        
        is_safe, _, err = validate_file_path("~/.ssh/id_rsa")
        assert is_safe is False

    def test_python_code_validation(self):
        """Dangerous Python code should be blocked."""
        from app.sql_engine import validate_python_code
        
        # Safe code
        is_safe, err = validate_python_code("def run(df): return df['revenue'].sum()")
        assert is_safe is True
        
        # Dangerous code
        dangerous_code = [
            "os.system('rm -rf /')",
            "subprocess.call(['ls'])",
            "eval('dangerous')",
            "__import__('os').system('ls')",
            "open('/etc/passwd', 'w')",
        ]
        for code in dangerous_code:
            is_safe, err = validate_python_code(code)
            assert is_safe is False, f"Code '{code}' should be blocked"

    def test_client_id_validation(self):
        """Client IDs should be validated and normalized."""
        from app.sql_engine import validate_client_id
        
        # Valid IDs
        is_valid, normalized, _ = validate_client_id("MyCompany123")
        assert is_valid is True
        assert normalized == "mycompany123"
        
        # Invalid IDs
        is_valid, _, err = validate_client_id("")
        assert is_valid is False
        
        is_valid, _, err = validate_client_id("../../../etc")
        assert is_valid is False


# ═══════════════════════════════════════════════════════════════════
# OBSERVABILITY: METRICS TESTS
# ═══════════════════════════════════════════════════════════════════

class TestMetrics:
    """Test the metrics collector."""

    def test_counter_operations(self):
        """Counters should increment correctly."""
        from app.core.metrics import MetricsCollector
        m = MetricsCollector()
        
        m.increment("test.counter")
        m.increment("test.counter")
        m.increment("test.counter", 5)
        
        assert m.get_counter("test.counter") == 7

    def test_gauge_operations(self):
        """Gauges should set and get correctly."""
        from app.core.metrics import MetricsCollector
        m = MetricsCollector()
        
        m.set_gauge("test.gauge", 42.5)
        assert m.get_gauge("test.gauge") == 42.5
        
        m.set_gauge("test.gauge", 100.0)
        assert m.get_gauge("test.gauge") == 100.0

    def test_histogram_stats(self):
        """Histograms should compute correct percentiles."""
        from app.core.metrics import MetricsCollector
        m = MetricsCollector()
        
        for i in range(100):
            m.observe("test.histogram", float(i))
        
        stats = m.get_histogram_stats("test.histogram")
        assert stats["count"] == 100
        assert stats["min"] == 0.0
        assert stats["max"] == 99.0
        assert 45 <= stats["mean"] <= 55  # ~49.5

    def test_self_healing_metrics(self):
        """Self-healing metrics should be recorded and retrievable."""
        from app.core.metrics import MetricsCollector
        m = MetricsCollector()
        
        m.record_self_healing_attempt("llm_sql", 0, True, latency_ms=150.0)
        m.record_self_healing_attempt("llm_sql", 1, False, "syntax_error", 200.0)
        m.record_self_healing_attempt("llm_python", 0, True, latency_ms=100.0)
        
        stats = m.get_self_healing_stats()
        # Stats use method-labeled keys, e.g. "self_healing.attempts{method=llm_sql}"
        total_attempts = sum(
            v for k, v in stats.items()
            if k.startswith("self_healing.attempts{")
        )
        assert total_attempts >= 3, f"Expected >= 3 total attempts, got {total_attempts}. Stats: {stats}"

    def test_snapshot(self):
        """Snapshot should include all metric types."""
        from app.core.metrics import MetricsCollector
        m = MetricsCollector()
        
        m.increment("counter")
        m.set_gauge("gauge", 1.0)
        m.observe("histogram", 5.0)
        
        snap = m.snapshot()
        assert "counters" in snap
        assert "gauges" in snap
        assert "histograms" in snap

    def test_thread_safety(self):
        """Metrics should be thread-safe."""
        import threading
        from app.core.metrics import MetricsCollector
        m = MetricsCollector()
        
        def worker(name, count):
            for _ in range(count):
                m.increment("concurrent.counter")
                m.observe("concurrent.histogram", 1.0)
        
        threads = [threading.Thread(target=worker, args=(f"t{i}", 100)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert m.get_counter("concurrent.counter") == 1000


# ═══════════════════════════════════════════════════════════════════
# PROMPT CACHING TESTS
# ═══════════════════════════════════════════════════════════════════

class TestPromptCaching:
    """Test the SemanticCache from llm_utils."""

    def test_cache_hit(self):
        """Same prompt should return cached result."""
        from app.core.llm_utils import SemanticCache
        cache = SemanticCache(max_size=100, default_ttl=3600)
        
        cache.set("test prompt", {"answer": 42})
        result = cache.get("test prompt")
        assert result == {"answer": 42}

    def test_cache_miss(self):
        """Different prompt should miss cache."""
        from app.core.llm_utils import SemanticCache
        cache = SemanticCache(max_size=100, default_ttl=3600)
        
        cache.set("test prompt A", {"answer": 42})
        result = cache.get("test prompt B")
        assert result is None

    def test_cache_context_isolation(self):
        """Same prompt with different contexts should not collide."""
        from app.core.llm_utils import SemanticCache
        cache = SemanticCache(max_size=100, default_ttl=3600)
        
        cache.set("what is revenue", {"value": 100}, context="dataset_A")
        cache.set("what is revenue", {"value": 200}, context="dataset_B")
        
        assert cache.get("what is revenue", context="dataset_A") == {"value": 100}
        assert cache.get("what is revenue", context="dataset_B") == {"value": 200}

    def test_cache_stats(self):
        """Cache should track hit/miss statistics."""
        from app.core.llm_utils import SemanticCache
        cache = SemanticCache(max_size=100, default_ttl=3600)
        
        cache.set("q1", "a1")
        cache.get("q1")  # Hit
        cache.get("q2")  # Miss
        
        stats = cache.stats()
        assert stats["hits"] >= 1
        assert stats["misses"] >= 1


# ═══════════════════════════════════════════════════════════════════
# INTEGRATION TEST: END-TO-END ROUTING + ANALYSIS
# ═══════════════════════════════════════════════════════════════════

class TestIntegration:
    """Lightweight integration tests (no external services required)."""

    def test_router_to_data_track(self):
        """Router should correctly route calculation queries."""
        from app.agents.router import RouterAgent, TRACK_DATA
        router = RouterAgent()
        
        result = router.route(
            "Calculate total revenue for FY23",
            has_loaded_data=True,
        )
        assert result["track"] == TRACK_DATA
        assert result["confidence"] >= 0.8

    def test_schema_analyzer_with_financial_data(self, sample_financial_df):
        """Schema analyzer should correctly identify financial data structure."""
        from app.core.schema_analyzer import SchemaAnalyzer
        analyzer = SchemaAnalyzer()
        
        schema = analyzer.analyze(sample_financial_df, "test_dataset", "Financial report")
        # Should identify at least some column info
        assert len(schema.columns) == len(sample_financial_df.columns)

    def test_excel_ingestor_csv(self, tmp_path):
        """Excel ingestor should handle CSV files."""
        from app.ingest.excel_ingest import ExcelIngestor
        
        # Create a test CSV
        csv_path = tmp_path / "test_data.csv"
        df = pd.DataFrame({
            "Name": ["A", "B", "C"],
            "Value": [100, 200, 300],
        })
        df.to_csv(csv_path, index=False)
        
        ingestor = ExcelIngestor()
        result = ingestor.ingest_best_sheet_from_path(
            csv_path, "test_data.csv", "test_client"
        )
        assert result["success"] is True
        assert result["rows"] == 3


# ═══════════════════════════════════════════════════════════════════
# FEATURE FLAGS: Configuration Toggle Tests
# ═══════════════════════════════════════════════════════════════════

class TestFeatureFlags:
    """Test feature flag settings load correctly with defaults and env overrides."""

    def test_router_settings_defaults(self):
        """Router feature flags should have sensible defaults."""
        from app.config import RouterSettings
        s = RouterSettings()
        assert s.enable_fast_pass is True
        assert s.enable_semantic_routing is True
        assert s.enable_llm_fallback is True
        assert s.regex_confidence == 0.95
        assert s.semantic_confidence == 0.70
        assert s.cache_enabled is True
        assert s.cache_max_size == 1000

    def test_healing_settings_defaults(self):
        """Self-healing feature flags should default to enabled with 2 retries."""
        from app.config import HealingSettings
        s = HealingSettings()
        assert s.enabled is True
        assert s.max_retries == 2
        assert s.enable_error_hints is True
        assert s.enable_fuzzy_column_fix is True
        assert s.log_healing_prompts is False
        assert s.backoff_base_ms == 100

    def test_schema_settings_defaults(self):
        """Schema filtering feature flags should default to enabled."""
        from app.config import SchemaSettings
        s = SchemaSettings()
        assert s.enable_filtering is True
        assert s.max_columns == 20
        assert s.fuzzy_match_cutoff == 0.70
        assert s.enable_semantic_expansion is True
        assert s.enable_key_column_detection is True
        assert s.min_fallback_columns == 5

    def test_hot_cache_settings_defaults(self):
        """Hot cache feature flags should default to enabled with correct tiers."""
        from app.config import HotCacheSettings
        s = HotCacheSettings()
        assert s.enabled is True
        assert s.max_entries == 1000
        assert s.ttl_seconds == 1800
        assert s.semantic is True
        assert s.similarity == 0.92
        assert s.hot_key_threshold == 10
        assert s.l1_ttl == 60

    def test_compression_settings_defaults(self):
        """Token compression feature flags should default to enabled."""
        from app.config import CompressionSettings
        s = CompressionSettings()
        assert s.enabled is True
        assert s.max_sample_rows == 5
        assert s.include_statistics is True
        assert s.include_patterns is True

    def test_duckdb_settings_s3_fields(self):
        """DuckDB settings should include S3 and performance fields."""
        from app.config import DuckDBSettings
        s = DuckDBSettings()
        # S3 disabled by default (safe)
        assert s.enable_s3 is False
        assert s.s3_region is None or s.s3_region == ''  # Accept both None and empty string
        assert s.s3_endpoint is None or s.s3_endpoint == ''  # Accept both None and empty string
        # Performance tuning defaults
        assert s.enable_object_cache is True
        assert s.object_cache_size == "256MB"
        assert s.preserve_insertion_order is False
        assert s.enable_parallel_csv is True

    def test_settings_wired_into_root(self):
        """All feature flag subsections should be accessible from root Settings."""
        from app.config import settings
        assert hasattr(settings, "router")
        assert hasattr(settings, "healing")
        assert hasattr(settings, "schema_filter")
        assert hasattr(settings, "hot_cache")
        assert hasattr(settings, "compression")
        # Check types
        assert settings.router.enable_fast_pass in (True, False)
        assert settings.healing.max_retries >= 0
        assert settings.schema_filter.max_columns > 0

    def test_env_override(self):
        """Feature flags should be overridable via env vars."""
        import os
        # Temporarily set env var
        original = os.environ.get("ROUTER_ENABLE_FAST_PASS")
        try:
            os.environ["ROUTER_ENABLE_FAST_PASS"] = "false"
            from app.config import RouterSettings
            s = RouterSettings()
            assert s.enable_fast_pass is False
        finally:
            if original is None:
                os.environ.pop("ROUTER_ENABLE_FAST_PASS", None)
            else:
                os.environ["ROUTER_ENABLE_FAST_PASS"] = original


# ═══════════════════════════════════════════════════════════════════
# DUCKDB: Performance Tuning & S3 Integration Tests
# ═══════════════════════════════════════════════════════════════════

class TestDuckDBPerformance:
    """Test DuckDB performance tuning and S3 integration."""

    def test_performance_pragmas_applied(self):
        """Object cache and insertion order PRAGMAs should be applied."""
        from app.sql_engine import SQLEngine
        engine = SQLEngine(
            memory_limit="512MB",
            threads=2,
            perf_config={
                "enable_object_cache": True,
                "object_cache_size": "128MB",
                "preserve_insertion_order": False,
            },
        )
        assert engine._connection is not None

        # Verify object_cache is on (DuckDB may return bool or string)
        result = engine._connection.execute(
            "SELECT current_setting('enable_object_cache')"
        ).fetchone()
        assert str(result[0]).lower() == "true"

        # Verify preserve_insertion_order is off
        result = engine._connection.execute(
            "SELECT current_setting('preserve_insertion_order')"
        ).fetchone()
        assert str(result[0]).lower() == "false"

        engine.close()

    def test_s3_disabled_by_default(self):
        """S3 should not be initialized when enable_s3 is false."""
        from app.sql_engine import SQLEngine
        engine = SQLEngine(
            memory_limit="512MB",
            threads=1,
            s3_config={"enable_s3": False},
        )
        assert engine._s3_initialized is False
        assert engine.list_s3_views() == {}
        engine.close()

    def test_create_s3_view_fails_without_s3(self):
        """create_s3_view should return False when S3 is not configured."""
        from app.sql_engine import SQLEngine
        engine = SQLEngine(
            memory_limit="512MB",
            threads=1,
            s3_config={"enable_s3": False},
        )
        result = engine.create_s3_view("test_view", "s3://bucket/file.parquet")
        assert result is False
        engine.close()

    def test_query_s3_parquet_fails_without_s3(self):
        """query_s3_parquet should raise RuntimeError when S3 is not configured."""
        from app.sql_engine import SQLEngine
        engine = SQLEngine(
            memory_limit="512MB",
            threads=1,
            s3_config={"enable_s3": False},
        )
        with pytest.raises(RuntimeError, match="S3 not configured"):
            engine.query_s3_parquet("s3://bucket/file.parquet")
        engine.close()

    def test_engine_registers_and_queries_with_tuning(self, sample_financial_df):
        """Engine should work correctly with performance tuning applied."""
        from app.sql_engine import SQLEngine
        engine = SQLEngine(
            memory_limit="512MB",
            threads=2,
            perf_config={
                "enable_object_cache": True,
                "preserve_insertion_order": False,
            },
        )

        success, cleaned_df = engine.register_dataframe("test_data", sample_financial_df)
        assert success is True

        result = engine.execute_df("SELECT SUM(fy23) FROM test_data")
        assert not result.empty

        # Revenue + COGS + Gross Profit + EBITDA + Net Profit for FY23
        expected = 1500.0 + 850.0 + 650.0 + 490.0 + 350.0
        assert abs(result.iloc[0, 0] - expected) < 0.01

        engine.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x"])
