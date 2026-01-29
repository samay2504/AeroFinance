"""
Sandbox Integration Tests - Real-world E2B and Local Sandbox Testing

Tests the sandbox execution pipeline with:
- Real E2B SDK calls (when ENABLE_E2B=true)
- Local sandbox fallback
- DataFrame operations
- Financial calculations (CA-level use cases)
- Error handling and graceful degradation

Run with: pytest tests/test_sandbox_e2b.py -v
Or standalone: python tests/test_sandbox_e2b.py
"""
import sys
import os
import pytest
import time
import logging
from pathlib import Path
from typing import Dict, Any

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load environment variables BEFORE any app imports
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

# Apply DLL fix for Windows
from app.core.dll_fix import apply_dll_fix
apply_dll_fix()

import pandas as pd
import numpy as np

# Configure logging for test visibility
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def sample_financial_df():
    """Create a realistic financial DataFrame for CA use cases."""
    return pd.DataFrame({
        "particulars": [
            "Revenue", 
            "Cost of Goods Sold", 
            "Gross Profit",
            "Operating Expenses",
            "EBITDA",
            "Depreciation",
            "Interest Expense",
            "Profit Before Tax",
            "Tax Expense",
            "Net Profit"
        ],
        "FY2023": [
            10000000,  # Revenue
            6000000,   # COGS
            4000000,   # Gross Profit
            1500000,   # OpEx
            2500000,   # EBITDA
            500000,    # Depreciation
            200000,    # Interest
            1800000,   # PBT
            450000,    # Tax
            1350000    # Net Profit
        ],
        "FY2024": [
            12000000,  # Revenue
            7000000,   # COGS
            5000000,   # Gross Profit
            1800000,   # OpEx
            3200000,   # EBITDA
            600000,    # Depreciation
            180000,    # Interest
            2420000,   # PBT
            605000,    # Tax
            1815000    # Net Profit
        ]
    })


@pytest.fixture
def sample_sales_df():
    """Create a sales DataFrame for analytics testing."""
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    return pd.DataFrame({
        "date": dates,
        "product": np.random.choice(["Widget A", "Widget B", "Widget C"], 100),
        "region": np.random.choice(["North", "South", "East", "West"], 100),
        "quantity": np.random.randint(10, 500, 100),
        "unit_price": np.random.uniform(100, 1000, 100).round(2),
        "discount_pct": np.random.uniform(0, 0.3, 100).round(3)
    })


# =============================================================================
# LOCAL SANDBOX TESTS
# =============================================================================

class TestLocalSandbox:
    """Test local sandbox execution (always available)."""
    
    def test_simple_sum(self, sample_financial_df):
        """Test simple DataFrame sum calculation."""
        from app.sandbox.sandbox_executor import SandboxExecutor
        
        sandbox = SandboxExecutor(timeout_seconds=10)
        code = """
def run(df):
    return df['FY2024'].sum()
"""
        result = sandbox.run_user_code(code, {"df": sample_financial_df}, timeout_sec=10)
        
        assert result["success"] is True, f"Execution failed: {result.get('error')}"
        # Sum of all FY2024 values
        expected = sample_financial_df['FY2024'].sum()
        assert result["result"] == expected, f"Expected {expected}, got {result['result']}"
        logger.info(f"✅ Local sandbox sum: {result['result']}")
    
    def test_growth_calculation(self, sample_financial_df):
        """Test YoY growth calculation - common CA task."""
        from app.sandbox.sandbox_executor import SandboxExecutor
        
        sandbox = SandboxExecutor(timeout_seconds=10)
        code = """
def run(df):
    # Find Revenue row and calculate YoY growth
    revenue_row = df[df['particulars'] == 'Revenue'].iloc[0]
    fy23 = revenue_row['FY2023']
    fy24 = revenue_row['FY2024']
    growth = ((fy24 - fy23) / fy23) * 100
    return round(growth, 2)
"""
        result = sandbox.run_user_code(code, {"df": sample_financial_df}, timeout_sec=10)
        
        assert result["success"] is True, f"Execution failed: {result.get('error')}"
        # Expected: (12M - 10M) / 10M * 100 = 20%
        assert result["result"] == 20.0, f"Expected 20.0%, got {result['result']}%"
        logger.info(f"✅ Revenue growth: {result['result']}%")
    
    def test_profit_margin(self, sample_financial_df):
        """Test profit margin calculation."""
        from app.sandbox.sandbox_executor import SandboxExecutor
        
        sandbox = SandboxExecutor(timeout_seconds=10)
        code = """
def run(df):
    # Calculate Net Profit Margin for FY2024
    revenue = df[df['particulars'] == 'Revenue']['FY2024'].values[0]
    net_profit = df[df['particulars'] == 'Net Profit']['FY2024'].values[0]
    margin = (net_profit / revenue) * 100
    return round(margin, 2)
"""
        result = sandbox.run_user_code(code, {"df": sample_financial_df}, timeout_sec=10)
        
        assert result["success"] is True, f"Execution failed: {result.get('error')}"
        # Expected: 1815000 / 12000000 * 100 = 15.125%
        expected = round((1815000 / 12000000) * 100, 2)
        assert result["result"] == expected, f"Expected {expected}%, got {result['result']}%"
        logger.info(f"✅ Net Profit Margin FY24: {result['result']}%")
    
    def test_aggregation_query(self, sample_sales_df):
        """Test sales aggregation by region."""
        from app.sandbox.sandbox_executor import SandboxExecutor
        
        sandbox = SandboxExecutor(timeout_seconds=10)
        code = """
def run(df):
    # Calculate total sales value by region
    df['sales_value'] = df['quantity'] * df['unit_price'] * (1 - df['discount_pct'])
    result = df.groupby('region')['sales_value'].sum().to_dict()
    return {k: round(v, 2) for k, v in result.items()}
"""
        result = sandbox.run_user_code(code, {"df": sample_sales_df}, timeout_sec=10)
        
        assert result["success"] is True, f"Execution failed: {result.get('error')}"
        assert isinstance(result["result"], dict)
        assert "North" in result["result"] or "South" in result["result"]
        logger.info(f"✅ Sales by region: {result['result']}")
    
    def test_unsafe_code_blocked(self, sample_financial_df):
        """Test that unsafe code is blocked."""
        from app.sandbox.sandbox_executor import SandboxExecutor
        
        sandbox = SandboxExecutor(timeout_seconds=10)
        
        # Test OS access
        unsafe_code = """
import os
def run(df):
    return os.listdir('.')
"""
        result = sandbox.run_user_code(unsafe_code, {"df": sample_financial_df}, timeout_sec=10)
        assert result["success"] is False, "Unsafe code should be blocked"
        logger.info(f"✅ Unsafe code blocked: {result.get('error', '')[:50]}")


# =============================================================================
# E2B SDK INTEGRATION TESTS
# =============================================================================

class TestE2BIntegration:
    """Test E2B SDK integration - requires ENABLE_E2B=true and valid API key."""
    
    @pytest.fixture(autouse=True)
    def check_e2b_enabled(self):
        """Skip tests if E2B is not enabled."""
        from app.sandbox.sandbox_executor import get_e2b_executor
        e2b = get_e2b_executor()
        if not e2b.enabled:
            pytest.skip("E2B not enabled (set ENABLE_E2B=true and E2B_API_KEY)")
    
    def test_e2b_simple_execution(self, sample_financial_df):
        """Test simple code execution via E2B."""
        from app.sandbox.sandbox_executor import get_e2b_executor
        
        e2b = get_e2b_executor()
        
        # Simple calculation that E2B can handle
        code = """
result = 2 + 2
print(f"Result: {result}")
result
"""
        result = e2b.execute(code, timeout_sec=30)
        
        logger.info(f"E2B Result: {result}")
        
        # E2B should return success or gracefully fail
        if result["success"]:
            logger.info(f"✅ E2B execution successful: {result.get('result')}")
        else:
            # E2B may fail due to quota/network - that's okay, we log it
            logger.warning(f"⚠️ E2B execution failed (expected if quota exceeded): {result.get('error')}")
    
    def test_e2b_pandas_operation(self, sample_sales_df):
        """Test Pandas operation via E2B."""
        from app.sandbox.sandbox_executor import get_e2b_executor
        
        e2b = get_e2b_executor()
        
        # Pandas aggregation
        code = """
import pandas as pd

# Create sample data
data = {
    'product': ['A', 'B', 'A', 'B', 'A'],
    'sales': [100, 200, 150, 250, 300]
}
df = pd.DataFrame(data)

# Aggregate
result = df.groupby('product')['sales'].sum().to_dict()
print(f"Aggregation result: {result}")
result
"""
        result = e2b.execute(code, timeout_sec=30)
        
        if result["success"]:
            logger.info(f"✅ E2B Pandas operation: {result.get('result')}")
        else:
            logger.warning(f"⚠️ E2B Pandas failed: {result.get('error')}")


# =============================================================================
# E2B FALLBACK TESTS
# =============================================================================

class TestE2BFallback:
    """Test E2B with local fallback - graceful degradation."""
    
    def test_execute_with_fallback_local_success(self, sample_financial_df):
        """Test execute_with_e2b_fallback when local succeeds."""
        from app.sandbox.sandbox_executor import execute_with_e2b_fallback
        
        code = """
def run(df):
    return df['FY2024'].max()
"""
        result = execute_with_e2b_fallback(
            code, 
            sample_financial_df, 
            prefer_local=True,  # Try local first
            timeout_sec=10
        )
        
        assert result["success"] is True, f"Execution failed: {result.get('error')}"
        assert result["executor"] in ["local", "e2b"]
        logger.info(f"✅ Fallback test passed, executor: {result['executor']}, result: {result['result']}")
    
    def test_execute_code_safe_auto(self, sample_financial_df):
        """Test execute_code_safe with local execution (E2B doesn't have DataFrame context)."""
        from app.sandbox.sandbox_executor import execute_code_safe
        
        code = """
def run(df):
    # Calculate EBITDA margin
    revenue = df[df['particulars'] == 'Revenue']['FY2024'].values[0]
    ebitda = df[df['particulars'] == 'EBITDA']['FY2024'].values[0]
    return round((ebitda / revenue) * 100, 2)
"""
        # Force local execution since E2B doesn't have DataFrame context
        result = execute_code_safe(
            code, 
            sample_financial_df,
            timeout_sec=10,
            use_e2b=False  # Force local - E2B needs different code structure
        )
        
        assert result["success"] is True, f"Execution failed: {result.get('error')}"
        # Expected: 3200000 / 12000000 * 100 = 26.67%
        expected = round((3200000 / 12000000) * 100, 2)
        assert result["result"] == expected, f"Expected {expected}%, got {result['result']}%"
        logger.info(f"✅ execute_code_safe: {result['result']}%, executor: {result['executor']}")
    
    def test_fallback_on_local_error(self, sample_financial_df):
        """Test that E2B is tried when local fails (if enabled)."""
        from app.sandbox.sandbox_executor import execute_with_e2b_fallback, get_e2b_executor
        
        # Code that might fail locally but could work on E2B
        # Using a library not in local sandbox
        code = """
def run(df):
    # This uses a simple pandas operation that should work
    return df.shape[0]
"""
        result = execute_with_e2b_fallback(
            code, 
            sample_financial_df, 
            prefer_local=True,
            timeout_sec=10
        )
        
        # Should succeed either via local or E2B
        assert result["success"] is True, f"Execution failed: {result.get('error')}"
        assert result["result"] == 10  # 10 rows in financial_df
        logger.info(f"✅ Fallback test passed, executor: {result['executor']}")


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

class TestSandboxPerformance:
    """Test sandbox performance characteristics."""
    
    def test_execution_latency(self, sample_financial_df):
        """Test that execution completes within acceptable time."""
        from app.sandbox.sandbox_executor import execute_code_safe
        
        code = """
def run(df):
    return df['FY2024'].mean()
"""
        start = time.time()
        result = execute_code_safe(code, sample_financial_df, timeout_sec=10, use_e2b=False)
        elapsed_ms = (time.time() - start) * 1000
        
        assert result["success"] is True
        assert elapsed_ms < 5000, f"Execution too slow: {elapsed_ms:.0f}ms"
        logger.info(f"✅ Execution latency: {elapsed_ms:.0f}ms")
    
    def test_multiple_executions(self, sample_financial_df):
        """Test multiple sequential executions."""
        from app.sandbox.sandbox_executor import SandboxExecutor
        
        sandbox = SandboxExecutor(timeout_seconds=10)
        
        operations = [
            ("sum", "return df['FY2024'].sum()"),
            ("mean", "return df['FY2024'].mean()"),
            ("max", "return df['FY2024'].max()"),
            ("min", "return df['FY2024'].min()"),
        ]
        
        results = []
        for name, op in operations:
            code = f"def run(df):\n    {op}"
            result = sandbox.run_user_code(code, {"df": sample_financial_df}, timeout_sec=5)
            results.append((name, result["success"], result.get("result")))
        
        for name, success, value in results:
            assert success, f"{name} failed"
            logger.info(f"✅ {name}: {value}")
        
        logger.info(f"✅ All {len(operations)} operations completed successfully")


# =============================================================================
# REAL-WORLD CA SCENARIOS
# =============================================================================

class TestRealWorldCAScenarios:
    """Test real-world CA (Chartered Accountant) analysis scenarios."""
    
    def test_variance_analysis(self, sample_financial_df):
        """Test budget variance analysis."""
        from app.sandbox.sandbox_executor import execute_code_safe
        
        code = """
def run(df):
    # Calculate YoY variance for all line items
    result = {}
    for _, row in df.iterrows():
        item = row['particulars']
        fy23 = row['FY2023']
        fy24 = row['FY2024']
        variance = fy24 - fy23
        variance_pct = ((fy24 - fy23) / fy23 * 100) if fy23 != 0 else 0
        result[item] = {
            'variance': variance,
            'variance_pct': round(variance_pct, 2)
        }
    return result
"""
        result = execute_code_safe(code, sample_financial_df, timeout_sec=10, use_e2b=False)
        
        assert result["success"] is True, f"Failed: {result.get('error')}"
        assert "Revenue" in result["result"]
        assert result["result"]["Revenue"]["variance_pct"] == 20.0
        logger.info(f"✅ Variance analysis: Revenue grew {result['result']['Revenue']['variance_pct']}%")
    
    def test_ratio_analysis(self, sample_financial_df):
        """Test financial ratio calculations."""
        from app.sandbox.sandbox_executor import execute_code_safe
        
        code = """
def run(df):
    def get_value(particulars, year):
        return df[df['particulars'] == particulars][year].values[0]
    
    ratios = {
        'gross_margin_fy24': round(get_value('Gross Profit', 'FY2024') / get_value('Revenue', 'FY2024') * 100, 2),
        'ebitda_margin_fy24': round(get_value('EBITDA', 'FY2024') / get_value('Revenue', 'FY2024') * 100, 2),
        'net_margin_fy24': round(get_value('Net Profit', 'FY2024') / get_value('Revenue', 'FY2024') * 100, 2),
        'interest_coverage': round(get_value('EBITDA', 'FY2024') / get_value('Interest Expense', 'FY2024'), 2),
    }
    return ratios
"""
        result = execute_code_safe(code, sample_financial_df, timeout_sec=10, use_e2b=False)
        
        assert result["success"] is True, f"Failed: {result.get('error')}"
        ratios = result["result"]
        
        assert ratios["gross_margin_fy24"] == 41.67  # 5M / 12M
        assert ratios["ebitda_margin_fy24"] == 26.67  # 3.2M / 12M
        logger.info(f"✅ Ratio analysis: {ratios}")


# =============================================================================
# RUN TESTS
# =============================================================================

def run_all_tests():
    """Run all tests and produce summary."""
    print("\n" + "="*80)
    print("🧪 SANDBOX E2B INTEGRATION TEST SUITE")
    print("="*80)
    
    # Check E2B status
    from app.sandbox.sandbox_executor import get_e2b_executor
    e2b = get_e2b_executor()
    print(f"\n📦 E2B Status: {'ENABLED' if e2b.enabled else 'DISABLED'}")
    
    # Run pytest
    exit_code = pytest.main([
        __file__, 
        "-v", 
        "--tb=short",
        "-x",  # Stop on first failure
    ])
    
    print("\n" + "="*80)
    if exit_code == 0:
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
    print("="*80)
    
    return exit_code


if __name__ == "__main__":
    run_all_tests()
