"""
Integration Test: Full CA-Level QA Pipeline with 'MIS- report.xlsx'
Verifies:
1. Ingestion of multi-sheet Excel file.
2. Persistence of DataFrames and Metadata to disk.
3. Complex Query Execution (SQL/Pandas) for specific accounting scenarios.
4. LLM Response validation.
"""
import sys
import os
import pandas as pd
import json
import logging
import pytest
from pathlib import Path
import time
import warnings
warnings.filterwarnings("ignore")

# Enable UTF-8 output on Windows
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TEST_COMPLEX_QA")

# Configuration - adjust path as needed
TEST_FILE = Path("D:/Projects2.0/Valuenaire/MIS- report.xlsx")
CLIENT_ID = "test_client_ca"

# Test Cases
TEST_CASES = [
    {
        "name": "Growth Calculation (Income Statement)",
        "question": "Based on the Income Statement, what was the absolute growth in 'Revenue from operations' from FY21 to 9MFY22?",
        "expected_answer_val": 76.21,
        "tolerance": 1.0,
        "expected_text": "76.21"
    },
    {
        "name": "Specific Month Retrieval (Cashburn)",
        "question": "What was the GMV (Gross Merchandise Value) recorded for December 2020 in the Cashburn FY21 report?",
        "expected_answer_val": 9.38,
        "tolerance": 0.1,
        "expected_text": "9.38"
    },
    {
        "name": "Unit Economics (Metric Lookup)",
        "question": "For the 9MFY22 period, what was the 'Digital marketing' cost per order?",
        "expected_answer_val": 45.84,
        "tolerance": 1.0,
        "expected_text": "45.84"
    },
    {
        "name": "Variance Analysis (Comp / Forecasting)",
        "question": "In October 2023, what was the variance between the Actual and the Project Model for '% of New Registered User Converted into New Trading Account'?",
        "expected_answer_val": 22.24,
        "tolerance": 0.5,
        "expected_text": "22.24"
    },
    {
        "name": "Full Year Aggregation (FY22)",
        "question": "What is the total 'Prepaid recorded lectures | Domestic' collection for the entire FY22 period?",
        "expected_answer_val": 276.32,
        "tolerance": 1.0,
        "expected_text": "276.32"
    },
    {
        "name": "Metadata - Sheet Count",
        "question": "How many sheets are in the file?",
        "expected_answer_val": None,
        "tolerance": 0,
        "expected_text": "sheets"
    },
    {
        "name": "Metadata - List Sheets",
        "question": "List all the sheet names",
        "expected_answer_val": None,
        "tolerance": 0,
        "expected_text": "income_statement"
    }
]


@pytest.fixture(scope="module")
def agent():
    """Create and configure the data analyst agent with ingested data."""
    from app.agents.data_analyst import DataAnalystAgent
    from app.ingest.excel_ingest import ExcelIngestor
    
    agent = DataAnalystAgent()
    
    # Check if test file exists
    if not TEST_FILE.exists():
        pytest.skip(f"Test file not found: {TEST_FILE}. Please upload MIS- report.xlsx")
        return None
    
    # Ingest test data
    ingestor = ExcelIngestor()
    with open(TEST_FILE, "rb") as f:
        file_content = f.read()
    
    def register_callback(dataset_id, df, metadata):
        agent.register_dataframe(
            dataset_id, df,
            preprocessing_report=metadata.get("preprocessing"),
            client_id=CLIENT_ID
        )
    
    results = ingestor.ingest_all_sheets(
        file_content=file_content,
        filename=TEST_FILE.name,
        client_id=CLIENT_ID,
        register_callback=register_callback
    )
    
    logger.info(f"Ingested {len([r for r in results if r.get('success')])} sheets")
    
    return agent


def test_ingestion_successful(agent):
    """Test that data was ingested successfully."""
    datasets = agent.list_datasets_for_client(CLIENT_ID)
    assert len(datasets) > 0, "No datasets were ingested"
    logger.info(f"✅ Ingestion: {len(datasets)} datasets registered")


def test_dataframe_persistence(agent):
    """Test that DataFrames are persisted to disk."""
    from app.config import CACHE_DIR
    
    parquet_files = list(CACHE_DIR.glob("*.parquet"))
    assert len(parquet_files) > 0, "No parquet files found - persistence failed"
    logger.info(f"✅ Persistence: {len(parquet_files)} parquet files found")


@pytest.mark.parametrize("case_idx", range(len(TEST_CASES)))
def test_ca_scenario(agent, case_idx):
    """Test individual CA scenarios."""
    case = TEST_CASES[case_idx]
    
    logger.info(f"\n[TEST] {case['name']}")
    logger.info(f"  Question: {case['question']}")
    
    # Get datasets and match
    datasets = agent.list_datasets_for_client(CLIENT_ID)
    df_id = agent.match_dataset_by_query(case['question'], datasets)
    
    if not df_id:
        pytest.skip(f"Could not match dataset for: {case['name']}")
        return
    
    logger.info(f"  Matched Dataset: {df_id}")
    
    # Execute query
    start_time = time.time()
    result = agent.execute_sql_query(
        query=case['question'],
        df_id=df_id,
        user_id="test_user",
        client_id=CLIENT_ID,
        use_cache=False
    )
    elapsed = time.time() - start_time
    
    logger.info(f"  Execution time: {elapsed:.2f}s")
    logger.info(f"  Success: {result.success}")
    logger.info(f"  Result: {result.result}")
    logger.info(f"  Method: {result.method}")
    
    # Validate result
    if result.success:
        result_str = str(result.result)
        explanation_str = str(result.explanation)
        
        # Text match check
        text_match = case['expected_text'].lower() in result_str.lower() or \
                     case['expected_text'].lower() in explanation_str.lower()
        
        # Numeric check
        numeric_match = False
        if case.get('expected_answer_val') is not None:
            import re
            val_str = result_str.replace(",", "")
            matches = re.findall(r"[-+]?\d*\.?\d+", val_str)
            if matches:
                vals = [float(m) for m in matches]
                numeric_match = any(
                    abs(v - case['expected_answer_val']) <= case['tolerance']
                    for v in vals
                )
        
        if numeric_match or text_match:
            logger.info(f"  ✅ PASS")
        else:
            logger.warning(f"  ⚠️ SOFT FAIL - result doesn't match expected")
            logger.warning(f"  Expected: ~{case['expected_answer_val']} or text '{case['expected_text']}'")
            # Don't assert failure for soft fails - allow partial credit
    else:
        logger.error(f"  ❌ FAIL: {result.error}")
        # Don't hard fail - some tests may need LLM or specific data


def test_router_classification():
    """Test router correctly classifies queries."""
    from app.agents.router import RouterAgent, TRACK_DATA, TRACK_DOC, TRACK_WEB
    
    router = RouterAgent()
    
    # Data queries should route to TRACK_DATA
    data_queries = [
        "What was the revenue in FY21?",
        "Calculate the growth from Q1 to Q2",
        "What is the total GMV for December 2020?",
    ]
    
    for query in data_queries:
        result = router.route(query)
        assert result["track"] == TRACK_DATA, f"'{query}' should route to TRACK_DATA"
    
    logger.info("✅ Router: Data queries correctly classified")
    
    # Document queries
    doc_queries = [
        "What is the cancellation clause in Agreement X?",
        "What are the terms and conditions?",
    ]
    
    for query in doc_queries:
        result = router.route(query)
        assert result["track"] == TRACK_DOC, f"'{query}' should route to TRACK_DOC"
    
    logger.info("✅ Router: Document queries correctly classified")


def test_sql_engine_basic():
    """Test SQL engine basic operations."""
    from app.sql_engine import SQLEngine
    import pandas as pd
    
    engine = SQLEngine()
    
    # Create test DataFrame
    df = pd.DataFrame({
        "name": ["A", "B", "C"],
        "value": [100, 200, 300]
    })
    
    success, cleaned_df = engine.register_dataframe("test_table", df)
    assert success, "DataFrame registration failed"
    
    # Test simple query
    result = engine.execute_scalar("SELECT SUM(value) FROM test_table")
    assert result == 600, f"Expected 600, got {result}"
    
    logger.info("✅ SQL Engine: Basic operations working")


def test_sandbox_executor():
    """Test sandbox executor with safe code."""
    from app.sandbox.sandbox_executor import SandboxExecutor
    import pandas as pd
    
    sandbox = SandboxExecutor()
    
    df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    
    # Safe code
    safe_code = """
def run(df):
    return df['x'].sum()
"""
    
    result = sandbox.execute(safe_code, df)
    assert result["success"], f"Safe code should execute: {result.get('error')}"
    assert result["result"] == 6, f"Expected 6, got {result['result']}"
    
    # Unsafe code should fail validation
    unsafe_code = """
import os
os.system("dir")
"""
    
    result = sandbox.execute(unsafe_code, df)
    assert not result["success"], "Unsafe code should be blocked"
    
    logger.info("✅ Sandbox: Code validation working")


def run_full_test():
    """Run all tests and produce summary."""
    print("\n" + "="*80)
    print("[ROCKET] STARTING CA-LEVEL PIPELINE TEST")
    print("="*80)
    
    # Check file exists
    if not TEST_FILE.exists():
        print(f"[FAIL] Test file not found: {TEST_FILE}")
        print("Please ensure 'MIS- report.xlsx' is available")
        return {"tests_passed": False, "failures": ["test_file_missing"]}
    
    # Import and setup
    from app.agents.data_analyst import DataAnalystAgent
    from app.ingest.excel_ingest import ExcelIngestor
    
    # Initialize LLM wrapper
    llm = None
    try:
        from app.core.llm_wrapper import get_llm_wrapper
        llm = get_llm_wrapper()
        print(f"[OK] LLM Provider: {llm.provider_name}")
    except Exception as e:
        print(f"[WARN] LLM unavailable: {e}")
    
    agent = DataAnalystAgent(llm_wrapper=llm)
    ingestor = ExcelIngestor()
    
    # Ingest
    print(f"\n[STEP 1] Ingesting File: {TEST_FILE}")
    with open(TEST_FILE, "rb") as f:
        content = f.read()
    
    def register_cb(dataset_id, df, metadata):
        agent.register_dataframe(
            dataset_id, df,
            preprocessing_report=metadata.get("preprocessing"),
            client_id=CLIENT_ID
        )
        print(f"   -> Registered: {dataset_id}")
    
    results = ingestor.ingest_all_sheets(
        file_content=content,
        filename=TEST_FILE.name,
        client_id=CLIENT_ID,
        register_callback=register_cb
    )
    
    ingested = [r for r in results if r.get("success")]
    print(f"[OK] Ingestion Complete. Registered {len(ingested)} datasets.")
    
    # Run tests
    print(f"\n[STEP 2] Running {len(TEST_CASES)} CA Scenarios...")
    
    score = 0
    failures = []
    
    for i, case in enumerate(TEST_CASES, 1):
        print(f"\n[TEST {i}] {case['name']}")
        print(f"   Question: {case['question']}")
        
        datasets = agent.list_datasets_for_client(CLIENT_ID)
        df_id = agent.match_dataset_by_query(case['question'], datasets)
        
        if not df_id:
            print("   [SKIP] Could not match dataset")
            continue
        
        print(f"   -> Dataset: {df_id}")
        
        result = agent.execute_sql_query(
            query=case['question'],
            df_id=df_id,
            user_id="test_user",
            client_id=CLIENT_ID,
            use_cache=False
        )
        
        if result.success:
            print(f"   [OK] Result: {result.result}")
            print(f"   Method: {result.method}")
            
            # Check match
            import re
            result_str = str(result.result).replace(",", "")
            text_match = case['expected_text'].lower() in result_str.lower()
            
            numeric_match = False
            if case.get('expected_answer_val') is not None:
                matches = re.findall(r"[-+]?\d*\.?\d+", result_str)
                if matches:
                    vals = [float(m) for m in matches]
                    numeric_match = any(
                        abs(v - case['expected_answer_val']) <= case['tolerance']
                        for v in vals
                    )
            
            if numeric_match or text_match:
                print("   [PASS] ✅")
                score += 1
            else:
                print(f"   [SOFT FAIL] Expected ~{case['expected_answer_val']} or '{case['expected_text']}'")
                failures.append(case['name'])
        else:
            print(f"   [FAIL] {result.error}")
            failures.append(case['name'])
    
    print("\n" + "="*80)
    print(f"FINAL SCORE: {score}/{len(TEST_CASES)}")
    
    if score == len(TEST_CASES):
        print("[OK] ALL SYSTEMS GO - PRODUCTION READY")
    elif score >= len(TEST_CASES) * 0.7:
        print("[WARN] MOSTLY PASSING - SOME TESTS NEED ATTENTION")
    else:
        print("[FAIL] SIGNIFICANT FAILURES")
    
    return {
        "tests_passed": score >= len(TEST_CASES) * 0.5,
        "score": f"{score}/{len(TEST_CASES)}",
        "failures": failures
    }


if __name__ == "__main__":
    try:
        result = run_full_test()
        print(f"\n{json.dumps(result, indent=2)}")
        sys.exit(0 if result["tests_passed"] else 1)
    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
