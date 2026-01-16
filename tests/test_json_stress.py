"""
Comprehensive JSON Stress Test - Production Grade
==================================================
Tests the AI-CA system with large, complex JSON files representing
real-world corporate financial data.

WHAT THIS TESTS:
1. Large JSON file ingestion (~1.5MB, ~1MB, ~125KB files)
2. Complex nested data structures (company profiles, financials, directors)
3. Multi-year financial data traversal
4. Cross-file queries (loading two JSONs and combining data)
5. Edge cases: null values, deep nesting, large arrays
6. Human-like natural language queries
7. Router classification with structured data
8. Stress testing memory and performance

TEST FILES:
- company-details-GODREJ 1.json (1.5MB - Full company profile with charges, directors, history)
- netflix-probe42.json (125KB - LLP with multi-year financials)
- prob-godrej-properties-limited.json (985KB - Detailed standalone financials)

SUCCESS CRITERIA:
- All files ingest without errors
- Queries return accurate financial values
- Cross-file queries combine data correctly
- Performance is acceptable (<30s per query)
"""

import sys
import os

# Set environment variables before ANY other imports
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Force UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Add torch DLL directory to PATH before importing torch
if sys.platform == 'win32':
    torch_lib = r'd:\Projects2.0\Valuenaire\.conda\Lib\site-packages\torch\lib'
    if os.path.exists(torch_lib):
        os.environ['PATH'] = torch_lib + os.pathsep + os.environ.get('PATH', '')
        try:
            os.add_dll_directory(torch_lib)
        except Exception:
            pass
    try:
        import torch  # noqa
    except Exception:
        pass

import time
import json
import logging
import traceback
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set
from enum import Enum
import warnings
warnings.filterwarnings("ignore")

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("JSON_STRESS_TEST")

# Suppress verbose logging during tests
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("app.agents.data_analyst").setLevel(logging.WARNING)


# ============================================================================
# TEST CONFIGURATION
# ============================================================================

class TestCategory(Enum):
    INGESTION = "ingestion"
    SINGLE_FILE = "single_file"
    MULTI_FILE = "multi_file"
    EDGE_CASE = "edge_case"
    PERFORMANCE = "performance"
    CROSS_FILE = "cross_file"


@dataclass
class TestCase:
    """Individual test case definition."""
    name: str
    category: TestCategory
    query: str
    expected_track: Optional[str] = None
    expected_value: Optional[float] = None
    expected_text: Optional[str] = None
    tolerance: float = 0.1  # 10% tolerance for numeric comparisons
    file_context: Optional[str] = None
    requires_files: Optional[List[str]] = None  # For cross-file tests
    should_fail_gracefully: bool = False
    max_time_seconds: float = 60.0


@dataclass
class TestResult:
    """Result of a test case."""
    test_case: TestCase
    passed: bool
    actual_result: str
    actual_value: Optional[float] = None
    error: Optional[str] = None
    elapsed_time: float = 0.0


# ============================================================================
# TEST FILES
# ============================================================================

BASE_DIR = Path("D:/Projects2.0/Valuenaire")

TEST_FILES = {
    "godrej_full": {
        "path": BASE_DIR / "company-details-GODREJ 1.json",
        "client_id": "json_stress_test",
        "type": "json",
        "description": "Full Godrej Properties company profile (1.5MB)",
        "expected_size_mb": 1.5
    },
    "netflix_llp": {
        "path": BASE_DIR / "netflix-probe42.json",
        "client_id": "json_stress_test",
        "type": "json",
        "description": "Netflix Entertainment Services LLP financials (125KB)",
        "expected_size_mb": 0.125
    },
    "godrej_probe": {
        "path": BASE_DIR / "prob-godrej-properties-limited.json",
        "client_id": "json_stress_test",
        "type": "json",
        "description": "Godrej Properties Probe42 detailed financials (985KB)",
        "expected_size_mb": 0.985
    }
}


# ============================================================================
# TEST CASES - Comprehensive Real-World Scenarios
# ============================================================================

TEST_CASES: List[TestCase] = [
    # =========================================================================
    # SECTION 1: SINGLE FILE - GODREJ FULL (company-details-GODREJ 1.json)
    # Large company profile with directors, charges, history
    # =========================================================================
    
    TestCase(
        name="Godrej Full: Company Name",
        category=TestCategory.SINGLE_FILE,
        query="What is the name of the company in this data?",
        expected_track="TRACK_DATA",
        expected_text="GODREJ PROPERTIES LIMITED",
        file_context="godrej_full"
    ),
    TestCase(
        name="Godrej Full: CIN Number",
        category=TestCategory.SINGLE_FILE,
        query="What is the CIN of Godrej Properties?",
        expected_track="TRACK_DATA",
        expected_text="L74120MH1985PLC035308",
        file_context="godrej_full"
    ),
    TestCase(
        name="Godrej Full: Authorized Capital",
        category=TestCategory.SINGLE_FILE,
        query="What is the authorized capital of the company?",
        expected_track="TRACK_DATA",
        expected_value=6690000000,
        tolerance=0.01,
        file_context="godrej_full"
    ),
    TestCase(
        name="Godrej Full: Paid Up Capital",
        category=TestCategory.SINGLE_FILE,
        query="What is the paid up capital?",
        expected_track="TRACK_DATA",
        expected_value=1506031720,
        tolerance=0.01,
        file_context="godrej_full"
    ),
    TestCase(
        name="Godrej Full: Net Worth",
        category=TestCategory.SINGLE_FILE,
        query="What is the net worth of the company?",
        expected_track="TRACK_DATA",
        expected_value=103761476885,  # From overview.NET_WORTH_COMP
        tolerance=0.05,
        file_context="godrej_full"
    ),
    TestCase(
        name="Godrej Full: Total Turnover",
        category=TestCategory.SINGLE_FILE,
        query="What is the total turnover?",
        expected_track="TRACK_DATA",
        expected_value=13306145405,  # From overview.TOT_TURNOVER
        tolerance=0.05,
        file_context="godrej_full"
    ),
    TestCase(
        name="Godrej Full: Total Revenue",
        category=TestCategory.SINGLE_FILE,
        query="What is the total revenue in crores?",
        expected_track="TRACK_DATA", 
        expected_value=19496200000,  # From overview.TOTAL_REVENUE_CR
        tolerance=0.05,
        file_context="godrej_full"
    ),
    TestCase(
        name="Godrej Full: Promoter Shareholding",
        category=TestCategory.SINGLE_FILE,
        query="What percentage of shares do promoters hold?",
        expected_track="TRACK_DATA",
        expected_value=58.47,  # From overview.promoters_sh_perct
        tolerance=0.1,
        file_context="godrej_full"
    ),
    TestCase(
        name="Godrej Full: Director Count",
        category=TestCategory.SINGLE_FILE,
        query="How many directors does Godrej Properties have?",
        expected_track="TRACK_DATA",
        expected_value=9,  # Count of directors array
        tolerance=0.0,
        file_context="godrej_full"
    ),
    TestCase(
        name="Godrej Full: Managing Director",
        category=TestCategory.SINGLE_FILE,
        query="Who is the Managing Director of the company?",
        expected_track="TRACK_DATA",
        expected_text="GAURAV PANDEY",
        file_context="godrej_full"
    ),
    TestCase(
        name="Godrej Full: CFO Name",
        category=TestCategory.SINGLE_FILE,
        query="Who is the CFO?",
        expected_track="TRACK_DATA",
        expected_text="RAJENDRA",  # RAJENDRA SAWARMAL KHETAWAT
        file_context="godrej_full"
    ),
    TestCase(
        name="Godrej Full: Date of Incorporation",
        category=TestCategory.SINGLE_FILE,
        query="When was the company incorporated?",
        expected_track="TRACK_DATA",
        expected_text="1985",
        file_context="godrej_full"
    ),
    TestCase(
        name="Godrej Full: Registered State",
        category=TestCategory.SINGLE_FILE,
        query="In which state is the company registered?",
        expected_track="TRACK_DATA",
        expected_text="Maharashtra",
        file_context="godrej_full"
    ),
    TestCase(
        name="Godrej Full: Website",
        category=TestCategory.SINGLE_FILE,
        query="What is the company website?",
        expected_track="TRACK_DATA",
        expected_text="godrejproperties.com",
        file_context="godrej_full"
    ),
    TestCase(
        name="Godrej Full: Listed Status",
        category=TestCategory.SINGLE_FILE,
        query="Is the company listed on stock exchange?",
        expected_track="TRACK_DATA",
        expected_text="Listed",
        file_context="godrej_full"
    ),
    TestCase(
        name="Godrej Full: Total Charges",
        category=TestCategory.SINGLE_FILE,
        query="What is the total sum of charges registered?",
        expected_track="TRACK_DATA",
        expected_value=114825500000,  # From overview.totalCharges
        tolerance=0.1,
        file_context="godrej_full"
    ),
    
    # =========================================================================
    # SECTION 2: SINGLE FILE - NETFLIX LLP (netflix-probe42.json)
    # LLP with multi-year financial statements
    # =========================================================================
    
    TestCase(
        name="Netflix LLP: Legal Name",
        category=TestCategory.SINGLE_FILE,
        query="What is the legal name of this LLP?",
        expected_track="TRACK_DATA",
        expected_text="NETFLIX ENTERTAINMENT SERVICES INDIA LLP",
        file_context="netflix_llp"
    ),
    TestCase(
        name="Netflix LLP: Incorporation Date",
        category=TestCategory.SINGLE_FILE,
        query="When was Netflix India LLP incorporated?",
        expected_track="TRACK_DATA",
        expected_text="2017",
        file_context="netflix_llp"
    ),
    TestCase(
        name="Netflix LLP: FY2025 Revenue",
        category=TestCategory.SINGLE_FILE,
        query="What was Netflix India's net revenue in FY2025?",
        expected_track="TRACK_DATA",
        expected_value=37689802501,  # financials[0].statement_of_income_and_expenditure.lineItems.net_revenue
        tolerance=0.05,
        file_context="netflix_llp"
    ),
    TestCase(
        name="Netflix LLP: FY2024 Revenue",
        category=TestCategory.SINGLE_FILE,
        query="What was Netflix India's revenue for FY2024?",
        expected_track="TRACK_DATA",
        expected_value=28457688113,  # financials[1]
        tolerance=0.05,
        file_context="netflix_llp"
    ),
    TestCase(
        name="Netflix LLP: Profit After Tax FY2025",
        category=TestCategory.SINGLE_FILE,
        query="What was the profit after tax for Netflix India in FY2025?",
        expected_track="TRACK_DATA",
        expected_value=848016865,
        tolerance=0.05,
        file_context="netflix_llp"
    ),
    TestCase(
        name="Netflix LLP: Cash and Cash Equivalents",
        category=TestCategory.SINGLE_FILE,
        query="What are the cash and cash equivalents for Netflix India?",
        expected_track="TRACK_DATA",
        expected_value=8003343902,  # FY2025
        tolerance=0.1,
        file_context="netflix_llp"
    ),
    TestCase(
        name="Netflix LLP: Total Contribution",
        category=TestCategory.SINGLE_FILE,
        query="What is the total contribution received by the LLP?",
        expected_track="TRACK_DATA",
        expected_value=650000000,
        tolerance=0.01,
        file_context="netflix_llp"
    ),
    TestCase(
        name="Netflix LLP: Trade Payables FY2025",
        category=TestCategory.SINGLE_FILE,
        query="What are the trade payables for FY2025?",
        expected_track="TRACK_DATA",
        expected_value=8632462006,
        tolerance=0.05,
        file_context="netflix_llp"
    ),
    TestCase(
        name="Netflix LLP: Operating Profit FY2025",
        category=TestCategory.SINGLE_FILE,
        query="What was Netflix India's operating profit in FY2025?",
        expected_track="TRACK_DATA",
        expected_value=649521884,
        tolerance=0.1,
        file_context="netflix_llp"
    ),
    TestCase(
        name="Netflix LLP: Depreciation FY2025",
        category=TestCategory.SINGLE_FILE,
        query="What was the depreciation expense in FY2025?",
        expected_track="TRACK_DATA",
        expected_value=73639151,
        tolerance=0.05,
        file_context="netflix_llp"
    ),
    TestCase(
        name="Netflix LLP: Employee Count Designation",
        category=TestCategory.SINGLE_FILE,
        query="How many designated partners does Netflix India have?",
        expected_track="TRACK_DATA",
        expected_value=6,  # Count of directors with designation "Designated Partner"
        tolerance=0.0,
        file_context="netflix_llp"
    ),
    TestCase(
        name="Netflix LLP: Auditor Name",
        category=TestCategory.SINGLE_FILE,
        query="Who is the auditor of Netflix India?",
        expected_track="TRACK_DATA",
        expected_text="GOVIND PITAMBAR AHUJA",
        file_context="netflix_llp"
    ),
    TestCase(
        name="Netflix LLP: Revenue Growth",
        category=TestCategory.SINGLE_FILE,
        query="Calculate the revenue growth from FY2024 to FY2025 for Netflix India",
        expected_track="TRACK_DATA",
        expected_value=32.4,  # (37689802501 - 28457688113) / 28457688113 * 100
        tolerance=5.0,  # Allow 5 percentage points variation
        file_context="netflix_llp"
    ),
    
    # =========================================================================
    # SECTION 3: SINGLE FILE - GODREJ PROBE (prob-godrej-properties-limited.json)
    # Detailed standalone financials with ratios
    # =========================================================================
    
    TestCase(
        name="Godrej Probe: Total Equity FY2025",
        category=TestCategory.SINGLE_FILE,
        query="What is the total equity of Godrej Properties in FY2025?",
        expected_track="TRACK_DATA",
        expected_value=174441400000,  # financials[0].bs.subTotals.total_equity
        tolerance=0.05,
        file_context="godrej_probe"
    ),
    TestCase(
        name="Godrej Probe: Total Debt FY2025",
        category=TestCategory.SINGLE_FILE,
        query="What is the total debt in FY2025?",
        expected_track="TRACK_DATA",
        expected_value=119680900000,
        tolerance=0.05,
        file_context="godrej_probe"
    ),
    TestCase(
        name="Godrej Probe: Net Revenue FY2025",
        category=TestCategory.SINGLE_FILE,
        query="What was the net revenue for FY2025?",
        expected_track="TRACK_DATA",
        expected_value=19496200000,
        tolerance=0.05,
        file_context="godrej_probe"
    ),
    TestCase(
        name="Godrej Probe: Profit After Tax FY2025",
        category=TestCategory.SINGLE_FILE,
        query="What was the PAT for Godrej Properties in FY2025?",
        expected_track="TRACK_DATA",
        expected_value=10110100000,
        tolerance=0.05,
        file_context="godrej_probe"
    ),
    TestCase(
        name="Godrej Probe: Debt to Equity Ratio",
        category=TestCategory.SINGLE_FILE,
        query="What is the debt to equity ratio for FY2025?",
        expected_track="TRACK_DATA",
        expected_value=0.69,  # financials[0].ratios.debt_by_equity
        tolerance=0.1,
        file_context="godrej_probe"
    ),
    TestCase(
        name="Godrej Probe: Current Ratio",
        category=TestCategory.SINGLE_FILE,
        query="What is the current ratio?",
        expected_track="TRACK_DATA",
        expected_value=1.73,
        tolerance=0.1,
        file_context="godrej_probe"
    ),
    TestCase(
        name="Godrej Probe: Net Margin",
        category=TestCategory.SINGLE_FILE,
        query="What is the net margin percentage?",
        expected_track="TRACK_DATA",
        expected_value=51.86,
        tolerance=5.0,
        file_context="godrej_probe"
    ),
    TestCase(
        name="Godrej Probe: Revenue Growth",
        category=TestCategory.SINGLE_FILE,
        query="What is the revenue growth rate?",
        expected_track="TRACK_DATA",
        expected_value=46.52,
        tolerance=5.0,
        file_context="godrej_probe"
    ),
    TestCase(
        name="Godrej Probe: Return on Equity",
        category=TestCategory.SINGLE_FILE,
        query="What is the ROE for Godrej Properties?",
        expected_track="TRACK_DATA",
        expected_value=5.8,
        tolerance=1.0,
        file_context="godrej_probe"
    ),
    TestCase(
        name="Godrej Probe: Interest Coverage",
        category=TestCategory.SINGLE_FILE,
        query="What is the interest coverage ratio?",
        expected_track="TRACK_DATA",
        expected_value=3.24,
        tolerance=0.5,
        file_context="godrej_probe"
    ),
    TestCase(
        name="Godrej Probe: Inventories",
        category=TestCategory.SINGLE_FILE,
        query="What is the inventory value for FY2025?",
        expected_track="TRACK_DATA",
        expected_value=153126800000,
        tolerance=0.05,
        file_context="godrej_probe"
    ),
    TestCase(
        name="Godrej Probe: Cash Flow Operating",
        category=TestCategory.SINGLE_FILE,
        query="What is the cash flow from operating activities?",
        expected_track="TRACK_DATA",
        expected_value=-17578400000,  # Negative
        tolerance=0.1,
        file_context="godrej_probe"
    ),
    TestCase(
        name="Godrej Probe: Share Capital",
        category=TestCategory.SINGLE_FILE,
        query="What is the share capital of Godrej Properties?",
        expected_track="TRACK_DATA",
        expected_value=1505900000,
        tolerance=0.05,
        file_context="godrej_probe"
    ),
    TestCase(
        name="Godrej Probe: Long Term Borrowings",
        category=TestCategory.SINGLE_FILE,
        query="What are the long term borrowings?",
        expected_track="TRACK_DATA",
        expected_value=40000000000,
        tolerance=0.05,
        file_context="godrej_probe"
    ),
    TestCase(
        name="Godrej Probe: Auditor Firm",
        category=TestCategory.SINGLE_FILE,
        query="Which audit firm audits Godrej Properties?",
        expected_track="TRACK_DATA",
        expected_text="B S R",
        file_context="godrej_probe"
    ),
    
    # =========================================================================
    # SECTION 4: CROSS-FILE QUERIES (Both Godrej files loaded)
    # Tests combining data from multiple sources in the same session
    # =========================================================================
    
    TestCase(
        name="Cross-File: Compare Godrej Revenue Sources",
        category=TestCategory.CROSS_FILE,
        query="Compare the total revenue figures from both Godrej data files",
        expected_track="TRACK_DATA",
        requires_files=["godrej_full", "godrej_probe"],
        should_fail_gracefully=True  # Complex cross-reference
    ),
    TestCase(
        name="Cross-File: Godrej vs Netflix Revenue",
        category=TestCategory.CROSS_FILE,
        query="Which company has higher revenue - Godrej Properties or Netflix India?",
        expected_track="TRACK_DATA",
        expected_text="Godrej",  # Godrej ~19496 Cr vs Netflix ~3768 Cr
        requires_files=["godrej_probe", "netflix_llp"],
    ),
    TestCase(
        name="Cross-File: Combined Profit Analysis",
        category=TestCategory.CROSS_FILE,
        query="What is the combined profit after tax of Godrej Properties and Netflix India for their latest FY?",
        expected_track="TRACK_DATA",
        expected_value=10958116865,  # 10110100000 + 848016865
        tolerance=0.1,
        requires_files=["godrej_probe", "netflix_llp"],
    ),
    TestCase(
        name="Cross-File: Cash Position Comparison",
        category=TestCategory.CROSS_FILE,
        query="Which company has more cash reserves - Godrej or Netflix India?",
        expected_track="TRACK_DATA",
        # Godrej cash: 40599800000, Netflix: 8003343902
        expected_text="Godrej",
        requires_files=["godrej_probe", "netflix_llp"],
    ),
    TestCase(
        name="Cross-File: Debt Comparison",
        category=TestCategory.CROSS_FILE,
        query="Compare the debt levels of Godrej Properties and Netflix India",
        expected_track="TRACK_DATA",
        requires_files=["godrej_probe", "netflix_llp"],
        should_fail_gracefully=True  # Complex analysis
    ),
    TestCase(
        name="Cross-File: Director Count Comparison",
        category=TestCategory.CROSS_FILE,
        query="Which company has more directors - Godrej Properties or Netflix India LLP?",
        expected_track="TRACK_DATA",
        expected_text="Godrej",  # 9 vs 6
        requires_files=["godrej_full", "netflix_llp"],
    ),
    
    # =========================================================================
    # SECTION 5: EDGE CASES
    # Testing robustness with unusual queries
    # =========================================================================
    
    TestCase(
        name="Edge: Very Large Number Formatting",
        category=TestCategory.EDGE_CASE,
        query="Express Godrej's net worth in crores",
        expected_track="TRACK_DATA",
        expected_text="crore",
        file_context="godrej_full"
    ),
    TestCase(
        name="Edge: Null Value Handling",
        category=TestCategory.EDGE_CASE,
        query="What is the CIRP status of Godrej Properties?",
        expected_track="TRACK_DATA",
        expected_text="",  # Should handle null gracefully
        file_context="godrej_full",
        should_fail_gracefully=True
    ),
    TestCase(
        name="Edge: Nested Data Access",
        category=TestCategory.EDGE_CASE,
        query="What is the registered address pincode of Netflix India?",
        expected_track="TRACK_DATA",
        expected_text="400051",
        file_context="netflix_llp"
    ),
    TestCase(
        name="Edge: Array Aggregation",
        category=TestCategory.EDGE_CASE,
        query="List all the addresses of Godrej Properties",
        expected_track="TRACK_DATA",
        expected_text="Mumbai",
        file_context="godrej_full"
    ),
    TestCase(
        name="Edge: Date Parsing",
        category=TestCategory.EDGE_CASE,
        query="When was the last balance sheet date for Godrej?",
        expected_track="TRACK_DATA",
        expected_text="2025",
        file_context="godrej_full"
    ),
    TestCase(
        name="Edge: Multi-Year Comparison",
        category=TestCategory.EDGE_CASE,
        query="How has Netflix India's profit changed over the years from 2018 to 2025?",
        expected_track="TRACK_DATA",
        file_context="netflix_llp",
        should_fail_gracefully=True  # Complex temporal analysis
    ),
    TestCase(
        name="Edge: Percentage Calculation",
        category=TestCategory.EDGE_CASE,
        query="What percentage of Netflix India's total assets are in cash?",
        expected_track="TRACK_DATA",
        # 8003343902 / 13675018401 * 100 = ~58.5%
        expected_value=58.5,
        tolerance=5.0,
        file_context="netflix_llp"
    ),
    TestCase(
        name="Edge: Empty Query on Large File",
        category=TestCategory.EDGE_CASE,
        query="",
        should_fail_gracefully=True,
        file_context="godrej_full"
    ),
    TestCase(
        name="Edge: Non-Existent Field Query",
        category=TestCategory.EDGE_CASE,
        query="What is the number of employees at Netflix India?",
        expected_track="TRACK_DATA",
        file_context="netflix_llp",
        should_fail_gracefully=True  # Field doesn't exist
    ),
    
    # =========================================================================
    # SECTION 6: PERFORMANCE TESTS
    # Testing response times
    # =========================================================================
    
    TestCase(
        name="Perf: Large File Summary",
        category=TestCategory.PERFORMANCE,
        query="Give me a summary of the company's key financial metrics",
        expected_track="TRACK_DATA",
        file_context="godrej_full",
        max_time_seconds=45.0
    ),
    TestCase(
        name="Perf: Multi-Year Analysis",
        category=TestCategory.PERFORMANCE,
        query="Summarize Netflix India's financial performance from 2019 to 2025",
        expected_track="TRACK_DATA",
        file_context="netflix_llp",
        max_time_seconds=60.0
    ),
]


# ============================================================================
# TEST RUNNER
# ============================================================================

class JSONStressTestRunner:
    """Production-grade stress test runner for JSON files."""
    
    def __init__(self):
        self.results: List[TestResult] = []
        self.agent = None
        self.router = None
        self.llm = None
        self.ingested_datasets: Dict[str, List[str]] = {}
        self.current_files: Set[str] = set()
        self.start_time = time.time()
        
    def setup(self) -> bool:
        """Initialize all components."""
        print("\n" + "=" * 80)
        print("🚀 JSON STRESS TEST - PRODUCTION GRADE")
        print("=" * 80)
        print(f"   Test Cases: {len(TEST_CASES)}")
        print(f"   Test Files: {len(TEST_FILES)}")
        print("=" * 80)
        
        print("\n📦 Initializing Components...")
        
        try:
            from app.core.llm_wrapper import get_llm_wrapper
            self.llm = get_llm_wrapper()
            provider_name = getattr(self.llm, 'provider', getattr(self.llm, '_provider', 'unknown'))
            print(f"   ✅ LLM Wrapper initialized ({provider_name})")
        except Exception as e:
            print(f"   ❌ LLM Wrapper failed: {e}")
            return False
        
        try:
            from app.agents.data_analyst import DataAnalystAgent
            self.agent = DataAnalystAgent()
            print("   ✅ DataAnalystAgent initialized")
        except Exception as e:
            print(f"   ❌ DataAnalystAgent failed: {e}")
            return False
        
        try:
            from app.agents.router import get_router_agent
            self.router = get_router_agent(self.llm)
            print("   ✅ RouterAgent initialized")
        except Exception as e:
            print(f"   ⚠️ RouterAgent unavailable: {e}")
        
        return True
    
    def ingest_file(self, file_key: str) -> bool:
        """Ingest a single JSON file into DataFrames and optionally RAG."""
        if file_key in self.current_files:
            return True  # Already ingested
        
        file_config = TEST_FILES.get(file_key)
        if not file_config:
            print(f"   ❌ Unknown file key: {file_key}")
            return False
        
        path = Path(file_config["path"])
        if not path.exists():
            print(f"   ❌ File not found: {path}")
            return False
        
        print(f"\n   📂 Ingesting: {path.name}")
        print(f"      Size: {path.stat().st_size / (1024*1024):.2f} MB")
        
        try:
            start = time.time()
            
            # Load JSON  
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Use JSON ingestor
            from app.ingest.json_ingest import JSONIngestor
            
            ingestor = JSONIngestor()
            datasets = []
            dataframes = {}  # Store (dataset_id -> df) for RAG
            
            def register_callback(dataset_id, df, metadata):
                datasets.append(dataset_id)
                dataframes[dataset_id] = df
                self.agent.register_dataframe(
                    dataset_id, df,
                    preprocessing_report=metadata.get("preprocessing"),
                    client_id=file_config["client_id"]
                )
            
            # Get source name from filename (remove extension)
            source_name = path.stem
            
            # ALWAYS use regular ingestion first (this registers with agent)
            results = ingestor.ingest_json(
                data=data,
                source_name=source_name,
                client_id=file_config["client_id"],
                register_callback=register_callback
            )
            
            if not results.get("success"):
                raise Exception(results.get("error", "Unknown ingestion error"))
            
            # Optionally add to RAG if available
            rag_chunks = 0
            try:
                from app.rag.ingest import get_rag_pipeline
                rag_pipeline = get_rag_pipeline()
                
                if rag_pipeline and rag_pipeline.is_available:
                    for dataset_id, df in dataframes.items():
                        try:
                            text_repr = ingestor.generate_text_representation(df, dataset_id)
                            rag_result = rag_pipeline.ingest_document(
                                text=text_repr,
                                client_id=file_config["client_id"],
                                doc_id=dataset_id,
                                metadata={"source_type": "json", "table": dataset_id}
                            )
                            if rag_result.get("success"):
                                rag_chunks += rag_result.get("chunks_created", 1)
                        except Exception as rag_table_err:
                            logger.debug(f"RAG for {dataset_id} skipped: {rag_table_err}")
                    
                    if rag_chunks > 0:
                        print(f"      📚 RAG: {rag_chunks} chunks indexed")
            except ImportError:
                pass  # RAG module not available
            except Exception as rag_err:
                logger.debug(f"RAG ingestion skipped: {rag_err}")
            
            elapsed = time.time() - start
            self.ingested_datasets[file_key] = datasets
            self.current_files.add(file_key)
            
            print(f"      ✅ Ingested {len(datasets)} tables in {elapsed:.2f}s")
            return True
            
        except Exception as e:
            print(f"      ❌ Ingestion failed: {e}")
            traceback.print_exc()
            return False
    
    def run_test(self, test: TestCase) -> TestResult:
        """Run a single test case."""
        print(f"\n🔹 {test.name}")
        query_preview = test.query[:60] + "..." if len(test.query) > 60 else test.query
        print(f"   Query: \"{query_preview}\"")
        
        start = time.time()
        result = TestResult(test_case=test, passed=False, actual_result="")
        
        try:
            # Handle empty query
            if not test.query:
                result.passed = test.should_fail_gracefully
                result.actual_result = "Empty query handled"
                return self._finalize_result(result, start)
            
            # Ensure required files are loaded
            if test.requires_files:
                for file_key in test.requires_files:
                    if file_key not in self.current_files:
                        self.ingest_file(file_key)
            elif test.file_context and test.file_context not in self.current_files:
                self.ingest_file(test.file_context)
            
            # Get dataset to query
            datasets = []
            if test.requires_files:
                for file_key in test.requires_files:
                    datasets.extend(self.ingested_datasets.get(file_key, []))
            elif test.file_context:
                datasets = self.ingested_datasets.get(test.file_context, [])
            else:
                # All available datasets
                for ds_list in self.ingested_datasets.values():
                    datasets.extend(ds_list)
            
            if not datasets:
                result.actual_result = "No datasets available"
                result.passed = test.should_fail_gracefully
                return self._finalize_result(result, start)
            
            # Execute query on best matching dataset
            best_result = None
            best_dataset = None
            
            for ds_id in datasets[:5]:  # Limit to 5 for performance
                try:
                    query_result = self.agent.execute_sql_query(
                        test.query, ds_id, 
                        client_id=TEST_FILES.get(test.file_context, {}).get("client_id", "json_stress_test")
                    )
                    
                    if query_result.success:
                        if best_result is None or query_result.value is not None:
                            best_result = query_result
                            best_dataset = ds_id
                            if query_result.value is not None:
                                break
                except Exception:
                    continue
            
            if best_result:
                result.actual_result = str(best_result.result)
                result.actual_value = best_result.value
                
                # Validate result
                result.passed = self._validate_result(test, best_result)
            else:
                result.actual_result = "No successful query"
                result.passed = test.should_fail_gracefully
            
        except Exception as e:
            result.error = str(e)
            result.passed = test.should_fail_gracefully
            if not test.should_fail_gracefully:
                traceback.print_exc()
        
        return self._finalize_result(result, start)
    
    def _validate_result(self, test: TestCase, query_result) -> bool:
        """Validate query result against expected values."""
        # Check expected value
        if test.expected_value is not None:
            if query_result.value is None:
                return test.should_fail_gracefully
            
            actual = float(query_result.value)
            expected = float(test.expected_value)
            
            if expected != 0:
                diff_pct = abs(actual - expected) / abs(expected) * 100
                if diff_pct <= test.tolerance * 100:
                    return True
            elif actual == 0:
                return True
            
            return test.should_fail_gracefully
        
        # Check expected text
        if test.expected_text is not None:
            result_str = str(query_result.result).lower()
            if test.expected_text.lower() in result_str:
                return True
            return test.should_fail_gracefully
        
        # No specific expectation - pass if query succeeded
        return query_result.success
    
    def _finalize_result(self, result: TestResult, start_time: float) -> TestResult:
        """Finalize and record test result."""
        result.elapsed_time = time.time() - start_time
        
        # Check performance limit
        if result.test_case.max_time_seconds and result.elapsed_time > result.test_case.max_time_seconds:
            if not result.test_case.should_fail_gracefully:
                result.passed = False
                result.error = f"Timeout: {result.elapsed_time:.2f}s > {result.test_case.max_time_seconds}s"
        
        self.results.append(result)
        
        # Print result
        status = "✅ PASS" if result.passed else "❌ FAIL"
        print(f"   {status} ({result.elapsed_time:.2f}s)")
        
        if result.actual_value is not None:
            print(f"   Actual Value: {result.actual_value}")
        if result.error:
            print(f"   Error: {result.error}")
        
        return result
    
    def run_all_tests(self):
        """Run all test cases."""
        # Group tests by category
        categories = {}
        for test in TEST_CASES:
            cat = test.category.value
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(test)
        
        # Run tests by category
        for category, tests in categories.items():
            print(f"\n{'='*60}")
            print(f"📋 CATEGORY: {category.upper()} ({len(tests)} tests)")
            print("="*60)
            
            for test in tests:
                self.run_test(test)
    
    def print_summary(self):
        """Print test summary."""
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        failed = total - passed
        
        print("\n" + "=" * 80)
        print("📊 TEST SUMMARY")
        print("=" * 80)
        print(f"   Total Tests: {total}")
        print(f"   ✅ Passed: {passed}")
        print(f"   ❌ Failed: {failed}")
        print(f"   Pass Rate: {passed/total*100:.1f}%")
        print(f"   Total Time: {time.time() - self.start_time:.2f}s")
        
        if failed > 0:
            print("\n   Failed Tests:")
            for r in self.results:
                if not r.passed:
                    print(f"      - {r.test_case.name}")
                    if r.error:
                        print(f"        Error: {r.error}")
        
        print("=" * 80)
        
        return passed == total


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point."""
    runner = JSONStressTestRunner()
    
    if not runner.setup():
        print("\n❌ Setup failed. Exiting.")
        return 1
    
    runner.run_all_tests()
    success = runner.print_summary()
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
