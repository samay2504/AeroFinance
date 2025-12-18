"""
LangChain Tools for CA Agent.
Lightweight, stateless tools decorated with @tool.

Tools:
- web_search: Multi-provider web search (DuckDuckGo, Brave, Bing)
- benford_test: Fraud detection using Benford's Law
- cohort_analysis: Customer retention cohort analysis
- three_way_match: Invoice/PO/Ledger reconciliation
- orchestrator: LLM-integrated tool execution
"""
from typing import Any, Dict, List, Optional

try:
    from langchain.tools import tool
except ImportError:
    # Fallback decorator if langchain not installed
    def tool(name: str = None, return_direct: bool = False):
        def decorator(func):
            func.name = name or func.__name__
            func.return_direct = return_direct
            return func
        return decorator

# Import tools
from .benford import benford_test, run_benford_test
from .three_way_match import three_way_match, run_three_way_match
from .cohort import cohort_analysis, run_cohort_analysis
from .web_search import web_search, search_tax_rate, search_regulation, search_market_benchmark

# Import orchestrator
from .orchestrator import (
    ToolOrchestrator,
    get_tool_orchestrator,
    search_and_respond,
    check_tax_rate,
    detect_fraud,
)

# Export all tools
__all__ = [
    # Core tools
    "benford_test",
    "three_way_match",
    "cohort_analysis",
    "web_search",
    # Standalone functions
    "run_benford_test",
    "run_three_way_match",
    "run_cohort_analysis",
    # Search helpers
    "search_tax_rate",
    "search_regulation",
    "search_market_benchmark",
    # Orchestrator
    "ToolOrchestrator",
    "get_tool_orchestrator",
    "search_and_respond",
    "check_tax_rate",
    "detect_fraud",
]

