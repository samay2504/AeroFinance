"""
LangChain Tools for CA Agent.
Lightweight, stateless tools decorated with @tool.
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

from .benford import benford_test
from .three_way_match import three_way_match
from .cohort import cohort_analysis
from .web_search import web_search

# Export all tools
__all__ = ["benford_test", "three_way_match", "cohort_analysis", "web_search"]
