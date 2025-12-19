"""
Tool Orchestrator - Production-grade LLM-integrated tool execution.
Provides seamless integration between LLM reasoning and tool execution.
"""
import logging
import re
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ToolCategory(Enum):
    """Categories of available tools."""
    SEARCH = "search"
    ANALYSIS = "analysis"
    RECONCILIATION = "reconciliation"
    FRAUD_DETECTION = "fraud_detection"


@dataclass
class ToolResult:
    """Standardized tool result."""
    success: bool
    tool_name: str
    result: Dict[str, Any]
    explanation: str = ""
    confidence: float = 1.0
    source: str = ""


@dataclass
class ToolDefinition:
    """Tool metadata and callable."""
    name: str
    description: str
    category: ToolCategory
    func: Callable
    requires_data: bool = False
    parameters: Dict[str, str] = field(default_factory=dict)


class ToolOrchestrator:
    """
    Production-grade tool orchestration for LLM integration.
    
    Features:
    - Automatic tool selection based on query intent
    - LLM-integrated responses (harmonious output)
    - Error handling and fallback
    - Result caching
    - Audit logging
    """
    
    def __init__(self, llm_wrapper=None):
        self._llm = llm_wrapper
        self._tools: Dict[str, ToolDefinition] = {}
        self._cache: Dict[str, ToolResult] = {}
        self._execution_history: List[Dict] = []
        
        # Register all available tools
        self._register_tools()
    
    def _register_tools(self):
        """Register all available tools with their metadata."""
        try:
            # Web Search Tools
            from app.tools.web_search import (
                web_search, search_tax_rate, search_regulation, search_market_benchmark
            )
            
            # Get underlying functions for LangChain tools
            web_search_func = web_search.func if hasattr(web_search, 'func') else web_search
            
            self._tools["web_search"] = ToolDefinition(
                name="web_search",
                description="Search the web for real-time information, facts, and regulations",
                category=ToolCategory.SEARCH,
                func=web_search_func,
                parameters={"query": "Search query", "num_results": "Number of results (1-5)"}
            )
            
            self._tools["search_tax_rate"] = ToolDefinition(
                name="search_tax_rate",
                description="Search for tax rates from official sources",
                category=ToolCategory.SEARCH,
                func=search_tax_rate,
                parameters={"tax_type": "corporate/income/gst/tds", "country": "Country", "year": "Year"}
            )
            
            self._tools["search_regulation"] = ToolDefinition(
                name="search_regulation",
                description="Search for regulatory information and compliance requirements",
                category=ToolCategory.SEARCH,
                func=search_regulation,
                parameters={"regulation_name": "Name of regulation", "country": "Country"}
            )
            
            self._tools["search_benchmark"] = ToolDefinition(
                name="search_benchmark",
                description="Search for industry benchmarks and market data",
                category=ToolCategory.SEARCH,
                func=search_market_benchmark,
                parameters={"metric": "Metric to search", "industry": "Industry sector"}
            )
            
        except ImportError as e:
            logger.warning(f"Web search tools not available: {e}")
        
        try:
            # Fraud Detection - Benford's Law
            from app.tools.benford import benford_test, run_benford_test
            
            benford_func = benford_test.func if hasattr(benford_test, 'func') else benford_test
            
            self._tools["benford_test"] = ToolDefinition(
                name="benford_test",
                description="Detect anomalies in numeric data using Benford's Law (fraud detection)",
                category=ToolCategory.FRAUD_DETECTION,
                func=benford_func,
                requires_data=True,
                parameters={"dataset_id": "Dataset ID", "column_name": "Numeric column to test"}
            )
            
            # Also register standalone version
            self._tools["run_benford_test"] = ToolDefinition(
                name="run_benford_test",
                description="Run Benford test on a list of values directly",
                category=ToolCategory.FRAUD_DETECTION,
                func=run_benford_test,
                parameters={"values": "List of numeric values", "threshold": "Deviation threshold"}
            )
            
        except ImportError as e:
            logger.warning(f"Benford tool not available: {e}")
        
        try:
            # Cohort Analysis
            from app.tools.cohort import cohort_analysis, run_cohort_analysis
            
            cohort_func = cohort_analysis.func if hasattr(cohort_analysis, 'func') else cohort_analysis
            
            self._tools["cohort_analysis"] = ToolDefinition(
                name="cohort_analysis",
                description="Build cohort retention analysis from transaction data",
                category=ToolCategory.ANALYSIS,
                func=cohort_func,
                requires_data=True,
                parameters={
                    "dataset_id": "Dataset ID",
                    "date_column": "Date column name",
                    "user_column": "User/customer ID column"
                }
            )
            
        except ImportError as e:
            logger.warning(f"Cohort tool not available: {e}")
        
        try:
            # Three-Way Match (Reconciliation)
            from app.tools.three_way_match import three_way_match, run_three_way_match
            
            match_func = three_way_match.func if hasattr(three_way_match, 'func') else three_way_match
            
            self._tools["three_way_match"] = ToolDefinition(
                name="three_way_match",
                description="Perform 3-way match between Invoice, PO, and Ledger for reconciliation",
                category=ToolCategory.RECONCILIATION,
                func=match_func,
                requires_data=True,
                parameters={
                    "invoice_dataset_id": "Invoice dataset",
                    "po_dataset_id": "PO dataset",
                    "ledger_dataset_id": "Ledger dataset"
                }
            )
            
        except ImportError as e:
            logger.warning(f"Three-way match tool not available: {e}")
        
        logger.info(f"Registered {len(self._tools)} tools: {list(self._tools.keys())}")
    
    def list_tools(self, category: Optional[ToolCategory] = None) -> List[Dict[str, str]]:
        """List available tools with descriptions."""
        tools = []
        for name, tool_def in self._tools.items():
            if category and tool_def.category != category:
                continue
            tools.append({
                "name": name,
                "description": tool_def.description,
                "category": tool_def.category.value,
                "parameters": tool_def.parameters
            })
        return tools
    
    def detect_tool_intent(self, query: str) -> Optional[str]:
        """
        Detect which tool should be used based on query intent.
        Uses pattern matching and keyword detection.
        """
        query_lower = query.lower()
        
        # Search intent patterns
        search_patterns = [
            (r"(search|find|look up|what is|current|latest)", "web_search"),
            (r"(tax rate|gst rate|tds rate|income tax)", "search_tax_rate"),
            (r"(regulation|compliance|rule|law|requirement)", "search_regulation"),
            (r"(benchmark|industry average|market rate|standard)", "search_benchmark"),
        ]
        
        # Analysis intent patterns
        analysis_patterns = [
            (r"(benford|fraud|anomaly|first digit|suspicious)", "benford_test"),
            (r"(cohort|retention|customer lifetime|churn)", "cohort_analysis"),
            (r"(reconcil|3-way|three-way|match|invoice.*po|po.*invoice)", "three_way_match"),
        ]
        
        all_patterns = search_patterns + analysis_patterns
        
        for pattern, tool_name in all_patterns:
            if re.search(pattern, query_lower):
                if tool_name in self._tools:
                    return tool_name
        
        return None
    
    def execute_tool(
        self,
        tool_name: str,
        **kwargs
    ) -> ToolResult:
        """
        Execute a tool by name with given parameters.
        
        Args:
            tool_name: Name of the tool to execute
            **kwargs: Tool parameters
            
        Returns:
            ToolResult with success status and output
        """
        if tool_name not in self._tools:
            return ToolResult(
                success=False,
                tool_name=tool_name,
                result={"error": f"Unknown tool: {tool_name}"},
                explanation=f"Tool '{tool_name}' not found. Available: {list(self._tools.keys())}"
            )
        
        tool_def = self._tools[tool_name]
        
        try:
            # Execute the tool
            result = tool_def.func(**kwargs)
            
            # Determine success
            success = result.get("result") in ("success", "pass")
            explanation = result.get("explain", result.get("explanation", ""))
            
            tool_result = ToolResult(
                success=success,
                tool_name=tool_name,
                result=result,
                explanation=explanation,
                source=result.get("provider", tool_name)
            )
            
            # Log execution
            self._execution_history.append({
                "tool": tool_name,
                "params": kwargs,
                "success": success,
                "category": tool_def.category.value
            })
            
            return tool_result
            
        except Exception as e:
            logger.error(f"Tool execution failed: {tool_name} - {e}")
            return ToolResult(
                success=False,
                tool_name=tool_name,
                result={"error": str(e)},
                explanation=f"Tool execution failed: {e}"
            )
    
    def execute_with_llm_response(
        self,
        query: str,
        tool_name: Optional[str] = None,
        **tool_kwargs
    ) -> Dict[str, Any]:
        """
        Execute tool and generate harmonious LLM response.
        
        This combines tool output with LLM explanation for a
        coherent, professional response.
        
        Args:
            query: User's original query
            tool_name: Tool to use (auto-detected if not provided)
            **tool_kwargs: Additional tool parameters
            
        Returns:
            Combined result with tool output and LLM explanation
        """
        # Auto-detect tool if not specified
        if not tool_name:
            tool_name = self.detect_tool_intent(query)
        
        if not tool_name:
            return {
                "success": False,
                "response": "I couldn't determine which tool to use for this query.",
                "suggestion": f"Available tools: {list(self._tools.keys())}"
            }
        
        # Execute the tool
        tool_result = self.execute_tool(tool_name, **tool_kwargs)
        
        # If we have an LLM, generate a harmonious response
        if self._llm and tool_result.success:
            try:
                llm_response = self._generate_llm_response(query, tool_result)
                return {
                    "success": True,
                    "tool_output": tool_result.result,
                    "response": llm_response,
                    "tool_used": tool_name,
                    "source": tool_result.source
                }
            except Exception as e:
                logger.warning(f"LLM response generation failed: {e}")
        
        # Return tool result with basic formatting
        return {
            "success": tool_result.success,
            "tool_output": tool_result.result,
            "response": tool_result.explanation,
            "tool_used": tool_name,
            "source": tool_result.source
        }
    
    def _generate_llm_response(self, query: str, tool_result: ToolResult) -> str:
        """Generate harmonious LLM response from tool output."""
        prompt = f"""You are a Chartered Accountant AI assistant. Based on the user's query and the tool output below, 
provide a clear, professional response that integrates the information.

User Query: {query}

Tool Used: {tool_result.tool_name}
Tool Output: {tool_result.result}

Provide a concise, accurate response that:
1. Directly answers the user's question
2. Cites relevant data from the tool output
3. Adds any necessary disclaimers or context
4. Uses professional CA terminology where appropriate

Response:"""

        response = self._llm.invoke(prompt)
        return str(response) if response else tool_result.explanation
    
    def get_execution_history(self) -> List[Dict]:
        """Get tool execution history for audit."""
        return self._execution_history.copy()


# ============================================================================
# SINGLETON ACCESS
# ============================================================================

_orchestrator: Optional[ToolOrchestrator] = None


def get_tool_orchestrator(llm_wrapper=None) -> ToolOrchestrator:
    """Get or create singleton tool orchestrator."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = ToolOrchestrator(llm_wrapper)
    elif llm_wrapper and _orchestrator._llm is None:
        _orchestrator._llm = llm_wrapper
    return _orchestrator


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def search_and_respond(query: str, llm_wrapper=None) -> Dict[str, Any]:
    """Quick search with LLM-enhanced response."""
    orchestrator = get_tool_orchestrator(llm_wrapper)
    # Note: web_search uses 'query' as parameter, which matches our first arg
    return orchestrator.execute_tool("web_search", query=query, num_results=3)


def check_tax_rate(tax_type: str, country: str = "India", year: int = 2024, llm_wrapper=None) -> Dict[str, Any]:
    """Get tax rate with professional response."""
    orchestrator = get_tool_orchestrator(llm_wrapper)
    return orchestrator.execute_with_llm_response(
        f"What is the {tax_type} tax rate in {country} for {year}?",
        tool_name="search_tax_rate",
        tax_type=tax_type,
        country=country,
        year=year
    )


def detect_fraud(dataset_id: str, column_name: str, llm_wrapper=None) -> Dict[str, Any]:
    """Run fraud detection with professional response."""
    orchestrator = get_tool_orchestrator(llm_wrapper)
    return orchestrator.execute_with_llm_response(
        f"Check for fraud in {column_name} column using Benford's Law",
        tool_name="benford_test",
        dataset_id=dataset_id,
        column_name=column_name
    )


__all__ = [
    "ToolOrchestrator",
    "ToolResult",
    "ToolCategory",
    "get_tool_orchestrator",
    "search_and_respond",
    "check_tax_rate",
    "detect_fraud",
]
