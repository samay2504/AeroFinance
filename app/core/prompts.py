"""
CA Agent Prompts - The "Brain of Logic" for the Agent.
All agent instructions come from this centralized registry.
Supports dynamic prompt templates with YAML override capability.
"""
import logging
from typing import Dict, Any, Optional
from pathlib import Path
import yaml
from langchain_core.prompts import PromptTemplate

logger = logging.getLogger(__name__)

# --- Global CA System Guardrails ---
CA_SYSTEM_GUARDRAILS = """
You are a Senior Chartered Accountant (CA) and Financial Advisor with expertise in:
- Financial Statement Analysis (GAAP/IFRS/Schedule III compliance)
- Management Information Systems (MIS) reporting
- Tax compliance and regulatory requirements
- Forensic accounting and fraud detection

STRICT RULES:
1. ACCURACY: Never guess a number. Use provided data tools for ANY calculation.
2. SOURCE-FIRST: Only answer based on provided data. If data missing, say "Data not found in records".
3. COMPLIANCE: Adhere to standard accounting principles (GAAP/IFRS/Schedule III as applicable).
4. VERIFICATION: If a numeric result cannot be verified, mark it as 'estimate' with disclaimer.
5. OUTPUT FORMAT: Use Markdown. Currency in 'INR Mn' or 'Cr' as per source document's denomination.
"""

# --- Track Classification ---
TRACK_DATA = "TRACK_DATA"  # SQL/Pandas analytics
TRACK_DOC = "TRACK_DOC"    # Document RAG / policy lookup
TRACK_WEB = "TRACK_WEB"    # Web search for latest info

# --- Prompt Templates Registry ---
_PROMPT_TEMPLATES: Dict[str, str] = {}


def _load_yaml_templates() -> Dict[str, str]:
    """Load prompt templates from YAML if available."""
    yaml_path = Path(__file__).parent.parent / "prompts" / "templates.yaml"
    if yaml_path.exists():
        try:
            with open(yaml_path) as f:
                data = yaml.safe_load(f) or {}
                return data.get("templates", {})
        except Exception as e:
            logger.warning(f"Failed to load prompt templates YAML: {e}")
    return {}


# Initialize templates from YAML or use defaults
_PROMPT_TEMPLATES = _load_yaml_templates()


def get_router_prompt(query: str, client_context: str) -> str:
    """
    Classifies intent into: TRACK_DATA, TRACK_DOC, or TRACK_WEB.
    Uses semantic understanding, not keyword matching.
    
    Args:
        query: User's question
        client_context: Context about the client's available datasets
        
    Returns:
        Formatted prompt for router classification
    """
    template_str = _PROMPT_TEMPLATES.get("router", """
{guardrails}

CLIENT/DATA CONTEXT:
{context}

USER QUERY: "{query}"

UNDERSTAND THE USER'S INTENT semantically (not just keywords) and classify into ONE track:

TRACK_DATA: Use when the user wants to:
- Query, analyze, or explore LOADED DATA (spreadsheets, CSV, Excel files)
- Get counts, sums, averages, totals, or any calculations FROM data
- Find specific values, records, or metrics in datasets
- Ask about sheet names, columns, rows, or data structure
- Ask "what is [term]" when that term might exist in loaded data
- Any question that can be answered by looking at tabular data
EXAMPLES: "how many sheets", "what are the sheet names", "total revenue", 
          "what is nifty", "show volume gainers", "list top stocks"

TRACK_DOC: Use when the user wants to:
- Look up definitions, policies, or clauses from DOCUMENTS (PDFs, contracts)
- Find legal terms, compliance text, or regulatory content
- Search through unstructured text documents (not spreadsheets)
- Answer questions about document content that requires reading prose
EXAMPLES: "what does clause 5 say", "find the cancellation policy", 
          "accounting policy for inventory"

TRACK_WEB: Use when the user wants to:
- Get CURRENT, REAL-TIME, or LATEST information from the internet
- Find external data not in loaded files (rates, news, regulations)
- Answer questions requiring up-to-date information
EXAMPLES: "current repo rate", "latest GST rules", "today's market news"

DECISION PRIORITY:
1. If data is loaded AND query seems related to analyzing that data → TRACK_DATA
2. If asking about document content/policies → TRACK_DOC  
3. If needing real-time/external info → TRACK_WEB
4. When in doubt with loaded data → TRACK_DATA

OUTPUT: Return ONLY the track code (TRACK_DATA, TRACK_DOC, or TRACK_WEB).
""")
    
    template = PromptTemplate(
        input_variables=["guardrails", "query", "context"],
        template=template_str
    )
    return template.format(guardrails=CA_SYSTEM_GUARDRAILS, query=query, context=client_context)


def get_data_analyst_sql_prompt(schema_info: str, query: str, available_columns: str = "", data_sample: str = "", semantic_info: str = "") -> str:
    """
    Generates DuckDB SQL for data analysis queries using semantic data understanding.
    
    Args:
        schema_info: Database schema information
        query: User's analytical question
        available_columns: Optional list of available columns
        data_sample: Sample rows from the actual data
        semantic_info: Semantic understanding of the data (period columns, label column, etc.)
        
    Returns:
        Formatted prompt for SQL generation
    """
    template_str = _PROMPT_TEMPLATES.get("data_analyst_sql", """
{guardrails}

DATABASE SCHEMA (DuckDB SQL):
{schema}

{columns_section}

{data_sample_section}

{semantic_section}

CRITICAL DuckDB SQL RULES:
1. Use ONLY the exact column names shown above - they are lowercase with underscores.
2. ALL numeric operations MUST use explicit CAST: CAST(column AS DOUBLE) for calculations.
3. Handle mixed types: Use TRY_CAST(column AS DOUBLE) for columns that might have non-numeric values.
4. String comparisons are case-sensitive - use LOWER() for matching text.
5. For NULL-like values, the data is pre-cleaned but use COALESCE(col, 0) for safety.
6. The table name is the sanitized version shown in schema - use EXACTLY that name.

ANTI-HALLUCINATION RULES (CRITICAL):
- DO NOT invent table names like 'customer_data', 'customers', 'users', 'orders', etc.
- ONLY use the EXACT table name provided in DATABASE SCHEMA above.
- ONLY use column names that EXIST in the schema - DO NOT make up columns.
- If the requested data doesn't exist in the schema, return a query that filters for it.
- NEVER generate CREATE TABLE, DROP, DELETE, INSERT, or UPDATE statements.

QUERY PATTERNS:
- For "growth from X to Y": Calculate (Y_value - X_value) using the appropriate columns
- For "value in period Z": Select the value from column matching period Z
- For aggregations: Use SUM(), AVG(), COUNT() with explicit CAST to DOUBLE
- For row filtering: Use WHERE with the label column to find specific metrics

USER QUERY: {query}

OUTPUT FORMAT:
Return ONLY valid JSON with this structure:
{{
    "sql": "SELECT CAST(... AS DOUBLE) FROM table_name WHERE ...",
    "columns_used": ["col1", "col2"],
    "explanation": "What this query does and how it answers the question"
}}
""")
    
    columns_section = f"AVAILABLE COLUMNS:\n{available_columns}" if available_columns else ""
    data_sample_section = f"DATA SAMPLE (first few rows for context):\n{data_sample}" if data_sample else ""
    semantic_section = f"DATA STRUCTURE UNDERSTANDING:\n{semantic_info}" if semantic_info else ""
    
    template = PromptTemplate(
        input_variables=["guardrails", "schema", "query", "columns_section", "data_sample_section", "semantic_section"],
        template=template_str
    )
    return template.format(
        guardrails=CA_SYSTEM_GUARDRAILS,
        schema=schema_info,
        query=query,
        columns_section=columns_section,
        data_sample_section=data_sample_section,
        semantic_section=semantic_section
    )


def get_data_analyst_python_prompt(schema_info: str, query: str, sample_data: str = "") -> str:
    """
    Generates Python/Pandas code for complex analysis when SQL is insufficient.
    
    Args:
        schema_info: DataFrame schema
        query: User's question
        sample_data: Optional sample of the data
        
    Returns:
        Formatted prompt for Python code generation
    """
    template_str = _PROMPT_TEMPLATES.get("data_analyst_python", """
{guardrails}

DATAFRAME SCHEMA:
{schema}

SAMPLE DATA:
{sample}

USER QUERY: {query}

Generate Python code that:
1. Works with a pandas DataFrame named 'df' (already loaded).
2. Defines a function 'run(df)' that returns the result.
3. Uses only: pd (pandas), np (numpy), math, decimal, datetime - ALREADY AVAILABLE IN SCOPE.
4. Returns a JSON-serializable result (dict, list, number, or string).
5. Handles missing values gracefully with pd.isna() or pd.notna().

CRITICAL RESTRICTIONS (code will be REJECTED if violated):
- DO NOT use import statements - modules are pre-loaded
- DO NOT use __import__, eval(), exec(), open(), or getattr()
- DO NOT access files, networks, or system resources
- DO NOT use globals(), locals(), or vars()
- Use 'pd' not 'pandas', 'np' not 'numpy'

OUTPUT FORMAT:
Return ONLY valid JSON:
{{
    "code": "def run(df):\\n    # your code here\\n    return result",
    "explanation": "Brief explanation of the approach"
}}
""")
    
    template = PromptTemplate(
        input_variables=["guardrails", "schema", "query", "sample"],
        template=template_str
    )
    return template.format(
        guardrails=CA_SYSTEM_GUARDRAILS,
        schema=schema_info,
        query=query,
        sample=sample_data or "Not provided"
    )


def get_ca_synthesis_prompt(results: str, query: str, method: str = "unknown") -> str:
    """
    Final synthesis - converts raw results into professional CA response.
    
    Args:
        results: Raw tool/query results
        query: Original user question
        method: Method used (sql_duckdb, pandas, llm_estimation)
        
    Returns:
        Formatted prompt for response synthesis
    """
    template_str = _PROMPT_TEMPLATES.get("synthesis", """
{guardrails}

RAW RESULTS FROM {method}:
{results}

ORIGINAL QUERY: {query}

INSTRUCTIONS:
1. Summarize findings professionally and concisely.
2. If numeric, present with appropriate precision and units.
3. If variance/anomaly exists, explain in plain terms.
4. Note the data source and calculation method used.
5. Add a brief 'Note' if any assumptions were made.

OUTPUT FORMAT:
Provide a clear, professional response suitable for presentation to management.
If result is uncertain, add a disclaimer.
""")
    
    template = PromptTemplate(
        input_variables=["guardrails", "results", "query", "method"],
        template=template_str
    )
    return template.format(
        guardrails=CA_SYSTEM_GUARDRAILS,
        results=results,
        query=query,
        method=method.upper()
    )


def get_document_rag_prompt(context: str, query: str) -> str:
    """
    Document RAG prompt for policy/clause lookups.
    
    Args:
        context: Retrieved document chunks
        query: User's question
        
    Returns:
        Formatted prompt for document QA
    """
    template_str = _PROMPT_TEMPLATES.get("document_rag", """
{guardrails}

DOCUMENT CONTEXT:
{context}

USER QUERY: {query}

INSTRUCTIONS:
1. Answer based ONLY on the provided context.
2. Quote relevant sections when appropriate.
3. If the answer is not in the context, say "Information not found in available documents".
4. For legal/compliance matters, recommend verification with original source.

OUTPUT: Provide a clear, accurate answer with source references.
""")
    
    template = PromptTemplate(
        input_variables=["guardrails", "context", "query"],
        template=template_str
    )
    return template.format(
        guardrails=CA_SYSTEM_GUARDRAILS,
        context=context,
        query=query
    )


def get_web_search_prompt(search_results: str, query: str) -> str:
    """
    Web search synthesis prompt.
    
    Args:
        search_results: Web search snippets
        query: User's question
        
    Returns:
        Formatted prompt for web result synthesis
    """
    template_str = _PROMPT_TEMPLATES.get("web_search", """
{guardrails}

WEB SEARCH RESULTS:
{results}

USER QUERY: {query}

INSTRUCTIONS:
1. Synthesize information from credible sources.
2. Cite sources with URLs.
3. Note the date relevance of information.
4. For tax rates/regulations, recommend verification with official sources.

OUTPUT: Provide factual answer with sources. Add date caveat if time-sensitive.
""")
    
    template = PromptTemplate(
        input_variables=["guardrails", "results", "query"],
        template=template_str
    )
    return template.format(
        guardrails=CA_SYSTEM_GUARDRAILS,
        results=search_results,
        query=query
    )


def get_dataset_match_prompt(query: str, datasets_info: str) -> str:
    """
    Dataset matching prompt to find the best dataset for a query.
    
    Args:
        query: User's question
        datasets_info: Information about available datasets
        
    Returns:
        Formatted prompt for dataset selection
    """
    template_str = _PROMPT_TEMPLATES.get("dataset_match", """
You are a data matching assistant. Match the user's query to the most relevant dataset.

AVAILABLE DATASETS:
{datasets}

USER QUERY: {query}

RULES:
1. Match based on column names, sheet names, and data descriptions.
2. Look for keywords in the query that match dataset content.
3. Consider time periods mentioned (FY21, 9MFY22, etc.).

OUTPUT FORMAT:
Return ONLY the dataset_id of the best match, or "NONE" if no match found.
""")
    
    template = PromptTemplate(
        input_variables=["datasets", "query"],
        template=template_str
    )
    return template.format(datasets=datasets_info, query=query)


def get_metadata_query_prompt(query: str, available_datasets: str, schema_info: str = "") -> str:
    """
    Generate prompt for understanding and answering metadata queries semantically.
    No hardcoding - LLM determines if this is a metadata query and how to answer.
    
    Args:
        query: User's question
        available_datasets: List of available datasets/sheets
        schema_info: Optional schema information
        
    Returns:
        Formatted prompt for metadata query handling
    """
    template_str = _PROMPT_TEMPLATES.get("metadata_query", """
You are a data assistant. Analyze the user's query and determine if they are asking about 
the STRUCTURE or METADATA of loaded data (not the data values themselves).

METADATA QUERIES include questions about:
- Number of sheets/tables/datasets
- Names of sheets/tables/datasets
- Column names or structure
- Data types or schema
- File information

AVAILABLE DATA:
{datasets}

{schema_section}

USER QUERY: "{query}"

INSTRUCTIONS:
1. Determine if this is a METADATA query (about structure) or a DATA query (about values)
   Matches METADATA: "what are the sheet names", "list all tables", "how many columns", "show me schema"
   Matches DATA (NOT metadata): "summary of sheet X", "calculate total", "show me the data", "value of X", "analyze trends"

2. If it is a DATA query (asking for summary, values, analysis), return "is_metadata_query": false
3. If METADATA query, provide a direct, complete answer
4. Include all relevant information (e.g., ALL sheet names, not just some)
5. Format the answer clearly and professionally

OUTPUT FORMAT:
Return JSON:
{{
    "is_metadata_query": true/false,
    "answer": "Your complete answer if metadata query, or null",
    "query_type": "sheet_count|sheet_names|column_info|schema|other|not_metadata"
}}
""")
    
    schema_section = f"SCHEMA INFORMATION:\n{schema_info}" if schema_info else ""
    
    template = PromptTemplate(
        input_variables=["datasets", "query", "schema_section"],
        template=template_str
    )
    return template.format(datasets=available_datasets, query=query, schema_section=schema_section)


# Tool placeholder descriptions for agent prompts
TOOL_DESCRIPTIONS = {
    "tool_benford": "benford_test: Runs Benford's Law test on numeric column to detect anomalies/fraud",
    "tool_3way": "three_way_match: Compares invoice, PO, and ledger for discrepancies (reconciliation)",
    "tool_cohort": "cohort_analysis: Builds cohort retention analysis from transaction data",
    "tool_web_search": "web_search: Multi-provider web search for current tax rates, regulations, benchmarks",
    "tool_search_tax": "search_tax_rate: Specialized search for tax rates from official sources",
    "tool_search_regulation": "search_regulation: Search for regulatory and compliance requirements",
    "tool_search_benchmark": "search_benchmark: Search for industry benchmarks and market data",
    "tool_date_normalize": "date_normalize: Normalizes various date formats to standard format",
}


def get_tool_description(tool_name: str) -> str:
    """Get description for a tool by name."""
    return TOOL_DESCRIPTIONS.get(tool_name, f"{tool_name}: No description available")


def get_tool_selection_prompt(query: str, available_tools: str = "") -> str:
    """
    Generate prompt for LLM to select appropriate tool.
    
    Args:
        query: User's question
        available_tools: Optional list of available tools
        
    Returns:
        Formatted prompt for tool selection
    """
    if not available_tools:
        available_tools = "\n".join([
            f"- {key}: {desc}" for key, desc in TOOL_DESCRIPTIONS.items()
        ])
    
    template_str = _PROMPT_TEMPLATES.get("tool_selection", """
{guardrails}

USER QUERY: {query}

AVAILABLE TOOLS:
{tools}

DETERMINE:
1. Which tool is most appropriate for this query?
2. What parameters are needed?
3. Can this be answered without a tool (pure data query)?

OUTPUT FORMAT:
Return ONLY valid JSON:
{{
    "tool": "tool_name or null",
    "parameters": {{"param1": "value1"}},
    "reason": "Why this tool was selected"
}}
""")
    
    template = PromptTemplate(
        input_variables=["guardrails", "query", "tools"],
        template=template_str
    )
    return template.format(
        guardrails=CA_SYSTEM_GUARDRAILS,
        query=query,
        tools=available_tools
    )


def get_fraud_analysis_prompt(results: str, column: str, dataset: str) -> str:
    """
    Generate prompt for fraud detection analysis interpretation.
    
    Args:
        results: Benford test results
        column: Column that was tested
        dataset: Dataset ID
        
    Returns:
        Formatted prompt for fraud analysis
    """
    template_str = _PROMPT_TEMPLATES.get("fraud_analysis", """
{guardrails}

BENFORD'S LAW TEST RESULTS:
{results}

TESTED COLUMN: {column}
DATASET: {dataset}

INTERPRET FINDINGS:
1. PASS: Data follows expected Benford distribution - no immediate red flags.
2. WARNING: Minor deviations detected - may warrant further investigation.
3. FAIL: Significant deviation from Benford's Law - potential data manipulation or fraud.

FOR FAILURES:
- Identify which digits deviate most from expected distribution
- Suggest specific line items or date ranges to investigate
- Recommend additional audit procedures

OUTPUT: Professional fraud analysis report suitable for audit committee.
""")
    
    template = PromptTemplate(
        input_variables=["guardrails", "results", "column", "dataset"],
        template=template_str
    )
    return template.format(
        guardrails=CA_SYSTEM_GUARDRAILS,
        results=results,
        column=column,
        dataset=dataset
    )


def get_reconciliation_prompt(results: str) -> str:
    """
    Generate prompt for reconciliation summary interpretation.
    
    Args:
        results: Three-way match results
        
    Returns:
        Formatted prompt for reconciliation report
    """
    template_str = _PROMPT_TEMPLATES.get("reconciliation", """
{guardrails}

THREE-WAY MATCH RESULTS:
{results}

RECONCILIATION SUMMARY:
- Matched: Transactions where Invoice, PO, and Ledger agree within tolerance
- Unmatched: Discrepancies requiring investigation

FOR DISCREPANCIES:
1. Categorize by type (price variance, quantity variance, timing difference)
2. Quantify the financial impact
3. Suggest resolution steps
4. Flag any that require management attention

OUTPUT: Professional reconciliation report suitable for CFO review.
""")
    
    template = PromptTemplate(
        input_variables=["guardrails", "results"],
        template=template_str
    )
    return template.format(
        guardrails=CA_SYSTEM_GUARDRAILS,
        results=results
    )


# Output schema definitions for validation
OUTPUT_SCHEMAS = {
    "sql_response": {
        "sql": str,
        "columns_used": list,
        "explanation": str
    },
    "python_response": {
        "code": str,
        "explanation": str
    },
    "analysis_result": {
        "value": (float, int, str, type(None)),
        "method": str,
        "explain": str,
        "confidence": float
    },
    "tool_selection": {
        "tool": str,
        "parameters": dict,
        "reason": str
    },
    "schema_analysis": {
        "label_column": str,
        "period_columns": dict,
        "header_row": int,
        "data_start_row": int,
        "data_type": str
    }
}


def get_output_schema(schema_name: str) -> Dict[str, Any]:
    """Get expected output schema by name."""
    return OUTPUT_SCHEMAS.get(schema_name, {})


# ============================================================================
# CONVENIENCE FUNCTIONS FOR TOOL INTEGRATION
# ============================================================================

def format_tool_response_prompt(
    tool_name: str,
    tool_output: Dict[str, Any],
    original_query: str
) -> str:
    """
    Format a tool's output into a professional response using LLM.
    
    Args:
        tool_name: Name of the tool that was used
        tool_output: Raw output from the tool
        original_query: User's original question
        
    Returns:
        Formatted prompt for response generation
    """
    import json
    results_str = json.dumps(tool_output, indent=2, default=str)
    
    if tool_name in ("benford_test", "run_benford_test"):
        return get_fraud_analysis_prompt(
            results_str,
            tool_output.get("column", "unknown"),
            tool_output.get("dataset", "unknown")
        )
    elif tool_name == "three_way_match":
        return get_reconciliation_prompt(results_str)
    elif tool_name in ("web_search", "search_tax_rate", "search_regulation", "search_benchmark"):
        return get_web_search_prompt(results_str, original_query)
    else:
        return get_ca_synthesis_prompt(results_str, original_query, tool_name)


__all__ = [
    "CA_SYSTEM_GUARDRAILS",
    "TRACK_DATA",
    "TRACK_DOC", 
    "TRACK_WEB",
    "get_router_prompt",
    "get_data_analyst_sql_prompt",
    "get_data_analyst_python_prompt",
    "get_ca_synthesis_prompt",
    "get_document_rag_prompt",
    "get_web_search_prompt",
    "get_dataset_match_prompt",
    "get_metadata_query_prompt",
    "get_tool_description",
    "get_tool_selection_prompt",
    "get_fraud_analysis_prompt",
    "get_reconciliation_prompt",
    "get_output_schema",
    "format_tool_response_prompt",
    "TOOL_DESCRIPTIONS",
    "OUTPUT_SCHEMAS",
]
