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
    
    Args:
        query: User's question
        client_context: Context about the client's available datasets
        
    Returns:
        Formatted prompt for router classification
    """
    template_str = _PROMPT_TEMPLATES.get("router", """
{guardrails}

CLIENT CONTEXT:
{context}

USER QUERY: {query}

CLASSIFY this query into exactly ONE track:

TRACK_DATA: For trends, growth rates, specific numbers, ledger scrutiny, period comparisons,
            aggregations, or any question requiring calculation from structured data.
            Keywords: what was, growth, total, sum, average, compare, variance, percentage.

TRACK_DOC: For policy definitions, contract clauses, notes to accounts, legal terms,
           compliance requirements, or textual information lookup.
           Keywords: what is the clause, policy, definition, terms, agreement.

TRACK_WEB: For latest tax rates, budget news, external benchmarks, current regulations,
           or any information requiring real-time/external data.
           Keywords: current rate, latest, 2024/2025, budget, external, benchmark.

OUTPUT FORMAT: Return ONLY the track code (TRACK_DATA, TRACK_DOC, or TRACK_WEB) with no explanation.
""")
    
    template = PromptTemplate(
        input_variables=["guardrails", "query", "context"],
        template=template_str
    )
    return template.format(guardrails=CA_SYSTEM_GUARDRAILS, query=query, context=client_context)


def get_data_analyst_sql_prompt(schema_info: str, query: str, available_columns: str = "") -> str:
    """
    Generates DuckDB SQL for data analysis queries.
    
    Args:
        schema_info: Database schema information
        query: User's analytical question
        available_columns: Optional list of available columns
        
    Returns:
        Formatted prompt for SQL generation
    """
    template_str = _PROMPT_TEMPLATES.get("data_analyst_sql", """
{guardrails}

DATABASE SCHEMA (DuckDB SQL):
{schema}

{columns_section}

REQUIREMENTS:
1. Use standard SQL syntax compatible with DuckDB.
2. Handle NULL values: 'Na/p', 'N/A', '-', '' should be treated as NULL.
3. For numeric columns with commas/percentages, values are pre-cleaned.
4. Period columns (fy21, 9mfy22, q1fy23) are preserved as-is for grouping.
5. Use CAST() for type conversions when needed.
6. Aggregate results at highest precision (no rounding unless asked).

USER QUERY: {query}

OUTPUT FORMAT:
Return ONLY valid JSON with this structure:
{{
    "sql": "SELECT ... FROM data ...",
    "columns": ["col1", "col2"],
    "explanation": "Brief explanation of the SQL approach"
}}
""")
    
    columns_section = f"AVAILABLE COLUMNS:\n{available_columns}" if available_columns else ""
    
    template = PromptTemplate(
        input_variables=["guardrails", "schema", "query", "columns_section"],
        template=template_str
    )
    return template.format(
        guardrails=CA_SYSTEM_GUARDRAILS,
        schema=schema_info,
        query=query,
        columns_section=columns_section
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
3. Uses only: pandas, numpy, math, decimal (no other imports).
4. Returns a JSON-serializable result (dict, list, number, or string).
5. Handles missing values gracefully.

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


# Tool placeholder descriptions for agent prompts
TOOL_DESCRIPTIONS = {
    "tool_benford": "benford_test: Runs Benford's Law test on numeric column to detect anomalies",
    "tool_3way": "three_way_match: Compares invoice, PO, and ledger for discrepancies",
    "tool_cohort": "cohort_analysis: Builds cohort retention analysis from transaction data",
    "tool_web_search": "web_search: Searches web for current tax rates, regulations, benchmarks",
    "tool_date_normalize": "date_normalize: Normalizes various date formats to standard format",
}


def get_tool_description(tool_name: str) -> str:
    """Get description for a tool by name."""
    return TOOL_DESCRIPTIONS.get(tool_name, f"{tool_name}: No description available")


# Output schema definitions for validation
OUTPUT_SCHEMAS = {
    "sql_response": {
        "sql": str,
        "columns": list,
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
    }
}


def get_output_schema(schema_name: str) -> Dict[str, Any]:
    """Get expected output schema by name."""
    return OUTPUT_SCHEMAS.get(schema_name, {})
