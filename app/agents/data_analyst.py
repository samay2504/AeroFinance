"""
Data Analyst Agent - SQL-first analytical agent with Pandas fallback.
Handles numeric queries, growth calculations, and period-based analysis.
"""
import logging
import re
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """Result of data analysis."""
    success: bool
    result: Any = None
    value: Optional[float] = None
    method: str = "unknown"
    explanation: str = ""
    error: Optional[str] = None
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class DataAnalystAgent:
    """
    SQL-first data analyst with:
    - Deterministic template matching
    - DuckDB SQL execution
    - LLM SQL generation fallback
    - Pandas fallback for complex operations
    - Sandboxed Python code execution
    """

    def __init__(self, llm_wrapper=None):
        self._llm = llm_wrapper
        self._sql_engine = None
        self._data_registry = None
        self._template_engine = None
        self._sandbox = None
        self.dataframes: Dict[str, pd.DataFrame] = {}
        self._init_components()

    def _init_components(self):
        """Initialize component dependencies."""
        try:
            from app.sql_engine import get_sql_engine
            self._sql_engine = get_sql_engine()
        except Exception as e:
            logger.warning(f"SQL engine unavailable: {e}")

        try:
            from app.core.data_registry import get_data_registry
            self._data_registry = get_data_registry()
        except Exception as e:
            logger.warning(f"Data registry unavailable: {e}")

        try:
            from app.sql_templates import get_template_engine
            self._template_engine = get_template_engine()
        except Exception as e:
            logger.warning(f"Template engine unavailable: {e}")

        try:
            from app.sandbox.sandbox_executor import SandboxExecutor
            self._sandbox = SandboxExecutor()
        except Exception as e:
            logger.warning(f"Sandbox unavailable: {e}")

    def register_dataframe(
        self,
        dataset_id: str,
        df: pd.DataFrame,
        preprocessing_report: Optional[Dict] = None,
        client_id: Optional[str] = None
    ) -> bool:
        """Register DataFrame for analysis."""
        if df is None or df.empty:
            return False

        # Store locally
        self.dataframes[dataset_id] = df

        # Register with SQL engine
        if self._sql_engine:
            self._sql_engine.register_dataframe(dataset_id, df)

        # Register with data registry
        if self._data_registry:
            metadata = {"preprocessing": preprocessing_report} if preprocessing_report else {}
            self._data_registry.register(dataset_id, df, metadata, client_id)

        logger.info(f"Registered dataset: {dataset_id} ({len(df)} rows)")
        return True

    def list_datasets_for_client(self, client_id: str) -> List[Dict[str, Any]]:
        """List datasets for a client."""
        if self._data_registry:
            return self._data_registry.list_for_client(client_id)
        
        # Fallback to local dataframes
        return [
            {"dataset_id": k, "rows": len(v), "columns": list(v.columns)}
            for k, v in self.dataframes.items()
            if client_id in k
        ]

    def match_dataset_by_query(
        self,
        query: str,
        datasets: List[Dict[str, Any]]
    ) -> Optional[str]:
        """Match query to best dataset based on content keywords."""
        if not datasets:
            return None

        query_lower = query.lower()
        
        # Keyword mapping to sheet types
        keyword_map = {
            "income statement": ["income", "statement", "p_l", "pnl"],
            "balance sheet": ["balance", "sheet", "assets"],
            "cashflow": ["cashflow", "cash_flow", "cash"],
            "cashburn": ["cashburn", "cash_burn", "burn"],
            "unit economics": ["unit", "economics", "per_order"],
            "gmv": ["gmv", "merchandise", "value"],
            "revenue": ["income", "revenue", "statement"],
            "collection": ["collection", "prepaid", "recorded"],
            "variance": ["variance", "actual", "budget", "forecasting"],
        }

        best_match = None
        best_score = 0

        for dataset in datasets:
            dataset_id = dataset.get("dataset_id", "")
            columns = dataset.get("columns", [])
            
            score = 0
            dataset_lower = dataset_id.lower()
            columns_lower = " ".join(str(c).lower() for c in columns)

            # Check keywords
            for keyword, patterns in keyword_map.items():
                if keyword in query_lower:
                    for pattern in patterns:
                        if pattern in dataset_lower or pattern in columns_lower:
                            score += 10

            # Direct column match
            for word in query_lower.split():
                if len(word) > 3:
                    if word in columns_lower:
                        score += 5
                    if word in dataset_lower:
                        score += 3

            # Period match
            periods = re.findall(r"fy\d{2}|9mfy\d{2}|q\d", query_lower)
            for period in periods:
                if period in columns_lower:
                    score += 8

            if score > best_score:
                best_score = score
                best_match = dataset_id

        # If no good match, return first dataset
        if best_match is None and datasets:
            best_match = datasets[0].get("dataset_id")

        return best_match

    def _get_dataframe(self, dataset_id: str, client_id: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Get DataFrame by ID."""
        if dataset_id in self.dataframes:
            return self.dataframes[dataset_id]
        
        if self._data_registry:
            return self._data_registry.get(dataset_id, client_id)
        
        return None

    def _get_schema_info(self, df: pd.DataFrame, dataset_id: str) -> Dict[str, Any]:
        """Get schema information for a DataFrame."""
        return {
            "dataset_id": dataset_id,
            "rows": len(df),
            "columns": list(df.columns),
            "dtypes": {str(k): str(v) for k, v in df.dtypes.items()},
            "sample": df.head(3).to_dict() if len(df) > 0 else {}
        }

    def execute_sql_query(
        self,
        query: str,
        df_id: str,
        user_id: str = "default",
        client_id: Optional[str] = None,
        use_cache: bool = True
    ) -> AnalysisResult:
        """
        Execute analytical query using SQL-first approach.
        
        Pipeline:
        1. Try deterministic SQL templates
        2. Try LLM SQL generation
        3. Fall back to Pandas/Python
        """
        df = self._get_dataframe(df_id, client_id)
        if df is None:
            return AnalysisResult(
                success=False,
                error=f"Dataset not found: {df_id}",
                method="error"
            )

        schema = self._get_schema_info(df, df_id)

        # Check for metadata queries first
        meta_result = self._handle_metadata_query(query, df, df_id, client_id)
        if meta_result:
            return meta_result

        # Step 1: Try deterministic template
        template_result = self._try_template_sql(query, df, df_id, schema)
        if template_result and template_result.success:
            return template_result

        # Step 2: Try LLM SQL generation
        if self._llm:
            llm_result = self._try_llm_sql(query, df, df_id, schema)
            if llm_result and llm_result.success:
                return llm_result

        # Step 3: Try LLM Python code generation
        if self._llm and self._sandbox:
            python_result = self._try_llm_python(query, df, df_id, schema)
            if python_result and python_result.success:
                return python_result

        # Step 4: Heuristic Pandas fallback
        heuristic_result = self._try_heuristic_pandas(query, df)
        if heuristic_result and heuristic_result.success:
            return heuristic_result

        return AnalysisResult(
            success=False,
            error="Could not process query with any available method",
            method="exhausted",
            explanation="Tried: template SQL, LLM SQL, LLM Python, heuristic Pandas"
        )

    def _handle_metadata_query(
        self,
        query: str,
        df: pd.DataFrame,
        df_id: str,
        client_id: Optional[str]
    ) -> Optional[AnalysisResult]:
        """Handle metadata queries about sheets/structure."""
        query_lower = query.lower()

        # Sheet count query
        if "how many sheets" in query_lower or "number of sheets" in query_lower:
            datasets = self.list_datasets_for_client(client_id) if client_id else list(self.dataframes.keys())
            count = len(datasets) if isinstance(datasets, list) else len(datasets)
            
            return AnalysisResult(
                success=True,
                result=f"There are {count} sheets available.",
                value=float(count),
                method="metadata",
                explanation="Counted registered datasets"
            )

        # List sheets query
        if "list" in query_lower and "sheet" in query_lower:
            datasets = self.list_datasets_for_client(client_id) if client_id else []
            if isinstance(datasets, list):
                sheet_names = [d.get("dataset_id", d) if isinstance(d, dict) else str(d) for d in datasets]
            else:
                sheet_names = list(self.dataframes.keys())
            
            return AnalysisResult(
                success=True,
                result=f"Available sheets: {', '.join(sheet_names)}",
                method="metadata",
                explanation="Listed all registered sheets"
            )

        return None

    def _try_template_sql(
        self,
        query: str,
        df: pd.DataFrame,
        df_id: str,
        schema: Dict[str, Any]
    ) -> Optional[AnalysisResult]:
        """Try deterministic SQL template matching."""
        if not self._template_engine or not self._sql_engine:
            return None

        try:
            result = self._template_engine.generate_deterministic_sql(query, schema, df_id)
            if not result:
                return None

            sql, method = result
            logger.info(f"Template SQL: {sql[:100]}")

            # Ensure DataFrame is registered
            self._sql_engine.register_dataframe(df_id, df)

            # Execute SQL
            result_df = self._sql_engine.execute_df(sql)
            
            if result_df.empty:
                return None

            # Extract scalar value if single result
            if result_df.shape == (1, 1):
                value = float(result_df.iloc[0, 0])
                return AnalysisResult(
                    success=True,
                    result=value,
                    value=value,
                    method=f"sql_duckdb:{method}",
                    explanation=f"Executed SQL: {sql[:100]}..."
                )
            else:
                return AnalysisResult(
                    success=True,
                    result=result_df.to_dict(),
                    method=f"sql_duckdb:{method}",
                    explanation=f"Returned {len(result_df)} rows"
                )

        except Exception as e:
            logger.warning(f"Template SQL failed: {e}")
            return None

    def _try_llm_sql(
        self,
        query: str,
        df: pd.DataFrame,
        df_id: str,
        schema: Dict[str, Any]
    ) -> Optional[AnalysisResult]:
        """Try LLM-generated SQL."""
        if not self._llm or not self._sql_engine:
            return None

        try:
            from app.core.prompts import get_data_analyst_sql_prompt

            # Get sanitized table name for SQL
            safe_table_name = self._sql_engine.get_safe_table_name(df_id)
            
            # Build schema string with sanitized table name
            schema_str = f"Table: {safe_table_name}\nColumns:\n"
            for col in schema["columns"][:20]:  # Limit columns shown
                dtype = schema["dtypes"].get(col, "unknown")
                schema_str += f"  - {col}: {dtype}\n"

            prompt = get_data_analyst_sql_prompt(schema_str, query)
            
            response = self._llm.invoke_with_structured_output(
                prompt,
                output_schema={"sql": str, "columns": list, "explanation": str}
            )

            if "error" in response or "sql" not in response:
                return None

            sql = response["sql"]
            
            # Validate SQL
            is_valid, error = self._sql_engine.validate_sql(sql)
            if not is_valid:
                logger.warning(f"LLM SQL invalid: {error}")
                return None

            # Ensure DataFrame is registered
            self._sql_engine.register_dataframe(df_id, df)

            # Execute
            result_df = self._sql_engine.execute_df(sql)
            
            if result_df.empty:
                return None

            if result_df.shape == (1, 1):
                value = float(result_df.iloc[0, 0])
                return AnalysisResult(
                    success=True,
                    result=value,
                    value=value,
                    method="sql_duckdb:llm_generated",
                    explanation=response.get("explanation", f"SQL: {sql[:80]}")
                )
            else:
                return AnalysisResult(
                    success=True,
                    result=result_df.to_dict(),
                    method="sql_duckdb:llm_generated",
                    explanation=response.get("explanation", f"Returned {len(result_df)} rows")
                )

        except Exception as e:
            logger.warning(f"LLM SQL failed: {e}")
            return None

    def _try_llm_python(
        self,
        query: str,
        df: pd.DataFrame,
        df_id: str,
        schema: Dict[str, Any]
    ) -> Optional[AnalysisResult]:
        """Try LLM-generated Python code in sandbox."""
        if not self._llm or not self._sandbox:
            return None

        try:
            from app.core.prompts import get_data_analyst_python_prompt

            schema_str = f"DataFrame: {df_id}\nShape: {df.shape}\nColumns: {list(df.columns)}"
            sample = df.head(3).to_string() if len(df) > 0 else ""
            
            prompt = get_data_analyst_python_prompt(schema_str, query, sample)
            
            response = self._llm.invoke_with_structured_output(
                prompt,
                output_schema={"code": str, "explanation": str}
            )

            if "error" in response or "code" not in response:
                return None

            code = response["code"]
            
            # Execute in sandbox
            result = self._sandbox.execute(code, df)
            
            if result.get("success"):
                value = result.get("result")
                if isinstance(value, (int, float)):
                    return AnalysisResult(
                        success=True,
                        result=value,
                        value=float(value),
                        method="pandas:llm_sandbox",
                        explanation=response.get("explanation", "Python code executed")
                    )
                else:
                    return AnalysisResult(
                        success=True,
                        result=value,
                        method="pandas:llm_sandbox",
                        explanation=response.get("explanation", "Python code executed")
                    )

        except Exception as e:
            logger.warning(f"LLM Python failed: {e}")
        
        return None

    def _try_heuristic_pandas(
        self,
        query: str,
        df: pd.DataFrame
    ) -> Optional[AnalysisResult]:
        """Heuristic Pandas operations for common patterns."""
        query_lower = query.lower()

        try:
            # Sum pattern
            if "sum" in query_lower or "total" in query_lower:
                for col in df.columns:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        # Look for column match in query
                        if col.lower() in query_lower or any(w in col.lower() for w in query_lower.split() if len(w) > 3):
                            value = df[col].sum()
                            return AnalysisResult(
                                success=True,
                                result=value,
                                value=float(value),
                                method="pandas:heuristic_sum",
                                explanation=f"Summed column: {col}"
                            )

            # Growth pattern
            if "growth" in query_lower:
                # Find period columns
                period_match = re.findall(r"(fy\d{2}|9mfy\d{2})", query_lower)
                if len(period_match) >= 2:
                    period1 = period_match[0]
                    period2 = period_match[1]
                    
                    # Find matching columns
                    col1 = col2 = None
                    for col in df.columns:
                        if period1 in col.lower():
                            col1 = col
                        if period2 in col.lower():
                            col2 = col
                    
                    if col1 and col2 and pd.api.types.is_numeric_dtype(df[col1]):
                        # Find row with metric
                        for keyword in ["revenue", "sales", "income"]:
                            if keyword in query_lower:
                                for idx, row in df.iterrows():
                                    row_str = str(row.values[0]).lower() if len(row) > 0 else ""
                                    if keyword in row_str:
                                        val1 = row[col1]
                                        val2 = row[col2]
                                        if pd.notna(val1) and pd.notna(val2):
                                            growth = float(val2) - float(val1)
                                            return AnalysisResult(
                                                success=True,
                                                result=round(growth, 2),
                                                value=round(growth, 2),
                                                method="pandas:heuristic_growth",
                                                explanation=f"Growth from {col1} to {col2}"
                                            )

        except Exception as e:
            logger.warning(f"Heuristic Pandas failed: {e}")

        return None


# Singleton instance
_data_analyst: Optional[DataAnalystAgent] = None


def get_data_analyst_agent(llm_wrapper=None) -> DataAnalystAgent:
    """Get or create singleton data analyst agent."""
    global _data_analyst
    if _data_analyst is None:
        _data_analyst = DataAnalystAgent(llm_wrapper)
    elif llm_wrapper and _data_analyst._llm is None:
        _data_analyst._llm = llm_wrapper
    return _data_analyst


__all__ = ["DataAnalystAgent", "AnalysisResult", "get_data_analyst_agent"]
